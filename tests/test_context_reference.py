from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    ContextReference,
    ReferenceMoments,
    array_sha256,
    build_context_reference,
    canonical_sha256,
    common_scale_features,
    context_kernel_actions,
    dense_genetic_kernels,
    exact_same_person_matrix,
    hutchinson_gram,
    kernel_gram,
    load_context_reference,
    rank_revealing_projector,
    reference_moments_after_deleting_groups,
    same_person_ustatistic,
    transfer_reference_gram,
    write_context_reference,
)


def _fixture(
    q: int,
    *,
    seed: int = 700,
    n: int = 14,
    m: int = 11,
    overlapping: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    basis = np.ones((n, q), dtype=np.float64)
    if q > 1:
        basis[:, 1:] = rng.normal(size=(n, q - 1))
    fixed = np.column_stack([np.ones(n), basis[:, 1:], rng.normal(size=n)])
    projector = rank_revealing_projector(fixed)
    if overlapping:
        annotations = np.column_stack(
            [np.linspace(0.2, 1.0, m), np.linspace(1.0, 0.3, m)]
        )
        annotation_names = ("overlap_a", "overlap_b")
    else:
        annotations = np.ones((m, 1), dtype=np.float64)
        annotation_names = ("all",)
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q))
    loo_groups = tuple(f"group:{index // 2}" for index in range(m))
    return {
        "genotype": genotype,
        "basis": basis,
        "fixed": fixed,
        "projector": projector,
        "annotations": annotations,
        "components": components,
        "loo_groups": loo_groups,
    }


def _build_reference(
    fixture: dict[str, object],
    *,
    gram_method: str = "exact",
    gram_probes: np.ndarray | None = None,
    same_person_method: str = "exact",
    variant_probes: np.ndarray | None = None,
    probe_tile_size: int = 3,
) -> ContextReference:
    genotype = np.asarray(fixture["genotype"])
    return build_context_reference(
        genotype=genotype,
        basis=fixture["basis"],
        projector=fixture["projector"],
        annotations=fixture["annotations"],
        component_index=fixture["components"],
        loo_groups=fixture["loo_groups"],
        basis_hash=array_sha256(fixture["basis"]),
        fixed_effect_hash=array_sha256(fixture["fixed"]),
        variant_hash=canonical_sha256({"variants": list(range(genotype.shape[1]))}),
        genotype_scaling="pre_scaled_input",
        gram_method=gram_method,
        gram_probes=gram_probes,
        same_person_method=same_person_method,
        variant_probes=variant_probes,
        probe_tile_size=probe_tile_size,
    )


def _component_masses(
    annotation_masses: np.ndarray, components: ContextComponentIndex
) -> np.ndarray:
    return np.asarray(
        [
            annotation_masses[component.annotation_index]
            for component in components.entries
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_exact_reference_matches_dense_q1_to_q4_with_overlapping_annotations(
    q: int,
) -> None:
    fixture = _fixture(q, seed=700 + q)
    reference = _build_reference(fixture)
    assert isinstance(reference, ContextReference)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    expected_gram = kernel_gram(kernels)
    expected_same_person = exact_same_person_matrix(kernels)
    np.testing.assert_allclose(reference.gram, expected_gram, rtol=3e-14, atol=3e-12)
    np.testing.assert_allclose(
        reference.same_person, expected_same_person, rtol=3e-14, atol=3e-12
    )
    np.testing.assert_allclose(
        reference.annotation_masses,
        np.sum(fixture["annotations"], axis=0),
        rtol=0.0,
        atol=0.0,
    )
    component_masses = _component_masses(
        reference.annotation_masses, fixture["components"]
    )
    reconstructed = np.sum(
        reference.gram_numerator_contributions, axis=0, dtype=np.float64
    ) / np.outer(component_masses, component_masses)
    np.testing.assert_allclose(reconstructed, reference.gram, rtol=4e-14, atol=4e-12)
    assert reference.gram_numerator_contributions.shape == (
        np.asarray(fixture["genotype"]).shape[1],
        len(fixture["components"]),
        len(fixture["components"]),
    )
    assert reference.manifest["feature_mode"] == "raw_projected"
    assert reference.manifest["genotype_scaling"] == "pre_scaled_input"
    assert reference.manifest["component_index_hash"] == fixture["components"].digest


def test_exact_reference_preserves_signed_cross_components_and_both_factors() -> None:
    fixture = _fixture(2, seed=700)
    reference = _build_reference(fixture)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    off_diagonal = fixture["components"].entries[2]
    assert (off_diagonal.q, off_diagonal.r, off_diagonal.kernel_factor) == (0, 1, 2)
    half_cross = 0.5 * kernels[off_diagonal.index]
    assert reference.gram[0, off_diagonal.index] == pytest.approx(
        2.0 * np.sum(kernels[0] * half_cross), rel=3e-14, abs=3e-13
    )
    assert reference.gram[off_diagonal.index, off_diagonal.index] == pytest.approx(
        4.0 * np.sum(half_cross * half_cross), rel=3e-14, abs=3e-13
    )
    diagonal_half = np.diag(half_cross)
    assert reference.same_person[0, off_diagonal.index] == pytest.approx(
        2.0 * np.dot(np.diag(kernels[0]), diagonal_half),
        rel=3e-14,
        abs=3e-13,
    )
    assert reference.same_person[
        off_diagonal.index, off_diagonal.index
    ] == pytest.approx(4.0 * np.dot(diagonal_half, diagonal_half), rel=3e-14, abs=3e-13)
    assert float(np.min(reference.gram)) < -1.0e-6
    assert float(np.min(reference.same_person)) < -1.0e-6


def test_reference_group_deletion_uses_lossless_numerators_and_full_same_person() -> (
    None
):
    fixture = _fixture(3, seed=733, m=12)
    reference = _build_reference(fixture)
    deleted_groups = ["group:1", "group:4"]
    observed = reference_moments_after_deleting_groups(reference, deleted_groups)
    assert isinstance(observed, ReferenceMoments)
    deleted = np.isin(np.asarray(fixture["loo_groups"]), deleted_groups)
    remaining_masses = reference.annotation_masses - np.sum(
        np.asarray(fixture["annotations"])[deleted], axis=0, dtype=np.float64
    )
    remaining_component_masses = _component_masses(
        remaining_masses, fixture["components"]
    )
    expected_gram = np.sum(
        reference.gram_numerator_contributions[~deleted], axis=0, dtype=np.float64
    ) / np.outer(remaining_component_masses, remaining_component_masses)
    np.testing.assert_allclose(
        observed.annotation_masses, remaining_masses, rtol=0, atol=0
    )
    np.testing.assert_allclose(observed.gram, expected_gram, rtol=4e-14, atol=4e-12)
    np.testing.assert_array_equal(observed.same_person, reference.same_person)
    with pytest.raises(ValueError, match="Unknown approximate-LOO group"):
        reference_moments_after_deleting_groups(reference, ["not:a:group"])


def test_fixed_probe_hutchinson_and_ustat_builds_are_tile_invariant() -> None:
    fixture = _fixture(3, seed=751, n=15, m=12)
    rng = np.random.default_rng(752)
    gram_probes = rng.choice([-1.0, 1.0], size=(15, 9))
    variant_probes = rng.choice([-1.0, 1.0], size=(12, 7))
    tiled = _build_reference(
        fixture,
        gram_method="hutchinson",
        gram_probes=gram_probes,
        same_person_method="ustat",
        variant_probes=variant_probes,
        probe_tile_size=2,
    )
    untiled = _build_reference(
        fixture,
        gram_method="hutchinson",
        gram_probes=gram_probes,
        same_person_method="ustat",
        variant_probes=variant_probes,
        probe_tile_size=9,
    )
    actions = context_kernel_actions(
        fixture["genotype"],
        fixture["basis"],
        fixture["projector"].projector,
        fixture["annotations"],
        fixture["components"],
        gram_probes,
    )
    expected_gram = hutchinson_gram(actions)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    expected_same_person = same_person_ustatistic(
        features,
        fixture["annotations"],
        fixture["components"],
        variant_probes,
        probe_tile_size=7,
    )
    np.testing.assert_allclose(tiled.gram, expected_gram, rtol=4e-14, atol=4e-12)
    np.testing.assert_allclose(
        tiled.same_person, expected_same_person, rtol=4e-14, atol=4e-12
    )
    np.testing.assert_allclose(tiled.gram, untiled.gram, rtol=4e-14, atol=4e-12)
    np.testing.assert_allclose(
        tiled.same_person, untiled.same_person, rtol=4e-14, atol=4e-12
    )
    np.testing.assert_allclose(
        tiled.gram_numerator_contributions,
        untiled.gram_numerator_contributions,
        rtol=5e-14,
        atol=5e-12,
    )
    component_masses = _component_masses(tiled.annotation_masses, fixture["components"])
    reconstructed = np.sum(
        tiled.gram_numerator_contributions, axis=0, dtype=np.float64
    ) / np.outer(component_masses, component_masses)
    np.testing.assert_allclose(reconstructed, tiled.gram, rtol=5e-14, atol=5e-12)


def _manual_same_person_ustatistic(
    features: np.ndarray,
    annotations: np.ndarray,
    components: ContextComponentIndex,
    probes: np.ndarray,
) -> np.ndarray:
    masses = np.sum(annotations, axis=0, dtype=np.float64)
    sketches: dict[tuple[int, int], np.ndarray] = {}
    for annotation_index in range(annotations.shape[1]):
        weighted_probes = np.sqrt(annotations[:, annotation_index])[:, None] * probes
        for q in range(features.shape[0]):
            sketches[annotation_index, q] = features[q] @ weighted_probes
    g = np.empty(
        (len(components), features.shape[1], probes.shape[1]), dtype=np.float64
    )
    for component in components.entries:
        left = sketches[component.annotation_index, component.q]
        right = sketches[component.annotation_index, component.r]
        g[component.index] = (
            component.kernel_factor * left * right / masses[component.annotation_index]
        )
    sums = np.sum(g, axis=2, dtype=np.float64)
    result = np.einsum("an,bn->ab", sums, sums, optimize=True)
    result -= np.einsum("anv,bnv->ab", g, g, optimize=True)
    result /= probes.shape[1] * (probes.shape[1] - 1)
    return 0.5 * (result + result.T)


def test_same_person_ustatistic_b2_matches_ordered_pair_formula_and_tiles() -> None:
    fixture = _fixture(2, seed=781, n=9, m=7)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    probes = np.array(
        [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    expected = _manual_same_person_ustatistic(
        features, fixture["annotations"], fixture["components"], probes
    )
    observed = same_person_ustatistic(
        features,
        fixture["annotations"],
        fixture["components"],
        probes,
        probe_tile_size=1,
    )
    untiled = same_person_ustatistic(
        features,
        fixture["annotations"],
        fixture["components"],
        probes,
        probe_tile_size=2,
    )
    np.testing.assert_allclose(observed, expected, rtol=4e-14, atol=4e-12)
    np.testing.assert_allclose(observed, untiled, rtol=4e-14, atol=4e-12)
    assert np.any(observed < 0.0)
    with pytest.raises(ValueError, match="at least two|B.*2|probe"):
        same_person_ustatistic(
            features,
            fixture["annotations"],
            fixture["components"],
            probes[:, :1],
        )


def test_same_person_ustatistic_is_unbiased_and_variance_decreases_with_probes() -> (
    None
):
    fixture = _fixture(2, seed=812, n=9, m=7, overlapping=False)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    target = exact_same_person_matrix(kernels)
    rng = np.random.default_rng(913)
    estimates: dict[int, np.ndarray] = {}
    for probe_count in (2, 16):
        estimates[probe_count] = np.asarray(
            [
                same_person_ustatistic(
                    features,
                    fixture["annotations"],
                    fixture["components"],
                    rng.choice([-1.0, 1.0], size=(7, probe_count)),
                    probe_tile_size=min(4, probe_count),
                )
                for _ in range(120)
            ]
        )
    mean_error_two = np.linalg.norm(np.mean(estimates[2], axis=0) - target) / max(
        1.0, np.linalg.norm(target)
    )
    mean_error_sixteen = np.linalg.norm(np.mean(estimates[16], axis=0) - target) / max(
        1.0, np.linalg.norm(target)
    )
    mse_two = np.mean(np.sum((estimates[2] - target) ** 2, axis=(1, 2)))
    mse_sixteen = np.mean(np.sum((estimates[16] - target) ** 2, axis=(1, 2)))
    assert mean_error_two < 0.12
    assert mean_error_sixteen < 0.04
    assert mse_sixteen < 0.2 * mse_two


def test_hutchinson_gram_is_unbiased_and_variance_decreases_with_probes() -> None:
    fixture = _fixture(2, seed=815, n=10, m=8, overlapping=False)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    target = kernel_gram(kernels)
    rng = np.random.default_rng(916)
    estimates: dict[int, np.ndarray] = {}
    for probe_count in (4, 32):
        estimates[probe_count] = np.asarray(
            [
                hutchinson_gram(
                    np.einsum(
                        "aij,jv->aiv",
                        kernels,
                        rng.choice([-1.0, 1.0], size=(10, probe_count)),
                        optimize=True,
                    )
                )
                for _ in range(180)
            ]
        )
    relative_mean_error = np.linalg.norm(np.mean(estimates[32], axis=0) - target) / max(
        1.0, np.linalg.norm(target)
    )
    mse_four = np.mean(np.sum((estimates[4] - target) ** 2, axis=(1, 2)))
    mse_thirty_two = np.mean(np.sum((estimates[32] - target) ** 2, axis=(1, 2)))
    assert relative_mean_error < 0.025
    assert mse_thirty_two < 0.22 * mse_four


def test_transfer_identity_unequal_n_signed_entries_and_invalid_sizes() -> None:
    gram = np.array([[3.0, -1.2], [-1.2, 2.0]])
    same_person = np.array([[0.5, -0.4], [-0.4, 0.7]])
    np.testing.assert_array_equal(
        transfer_reference_gram(gram, same_person, reference_n=10, study_n=10),
        gram,
    )
    observed = transfer_reference_gram(gram, same_person, reference_n=10, study_n=6)
    expected = (6.0 / 10.0) * same_person + (6.0 * 5.0 / (10.0 * 9.0)) * (
        gram - same_person
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-16)
    assert observed[0, 1] < 0.0
    with pytest.raises(ValueError, match="reference_n"):
        transfer_reference_gram(gram, same_person, reference_n=1, study_n=6)
    with pytest.raises(ValueError, match="study_n"):
        transfer_reference_gram(gram, same_person, reference_n=10, study_n=0)


def test_reference_build_rejects_invalid_methods_probes_and_tile() -> None:
    fixture = _fixture(2, seed=831)
    with pytest.raises(ValueError, match="gram_method"):
        _build_reference(fixture, gram_method="unsupported")
    with pytest.raises(ValueError, match="same_person_method"):
        _build_reference(fixture, same_person_method="unsupported")
    with pytest.raises(ValueError, match="Gram probes"):
        _build_reference(
            fixture,
            gram_method="hutchinson",
            gram_probes=np.ones((3, 2)),
        )
    with pytest.raises(ValueError, match="at least two probes"):
        _build_reference(
            fixture,
            same_person_method="ustat",
            variant_probes=np.ones((11, 1)),
        )
    with pytest.raises(ValueError, match="probe_tile_size|tile"):
        _build_reference(fixture, probe_tile_size=0)


def test_reference_round_trip_is_hash_bound_and_tamper_evident(tmp_path: Path) -> None:
    fixture = _fixture(2, seed=851)
    reference = _build_reference(fixture)
    manifest, arrays = write_context_reference(reference, tmp_path / "reference")
    loaded = load_context_reference(
        manifest, expected={"basis_hash": reference.manifest["basis_hash"]}
    )
    assert isinstance(loaded, ContextReference)
    np.testing.assert_array_equal(loaded.gram, reference.gram)
    np.testing.assert_array_equal(loaded.same_person, reference.same_person)
    np.testing.assert_array_equal(
        loaded.gram_numerator_contributions,
        reference.gram_numerator_contributions,
    )
    assert loaded.component_index.names == reference.component_index.names
    with pytest.raises(ValueError, match="mismatch"):
        load_context_reference(
            manifest, expected={"variant_hash": canonical_sha256({"wrong": True})}
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_context_reference(manifest)
    assert arrays.stat().st_size > 0
