from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import summit.context.transform as transform_module
from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    ContextRankError,
    PhenotypeTransformSpec,
    SimultaneousBandResult,
    TransformTrajectory,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    build_context_transform_summary,
    build_transform_trajectory,
    canonical_sha256,
    classify_scale_trajectory,
    common_scale_features,
    dense_genetic_kernels,
    dense_residual_kernels,
    fit_context_model,
    fit_context_transform_scan,
    jackknife_covariance_from_pseudo_values,
    kernel_rhs,
    load_context_transform_summary,
    project_normalize_phenotype,
    rank_revealing_projector,
    select_simultaneous_band_coordinate,
    simultaneous_trajectory_bands,
    trait_moments_after_deleting_groups,
    write_context_transform_summary,
)


def _fixture(
    *,
    seed: int = 6101,
    n: int = 36,
    m: int = 24,
    annotation_name: str = "all",
) -> dict[str, object]:
    if m % 6:
        raise ValueError("The fixture requires six equal-size LOO groups.")
    rng = np.random.default_rng(seed)
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    context = rng.normal(size=n)
    basis = np.column_stack([np.ones(n), context])
    fixed = np.column_stack([np.ones(n), context, rng.normal(size=n)])
    projector = rank_revealing_projector(fixed)
    latent = 0.35 * context + rng.normal(scale=0.7, size=n)
    original = np.exp(0.35 * latent) + 0.4
    user_column = np.sqrt(original + 0.2)
    annotations = np.ones((m, 1), dtype=np.float64)
    components = ContextComponentIndex(
        (annotation_name,), ContextPairIndex(basis.shape[1])
    )
    groups = tuple(f"group:{index // (m // 6)}" for index in range(m))
    return {
        "genotype": genotype,
        "basis": basis,
        "fixed": fixed,
        "projector": projector,
        "original": original,
        "user_column": user_column,
        "annotations": annotations,
        "components": components,
        "residual_basis": np.ones((n, 1), dtype=np.float64),
        "residual_names": ("residual:constant",),
        "groups": groups,
        "basis_hash": array_sha256(basis),
        "fixed_effect_hash": array_sha256(fixed),
        "variant_hash": canonical_sha256({"variants": list(range(m))}),
    }


def _specs(*, include_invalid: bool = False) -> tuple[PhenotypeTransformSpec, ...]:
    result = [
        PhenotypeTransformSpec("identity", "identity"),
        PhenotypeTransformSpec("log", "log", shift=0.25),
        PhenotypeTransformSpec("box_zero", "box_cox", shift=0.25, box_cox_lambda=0.0),
        PhenotypeTransformSpec("box_one", "box_cox", shift=0.25, box_cox_lambda=1.0),
        PhenotypeTransformSpec("user", "user_supplied", source="prepared"),
    ]
    if include_invalid:
        result.insert(1, PhenotypeTransformSpec("invalid_log", "log", shift=-100.0))
    return tuple(result)


def _common(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "genotype": fixture["genotype"],
        "basis": fixture["basis"],
        "projector": fixture["projector"],
        "annotations": fixture["annotations"],
        "component_index": fixture["components"],
        "basis_hash": fixture["basis_hash"],
        "fixed_effect_hash": fixture["fixed_effect_hash"],
        "variant_hash": fixture["variant_hash"],
        "loo_groups": fixture["groups"],
        "genotype_scaling": "pre_scaled_input",
    }


def _build_reference(fixture: dict[str, object]):
    return build_context_reference(
        **_common(fixture), gram_method="exact", same_person_method="exact"
    )


def _build_transform_summary(
    fixture: dict[str, object],
    *,
    transformations: tuple[PhenotypeTransformSpec, ...] | None = None,
    block_size: int = 5,
    transform_tile_size: int = 2,
    original_phenotype: object | None = None,
    retained_mask: object | None = None,
    user_column: object | None = None,
):
    return build_context_transform_summary(
        **_common(fixture),
        original_phenotype=(
            fixture["original"] if original_phenotype is None else original_phenotype
        ),
        transformations=_specs() if transformations is None else transformations,
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        original_trait="fixture_trait",
        retained_mask=retained_mask,
        user_transforms={
            "prepared": fixture["user_column"] if user_column is None else user_column
        },
        block_size=block_size,
        transform_tile_size=transform_tile_size,
    )


def _transformed_values(
    fixture: dict[str, object], spec: PhenotypeTransformSpec
) -> np.ndarray:
    original = np.asarray(fixture["original"])
    if spec.kind == "identity":
        return original
    if spec.kind == "user_supplied":
        return np.asarray(fixture["user_column"])
    shifted = original + float(spec.shift)
    if spec.kind == "log" or spec.box_cox_lambda == 0.0:
        return np.log(shifted)
    lam = float(spec.box_cox_lambda)
    return np.expm1(lam * np.log(shifted)) / lam


def _single_summary(fixture: dict[str, object], spec: PhenotypeTransformSpec):
    return build_context_trait_summary(
        **_common(fixture),
        phenotype=_transformed_values(fixture, spec),
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        block_size=5,
    )


def _band(
    lower: object,
    upper: object,
    *,
    ids: tuple[str, ...] = ("lambda:0", "lambda:1"),
    coordinate: str = "coordinate",
    critical: float = 2.5,
    confidence: float = 0.95,
    maxima: np.ndarray | None = None,
    declared_ids: tuple[str, ...] = (),
    invalid_ids: tuple[str, ...] = (),
) -> SimultaneousBandResult:
    lower_array = np.asarray(lower, dtype=np.float64).reshape(len(ids), 1)
    upper_array = np.asarray(upper, dtype=np.float64).reshape(len(ids), 1)
    estimates = 0.5 * (lower_array + upper_array)
    standard_errors = np.full_like(estimates, 0.1)
    if maxima is None:
        maxima = np.linspace(0.1, 3.0, 63)
    return SimultaneousBandResult(
        transform_ids=ids,
        coordinate_names=(coordinate,),
        estimates=estimates,
        standard_errors=standard_errors,
        lower=lower_array,
        upper=upper_array,
        critical_value=critical,
        confidence_level=confidence,
        multiplier_max_statistics=np.asarray(maxima, dtype=np.float64),
        status="experimental_equal_group_studentized_multiplier_defined",
        declared_transform_ids=declared_ids,
        invalid_transform_ids=invalid_ids,
    )


def test_transform_spec_round_trip_and_invalid_declarations_fail_closed() -> None:
    for spec in _specs():
        rebuilt = PhenotypeTransformSpec.from_dict(spec.to_dict())
        assert rebuilt == spec
        assert rebuilt.digest == spec.digest
    with pytest.raises(ValueError, match="explicit finite shift"):
        PhenotypeTransformSpec("log", "log")
    with pytest.raises(ValueError, match="explicit finite lambda"):
        PhenotypeTransformSpec("box", "box_cox", shift=1.0)
    with pytest.raises(ValueError, match="source column"):
        PhenotypeTransformSpec("user", "user_supplied")
    with pytest.raises(ValueError, match="Unsupported phenotype transform"):
        PhenotypeTransformSpec("bad", "rank_inverse_normal")


def test_identity_log_boxcox_and_user_batch_equal_separate_trait_summaries() -> None:
    fixture = _fixture()
    batched = _build_transform_summary(fixture)
    assert batched.transform_ids == tuple(spec.transform_id for spec in _specs())
    for index, spec in enumerate(_specs()):
        separate = _single_summary(fixture, spec)
        observed = batched.to_single_summary(spec.transform_id)
        np.testing.assert_allclose(
            observed.genetic_rhs, separate.genetic_rhs, rtol=2e-14, atol=2e-12
        )
        np.testing.assert_allclose(
            observed.residual_rhs, separate.residual_rhs, rtol=2e-14, atol=2e-12
        )
        np.testing.assert_allclose(
            observed.rhs_numerator_contributions,
            separate.rhs_numerator_contributions,
            rtol=2e-14,
            atol=2e-12,
        )
        np.testing.assert_array_equal(observed.genetic_traces, batched.genetic_traces)
        np.testing.assert_array_equal(
            observed.genetic_residual, batched.genetic_residual
        )
        assert observed.manifest["transformation"]["spec"]["transform_id"] == (
            batched.transform_ids[index]
        )

    # Box--Cox lambda zero is the log limit.  Lambda one differs from identity
    # only by a constant when an intercept is in the fixed-effect projector.
    np.testing.assert_allclose(
        batched.genetic_rhs[1], batched.genetic_rhs[2], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        batched.genetic_rhs[0], batched.genetic_rhs[3], rtol=2e-14, atol=2e-12
    )


def test_batched_rhs_matches_direct_dense_kernels_including_off_diagonal_factor() -> (
    None
):
    fixture = _fixture(seed=6102)
    summary = _build_transform_summary(fixture)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    genetic_kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    residual_kernels = dense_residual_kernels(
        fixture["projector"].projector, fixture["residual_basis"]
    )
    for index, spec in enumerate(_specs()):
        y = project_normalize_phenotype(
            _transformed_values(fixture, spec), fixture["projector"]
        )
        np.testing.assert_allclose(
            summary.genetic_rhs[index],
            kernel_rhs(genetic_kernels, y),
            rtol=3e-14,
            atol=3e-12,
        )
        np.testing.assert_allclose(
            summary.residual_rhs[index],
            kernel_rhs(residual_kernels, y),
            rtol=3e-14,
            atol=3e-12,
        )
    off_diagonal = next(
        entry.index
        for entry in fixture["components"].pair_index.entries
        if (entry.q, entry.r) == (0, 1)
    )
    y = project_normalize_phenotype(fixture["original"], fixture["projector"])
    z0 = np.asarray(fixture["genotype"]).T @ (np.asarray(fixture["basis"])[:, 0] * y)
    z1 = np.asarray(fixture["genotype"]).T @ (np.asarray(fixture["basis"])[:, 1] * y)
    expected = 2.0 * np.dot(z0, z1) / np.asarray(fixture["genotype"]).shape[1]
    np.testing.assert_allclose(summary.genetic_rhs[0, off_diagonal], expected)


def test_invalid_transform_keeps_declared_position_and_never_changes_mask() -> None:
    fixture = _fixture(seed=6103)
    original = np.concatenate([[np.nan], np.asarray(fixture["original"]), [np.nan]])
    user = np.concatenate([[np.nan], np.asarray(fixture["user_column"]), [np.nan]])
    mask = np.zeros(original.size, dtype=bool)
    mask[1:-1] = True
    specs = _specs(include_invalid=True)
    summary = _build_transform_summary(
        fixture,
        transformations=specs,
        original_phenotype=original,
        retained_mask=mask,
        user_column=user,
    )
    assert tuple(
        record.spec.transform_id for record in summary.transform_records
    ) == tuple(spec.transform_id for spec in specs)
    assert summary.transform_records[1].status.startswith("invalid_")
    assert summary.transform_ids == tuple(
        spec.transform_id for spec in specs if spec.transform_id != "invalid_log"
    )
    assert summary.manifest["retained_mask"]["sha256"] == array_sha256(mask)
    np.testing.assert_allclose(
        summary.to_single_summary("user").genetic_rhs,
        _single_summary(fixture, _specs()[-1]).genetic_rhs,
        rtol=2e-14,
        atol=2e-12,
    )


def test_transform_tiling_is_invariant_bounded_and_visits_each_block_once() -> None:
    fixture = _fixture(seed=6104)
    one = _build_transform_summary(fixture, block_size=5, transform_tile_size=1)
    three = _build_transform_summary(fixture, block_size=5, transform_tile_size=3)
    np.testing.assert_allclose(
        one.genetic_rhs, three.genetic_rhs, rtol=2e-14, atol=2e-12
    )
    np.testing.assert_allclose(
        one.rhs_numerator_contributions,
        three.rhs_numerator_contributions,
        rtol=2e-14,
        atol=2e-12,
    )
    expected_blocks = math.ceil(np.asarray(fixture["genotype"]).shape[1] / 5)
    assert one.decode_passes == three.decode_passes == 1
    assert one.decoded_blocks == three.decoded_blocks == expected_blocks
    q_count = np.asarray(fixture["basis"]).shape[1]
    assert one.maximum_rhs_columns == q_count
    assert three.maximum_rhs_columns == q_count * 3
    assert (
        one.manifest["context_trait_template"]["backend"]["rhs_layout"]
        == "q_major_transform_minor_within_tile"
    )
    assert (
        three.manifest["context_trait_template"]["backend"]["rhs_layout"]
        == "q_major_transform_minor_within_tile"
    )


def test_transform_validity_and_summaries_are_invariant_to_tiny_phenotype_units() -> (
    None
):
    fixture = _fixture(seed=6118)
    identity = (PhenotypeTransformSpec("identity", "identity"),)
    ordinary = _build_transform_summary(fixture, transformations=identity)
    tiny = _build_transform_summary(
        fixture,
        transformations=identity,
        original_phenotype=np.asarray(fixture["original"]) * 1.0e-12,
    )
    assert ordinary.transform_records[0].valid
    assert tiny.transform_records[0].valid
    assert tiny.transform_records[0].status == "valid"
    np.testing.assert_allclose(
        tiny.genetic_rhs, ordinary.genetic_rhs, rtol=3e-14, atol=3e-12
    )
    np.testing.assert_allclose(
        tiny.residual_rhs, ordinary.residual_rhs, rtol=3e-14, atol=3e-12
    )
    np.testing.assert_allclose(
        tiny.rhs_numerator_contributions,
        ordinary.rhs_numerator_contributions,
        rtol=3e-14,
        atol=3e-12,
    )
    ordinary_ss = ordinary.transform_records[0].diagnostics["phenotype_scale"][
        "projected_sum_squares_before_normalization"
    ]
    tiny_ss = tiny.transform_records[0].diagnostics["phenotype_scale"][
        "projected_sum_squares_before_normalization"
    ]
    np.testing.assert_allclose(tiny_ss, ordinary_ss * 1.0e-24, rtol=3e-14)


def test_contributions_reconstruct_full_and_deleted_rhs_for_every_transform() -> None:
    fixture = _fixture(seed=6105)
    summary = _build_transform_summary(fixture)
    component_masses = np.asarray(
        [
            summary.annotation_masses[entry.annotation_index]
            for entry in summary.component_index.entries
        ]
    )
    np.testing.assert_allclose(
        summary.rhs_numerator_contributions.sum(axis=0) / component_masses[None, :],
        summary.genetic_rhs,
        rtol=2e-15,
        atol=2e-12,
    )
    group = "group:2"
    observed = summary.rhs_after_deleting_groups((group,))
    for index, transform_id in enumerate(summary.transform_ids):
        single = summary.to_single_summary(transform_id)
        deleted = trait_moments_after_deleting_groups(single, (group,))
        expected = np.concatenate([deleted.genetic_rhs, deleted.residual_rhs])
        np.testing.assert_allclose(observed[index], expected, rtol=2e-15, atol=2e-12)


def test_batched_fit_and_every_joint_loo_replicate_equal_separate_fits() -> None:
    fixture = _fixture(seed=6106)
    reference = _build_reference(fixture)
    summary = _build_transform_summary(fixture)
    groups = tuple(dict.fromkeys(str(value) for value in fixture["groups"]))
    fit = fit_context_transform_scan(reference, summary, loo_groups=groups)
    separate = [
        fit_context_model(
            reference, summary.to_single_summary(transform_id), loo_groups=groups
        )
        for transform_id in summary.transform_ids
    ]
    np.testing.assert_allclose(
        fit.coefficients,
        np.stack([item.raw_coefficients for item in separate]),
        rtol=5e-12,
        atol=5e-11,
    )
    np.testing.assert_allclose(
        fit.loo_coefficients,
        np.stack([item.loo_coefficients for item in separate], axis=1),
        rtol=8e-12,
        atol=8e-11,
    )
    expected_pseudo = (
        len(groups) * fit.coefficients[None]
        - (len(groups) - 1.0) * fit.loo_coefficients
    )
    np.testing.assert_allclose(fit.pseudo_values, expected_pseudo, rtol=0.0, atol=0.0)
    centered_loo = fit.loo_coefficients - np.mean(
        fit.loo_coefficients, axis=0, keepdims=True
    )
    expected_covariance = (
        (len(groups) - 1.0)
        / len(groups)
        * (
            centered_loo.reshape(len(groups), -1).T
            @ centered_loo.reshape(len(groups), -1)
        )
    )
    np.testing.assert_allclose(
        fit.joint_covariance(), expected_covariance, rtol=5e-13, atol=5e-11
    )
    assert fit.manifest["solve"]["factorizations"] == 1 + len(groups)
    p_count = fit.coefficients.shape[1]
    assert abs(fit.joint_covariance()[0, 3 * p_count]) > 1.0e-12


def test_transform_fit_requires_all_loo_groups_but_preserves_arbitrary_order() -> None:
    fixture = _fixture(seed=6122)
    summary = _build_transform_summary(
        fixture, transformations=(PhenotypeTransformSpec("identity", "identity"),)
    )
    reference = _build_reference(fixture)
    available = tuple(dict.fromkeys(str(value) for value in fixture["groups"]))

    with pytest.raises(ValueError):
        fit_context_transform_scan(reference, summary, loo_groups=available[:-1])

    reordered = available[::-1]
    fit = fit_context_transform_scan(reference, summary, loo_groups=reordered)
    assert fit.jackknife_groups == reordered


def test_fit_factorizes_once_per_full_or_deleted_matrix_not_per_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(seed=6107)
    reference = _build_reference(fixture)
    summary = _build_transform_summary(fixture)
    groups = tuple(dict.fromkeys(str(value) for value in fixture["groups"]))
    calls = 0
    original = transform_module._rank_diagnostics

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(transform_module, "_rank_diagnostics", counted)
    fit_context_transform_scan(reference, summary, loo_groups=groups)
    assert calls == 1 + len(groups)


def test_transform_reordering_only_permutes_fit_and_joint_covariance() -> None:
    fixture = _fixture(seed=6108)
    reference = _build_reference(fixture)
    specs = (_specs()[0], _specs()[1], _specs()[-1])
    forward = fit_context_transform_scan(
        reference, _build_transform_summary(fixture, transformations=specs)
    )
    reverse = fit_context_transform_scan(
        reference, _build_transform_summary(fixture, transformations=specs[::-1])
    )
    permutation = [
        reverse.transform_ids.index(value) for value in forward.transform_ids
    ]
    np.testing.assert_allclose(
        forward.coefficients, reverse.coefficients[permutation], rtol=5e-12, atol=5e-11
    )
    np.testing.assert_allclose(
        forward.loo_coefficients,
        reverse.loo_coefficients[:, permutation],
        rtol=8e-12,
        atol=8e-11,
    )
    p_count = forward.coefficients.shape[1]
    flat_permutation = np.concatenate(
        [np.arange(index * p_count, (index + 1) * p_count) for index in permutation]
    )
    reverse_covariance = reverse.joint_covariance()
    np.testing.assert_allclose(
        forward.joint_covariance(),
        reverse_covariance[np.ix_(flat_permutation, flat_permutation)],
        rtol=8e-12,
        atol=8e-11,
    )


def test_fit_manifest_mismatch_and_structural_rank_failure_are_not_regularized() -> (
    None
):
    fixture = _fixture(seed=6109)
    reference = _build_reference(fixture)
    mismatched = dict(fixture)
    mismatched["variant_hash"] = canonical_sha256({"wrong": True})
    with pytest.raises(ValueError, match="compatibility mismatch"):
        fit_context_transform_scan(reference, _build_transform_summary(mismatched))

    rank_fixture = _fixture(seed=6110)
    rank_fixture["basis"] = np.ones_like(rank_fixture["basis"])
    rank_fixture["basis_hash"] = array_sha256(rank_fixture["basis"])
    rank_reference = _build_reference(rank_fixture)
    rank_summary = _build_transform_summary(rank_fixture)
    with pytest.raises(ContextRankError, match="not identifiable"):
        fit_context_transform_scan(rank_reference, rank_summary)


def test_derived_trajectory_recomputes_every_loo_and_fails_closed_on_domain() -> None:
    fixture = _fixture(seed=6111)
    fit = fit_context_transform_scan(
        _build_reference(fixture), _build_transform_summary(fixture)
    )

    def coordinates(row: np.ndarray) -> np.ndarray:
        return np.asarray([row[0] - row[1], row[-1] * row[-1]])

    trajectory = build_transform_trajectory(
        fit, coordinates, coordinate_names=("contrast", "residual_square")
    )
    expected = np.stack([coordinates(row) for row in fit.coefficients])
    expected_loo = np.stack(
        [
            np.stack([coordinates(row) for row in fit.loo_coefficients[group]])
            for group in range(fit.loo_coefficients.shape[0])
        ]
    )
    np.testing.assert_allclose(trajectory.estimates, expected)
    np.testing.assert_allclose(trajectory.loo_estimates, expected_loo)
    with pytest.raises(ValueError, match="finite"):
        build_transform_trajectory(fit, lambda row: np.asarray([np.nan]))


def test_pseudo_value_covariance_matches_equal_group_formula_and_permutation() -> None:
    rng = np.random.default_rng(6112)
    groups, transforms, parameters = 8, 4, 3
    full = rng.normal(size=(transforms, parameters))
    loo = rng.normal(size=(groups, transforms, parameters))
    pseudo = groups * full[None] - (groups - 1.0) * loo
    centered = loo - np.mean(loo, axis=0, keepdims=True)
    expected = (
        (groups - 1.0)
        / groups
        * (centered.reshape(groups, -1).T @ centered.reshape(groups, -1))
    )
    observed = jackknife_covariance_from_pseudo_values(pseudo)
    np.testing.assert_allclose(observed, expected, rtol=8e-15, atol=8e-14)
    permutation = np.asarray([2, 0, 3, 1])
    flat = np.concatenate(
        [
            np.arange(index * parameters, (index + 1) * parameters)
            for index in permutation
        ]
    )
    permuted = jackknife_covariance_from_pseudo_values(pseudo[:, permutation])
    np.testing.assert_allclose(permuted, observed[np.ix_(flat, flat)])


def test_fit_and_trajectory_standard_errors_do_not_eagerly_build_joint_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(seed=6122)
    rng = np.random.default_rng(6122)
    fixture["annotations"] = rng.uniform(0.2, 1.5, size=(24, 3))
    fixture["components"] = ContextComponentIndex(
        ("annotation:0", "annotation:1", "annotation:2"), ContextPairIndex(2)
    )
    specs = tuple(
        PhenotypeTransformSpec(
            f"lambda:{index}",
            "box_cox",
            shift=0.25,
            box_cox_lambda=float(value),
        )
        for index, value in enumerate(np.linspace(-1.0, 1.5, 12))
    )

    def forbidden_eager_covariance(*args, **kwargs):
        raise AssertionError("joint covariance must remain opt-in")

    monkeypatch.setattr(
        transform_module,
        "jackknife_covariance_from_pseudo_values",
        forbidden_eager_covariance,
    )
    fit = fit_context_transform_scan(
        _build_reference(fixture),
        _build_transform_summary(fixture, transformations=specs),
    )
    trajectory = build_transform_trajectory(fit)

    assert fit.coefficients.shape == (12, 10)
    assert fit.standard_errors.shape == fit.coefficients.shape
    assert trajectory.standard_errors.shape == fit.coefficients.shape
    with pytest.raises(AssertionError, match="opt-in"):
        fit.joint_covariance()


def test_multiplier_max_statistic_has_exact_recentered_studentization_and_is_equivariant() -> (
    None
):
    rng = np.random.default_rng(6113)
    group_count, transform_count, coordinate_count = 12, 3, 2
    pseudo = rng.normal(size=(group_count, transform_count, coordinate_count))
    estimates = rng.normal(size=(transform_count, coordinate_count))
    covariance = jackknife_covariance_from_pseudo_values(pseudo)
    standard_errors = np.sqrt(np.diag(covariance)).reshape(estimates.shape)
    trajectory = TransformTrajectory(
        transform_ids=tuple(f"lambda:{index}" for index in range(transform_count)),
        coordinate_names=("amplification", "heterogeneity"),
        estimates=estimates,
        loo_estimates=(group_count * estimates[None] - pseudo) / (group_count - 1.0),
        pseudo_values=pseudo,
        standard_errors=standard_errors,
    )
    draws, seed = 2048, 19
    observed = simultaneous_trajectory_bands(trajectory, draws=draws, seed=seed)
    centered = pseudo - np.mean(pseudo, axis=0, keepdims=True)
    signs = np.random.default_rng(seed).integers(
        0, 2, size=(draws, group_count), dtype=np.int8
    )
    signs = signs.astype(np.float64) * 2.0 - 1.0
    multiplier_means = (
        np.einsum("bg,gld->bld", signs, centered, optimize=True) / group_count
    )
    centered_sum_squares = np.sum(centered * centered, axis=0)
    bootstrap_sum_squares = np.maximum(
        centered_sum_squares[None] - group_count * multiplier_means * multiplier_means,
        0.0,
    )
    bootstrap_standard_errors = np.sqrt(
        bootstrap_sum_squares / (group_count * (group_count - 1.0))
    )
    expected_maxima = np.max(
        np.abs(multiplier_means) / bootstrap_standard_errors, axis=(1, 2)
    )
    np.testing.assert_allclose(observed.multiplier_max_statistics, expected_maxima)
    assert observed.status == "experimental_equal_group_studentized_multiplier_defined"

    permutation = np.asarray([2, 0, 1])
    scales = np.asarray([-3.0, 0.25])[None, :]
    equivariant = TransformTrajectory(
        transform_ids=tuple(trajectory.transform_ids[index] for index in permutation),
        coordinate_names=trajectory.coordinate_names,
        estimates=estimates[permutation] * scales,
        loo_estimates=(
            group_count * (estimates[permutation] * scales)[None]
            - pseudo[:, permutation] * scales
        )
        / (group_count - 1.0),
        pseudo_values=pseudo[:, permutation] * scales,
        standard_errors=standard_errors[permutation] * np.abs(scales),
    )
    transformed = simultaneous_trajectory_bands(equivariant, draws=draws, seed=seed)
    np.testing.assert_allclose(
        transformed.multiplier_max_statistics, observed.multiplier_max_statistics
    )
    assert transformed.critical_value == observed.critical_value


def test_multiplier_rejects_pseudo_values_inconsistent_with_full_and_loo() -> None:
    rng = np.random.default_rng(6123)
    group_count = 8
    estimates = rng.normal(size=(2, 1))
    loo = rng.normal(size=(group_count, 2, 1))
    pseudo = group_count * estimates[None] - (group_count - 1.0) * loo
    covariance = jackknife_covariance_from_pseudo_values(pseudo)
    standard_errors = np.sqrt(np.diag(covariance)).reshape(estimates.shape)
    inconsistent_loo = loo.copy()
    inconsistent_loo[0, 0, 0] += 1.0
    trajectory = TransformTrajectory(
        transform_ids=("lambda:0", "lambda:1"),
        coordinate_names=("amplification",),
        estimates=estimates,
        loo_estimates=inconsistent_loo,
        pseudo_values=pseudo,
        standard_errors=standard_errors,
    )

    with pytest.raises(ValueError, match="(?i)pseudo|leave|loo|identity|consistent"):
        simultaneous_trajectory_bands(trajectory, draws=63, seed=1)


@pytest.mark.parametrize(
    "transform_ids, declared_ids, invalid_ids",
    [
        (("observed",), ("observed", "silently_missing"), ()),
        (("second", "first"), ("first", "invalid", "second"), ("invalid",)),
        (
            ("observed",),
            ("invalid:0", "observed", "invalid:1"),
            ("invalid:1", "invalid:0"),
        ),
    ],
)
def test_multiplier_requires_declared_order_to_partition_valid_and_invalid_transforms(
    transform_ids: tuple[str, ...],
    declared_ids: tuple[str, ...],
    invalid_ids: tuple[str, ...],
) -> None:
    rng = np.random.default_rng(6124)
    group_count = 8
    estimates = rng.normal(size=(len(transform_ids), 1))
    loo = rng.normal(size=(group_count, len(transform_ids), 1))
    pseudo = group_count * estimates[None] - (group_count - 1.0) * loo
    standard_errors = np.sqrt(
        np.diag(jackknife_covariance_from_pseudo_values(pseudo))
    ).reshape(estimates.shape)
    trajectory = TransformTrajectory(
        transform_ids=transform_ids,
        coordinate_names=("amplification",),
        estimates=estimates,
        loo_estimates=loo,
        pseudo_values=pseudo,
        standard_errors=standard_errors,
        declared_transform_ids=declared_ids,
        invalid_transform_ids=invalid_ids,
    )

    with pytest.raises(ValueError, match="(?i)declared|transform|order|partition"):
        simultaneous_trajectory_bands(trajectory, draws=63, seed=1)


def test_degenerate_studentized_multiplier_critical_is_indeterminate() -> None:
    pseudo = np.asarray(
        [
            [[-1.0, -1.0]],
            [[-1.0, -1.0]],
            [[-1.0, 1.0]],
            [[1.0, -1.0]],
            [[1.0, 1.0]],
            [[1.0, 1.0]],
        ]
    )
    group_count = pseudo.shape[0]
    estimates = np.zeros((1, 2))
    loo = (group_count * estimates[None] - pseudo) / (group_count - 1.0)
    standard_errors = np.sqrt(
        np.diag(jackknife_covariance_from_pseudo_values(pseudo))
    ).reshape(estimates.shape)
    trajectory = TransformTrajectory(
        transform_ids=("identity",),
        coordinate_names=("amplification", "heterogeneity"),
        estimates=estimates,
        loo_estimates=loo,
        pseudo_values=pseudo,
        standard_errors=standard_errors,
        declared_transform_ids=("identity",),
    )

    bands = simultaneous_trajectory_bands(trajectory, draws=4096, seed=913)
    assert bands.status.startswith("indeterminate")
    assert not np.isfinite(bands.critical_value)


def test_classification_propagates_indeterminate_zero_information_joint_band() -> None:
    group_count = 6
    pseudo = np.zeros((group_count, 1, 2))
    trajectory = TransformTrajectory(
        transform_ids=("identity",),
        coordinate_names=("amplification", "heterogeneity"),
        estimates=np.zeros((1, 2)),
        loo_estimates=np.zeros_like(pseudo),
        pseudo_values=pseudo,
        standard_errors=np.zeros((1, 2)),
        declared_transform_ids=("identity",),
    )
    joint = simultaneous_trajectory_bands(trajectory, draws=63, seed=1)
    assert joint.status.startswith("indeterminate")

    result = classify_scale_trajectory(
        select_simultaneous_band_coordinate(joint, "amplification"),
        select_simultaneous_band_coordinate(joint, "heterogeneity"),
        amplification_margin=0.1,
        heterogeneity_margin=0.1,
    )
    assert result.classification == "indeterminate"
    assert result.indeterminate_transform_ids == ("identity",)


@pytest.mark.parametrize(
    "pseudo_values, standard_errors",
    [
        (np.ones((1, 2, 1)), np.ones((2, 1))),
        (np.ones((6, 1, 1)), np.ones((2, 1))),
        (np.full((6, 2, 1), np.nan), np.ones((2, 1))),
        (np.ones((6, 2, 1)), np.full((2, 1), 7.0)),
    ],
)
def test_multiplier_rejects_malformed_or_internally_inconsistent_trajectory(
    pseudo_values: np.ndarray, standard_errors: np.ndarray
) -> None:
    trajectory = TransformTrajectory(
        transform_ids=("lambda:0", "lambda:1"),
        coordinate_names=("amplification",),
        estimates=np.zeros((2, 1)),
        loo_estimates=np.zeros((6, 2, 1)),
        pseudo_values=pseudo_values,
        standard_errors=standard_errors,
    )
    with pytest.raises(
        ValueError, match="(?i)trajectory|pseudo|standard|group|shape|finite"
    ):
        simultaneous_trajectory_bands(trajectory, draws=63, seed=1)


def test_multiplier_single_coordinate_has_reasonable_moderate_group_null_coverage() -> (
    None
):
    rng = np.random.default_rng(6114)
    group_count = 48
    covered = 0
    repetitions = 192
    for repetition in range(repetitions):
        pseudo = rng.normal(size=(group_count, 1, 1))
        estimate = np.mean(pseudo, axis=0)
        covariance = jackknife_covariance_from_pseudo_values(pseudo)
        standard_error = np.sqrt(np.diag(covariance)).reshape(1, 1)
        loo = (group_count * estimate[None] - pseudo) / (group_count - 1.0)
        trajectory = TransformTrajectory(
            transform_ids=("identity",),
            coordinate_names=("mean",),
            estimates=estimate,
            loo_estimates=loo,
            pseudo_values=pseudo,
            standard_errors=standard_error,
        )
        band = simultaneous_trajectory_bands(
            trajectory, draws=511, seed=10_000 + repetition
        )
        covered += int(band.lower[0, 0] <= 0.0 <= band.upper[0, 0])
    coverage = covered / repetitions
    assert 0.90 <= coverage <= 0.99


def test_recentered_studentization_repairs_multicoordinate_moderate_group_coverage() -> (
    None
):
    rng = np.random.default_rng(6121)
    group_count = 8
    coordinate_count = 10
    repetitions = 192
    draws = 511
    repaired_covered = 0
    fixed_se_covered = 0
    for repetition in range(repetitions):
        pseudo = rng.normal(size=(group_count, 1, coordinate_count))
        estimate = np.mean(pseudo, axis=0)
        covariance = jackknife_covariance_from_pseudo_values(pseudo)
        standard_errors = np.sqrt(np.diag(covariance)).reshape(estimate.shape)
        loo = (group_count * estimate[None] - pseudo) / (group_count - 1.0)
        trajectory = TransformTrajectory(
            transform_ids=("identity",),
            coordinate_names=tuple(
                f"coordinate:{index}" for index in range(coordinate_count)
            ),
            estimates=estimate,
            loo_estimates=loo,
            pseudo_values=pseudo,
            standard_errors=standard_errors,
        )
        seed = 20_000 + repetition
        repaired = simultaneous_trajectory_bands(trajectory, draws=draws, seed=seed)
        repaired_covered += int(
            np.all((repaired.lower <= 0.0) & (repaired.upper >= 0.0))
        )

        centered = pseudo - np.mean(pseudo, axis=0, keepdims=True)
        signs = np.random.default_rng(seed).integers(
            0, 2, size=(draws, group_count), dtype=np.int8
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        multiplier_means = (
            np.einsum("bg,gld->bld", signs, centered, optimize=True) / group_count
        )
        fixed_maxima = np.max(
            np.abs(multiplier_means) / standard_errors[None], axis=(1, 2)
        )
        fixed_critical = np.quantile(fixed_maxima, 0.95, method="higher")
        fixed_se_covered += int(
            np.max(np.abs(estimate) / standard_errors) <= fixed_critical
        )

    repaired_coverage = repaired_covered / repetitions
    fixed_se_coverage = fixed_se_covered / repetitions
    assert 0.88 <= repaired_coverage <= 0.99
    assert fixed_se_coverage < 0.82
    assert repaired_coverage - fixed_se_coverage > 0.15


def test_coordinate_selector_preserves_one_joint_multiplier_calibration() -> None:
    fixture = _fixture(seed=6119)
    fit = fit_context_transform_scan(
        _build_reference(fixture), _build_transform_summary(fixture)
    )
    trajectory = build_transform_trajectory(
        fit,
        lambda row: np.asarray([row[0] - row[1], row[1] * row[1]]),
        coordinate_names=("amplification", "heterogeneity"),
    )
    joint = simultaneous_trajectory_bands(trajectory, draws=1023, seed=73)
    amplification = select_simultaneous_band_coordinate(joint, "amplification")
    heterogeneity = select_simultaneous_band_coordinate(joint, 1)
    for selected in (amplification, heterogeneity):
        assert selected.transform_ids == joint.transform_ids
        assert selected.critical_value == joint.critical_value
        assert selected.confidence_level == joint.confidence_level
        assert selected.status == joint.status
        assert selected.declared_transform_ids == joint.declared_transform_ids
        assert selected.invalid_transform_ids == joint.invalid_transform_ids
        np.testing.assert_array_equal(
            selected.multiplier_max_statistics, joint.multiplier_max_statistics
        )
    np.testing.assert_array_equal(amplification.lower[:, 0], joint.lower[:, 0])
    np.testing.assert_array_equal(heterogeneity.upper[:, 0], joint.upper[:, 1])
    # The two selected objects remain acceptable as one jointly calibrated family.
    classify_scale_trajectory(
        amplification,
        heterogeneity,
        amplification_margin=10.0,
        heterogeneity_margin=10.0,
    )


def test_invalid_declared_transform_propagates_and_prevents_robust_family_claim() -> (
    None
):
    fixture = _fixture(seed=6120)
    specs = _specs(include_invalid=True)
    summary = _build_transform_summary(fixture, transformations=specs)
    fit = fit_context_transform_scan(_build_reference(fixture), summary)
    declared = tuple(spec.transform_id for spec in specs)
    assert (
        tuple(record.spec.transform_id for record in fit.transform_records) == declared
    )
    assert fit.manifest["declared_transform_ids"] == list(declared)
    assert fit.manifest["invalid_transform_ids"] == ["invalid_log"]

    trajectory = build_transform_trajectory(
        fit,
        lambda row: np.asarray([row[0], row[1] * row[1]]),
        coordinate_names=("amplification", "heterogeneity"),
    )
    assert trajectory.declared_transform_ids == declared
    assert trajectory.invalid_transform_ids == ("invalid_log",)
    joint = simultaneous_trajectory_bands(trajectory, draws=511, seed=81)
    assert joint.declared_transform_ids == declared
    assert joint.invalid_transform_ids == ("invalid_log",)

    # Decisive non-equivalence at every analyzable scale cannot establish
    # robustness over a declared family containing an unavailable scale.
    valid_ids = summary.transform_ids
    maxima = np.linspace(0.1, 3.0, 63)
    amplification = _band(
        np.full(len(valid_ids), 0.5),
        np.full(len(valid_ids), 0.8),
        ids=valid_ids,
        coordinate="amplification",
        maxima=maxima,
        declared_ids=declared,
        invalid_ids=("invalid_log",),
    )
    heterogeneity = _band(
        np.full(len(valid_ids), 0.3),
        np.full(len(valid_ids), 0.6),
        ids=valid_ids,
        coordinate="heterogeneity",
        maxima=maxima,
        declared_ids=declared,
        invalid_ids=("invalid_log",),
    )
    result = classify_scale_trajectory(
        amplification,
        heterogeneity,
        amplification_margin=0.2,
        heterogeneity_margin=0.1,
    )
    assert result.classification == "indeterminate"
    assert result.indeterminate_transform_ids == ("invalid_log",)


def test_scale_classification_requires_same_scale_equivalence_and_not_nonsignificance() -> (
    None
):
    removable = classify_scale_trajectory(
        _band([-0.1, 0.7], [0.1, 0.9], coordinate="amplification"),
        _band([-0.05, 0.2], [0.05, 0.4], coordinate="heterogeneity"),
        amplification_margin=0.2,
        heterogeneity_margin=0.1,
    )
    assert removable.classification == "removable_within_declared_family"
    assert removable.removable_transform_ids == ("lambda:0",)

    robust = classify_scale_trajectory(
        _band([-0.1, 0.5], [0.1, 0.7], coordinate="amplification"),
        _band([0.2, -0.05], [0.4, 0.05], coordinate="heterogeneity"),
        amplification_margin=0.2,
        heterogeneity_margin=0.1,
    )
    assert robust.classification == "robust_within_declared_family"
    assert robust.removable_transform_ids == ()

    weak = classify_scale_trajectory(
        _band([-0.5, -0.5], [0.5, 0.5], coordinate="amplification"),
        _band([-0.3, -0.3], [0.3, 0.3], coordinate="heterogeneity"),
        amplification_margin=0.2,
        heterogeneity_margin=0.1,
    )
    assert weak.classification == "indeterminate"


def test_scale_classification_rejects_separately_calibrated_mechanism_bands() -> None:
    amplification = _band(
        [-0.1, -0.1],
        [0.1, 0.1],
        coordinate="amplification",
        critical=2.1,
        maxima=np.linspace(0.1, 2.8, 63),
    )
    heterogeneity = _band(
        [-0.05, -0.05],
        [0.05, 0.05],
        coordinate="heterogeneity",
        critical=2.7,
        maxima=np.linspace(0.2, 3.4, 63),
    )
    with pytest.raises(ValueError, match="joint|simultaneous"):
        classify_scale_trajectory(
            amplification,
            heterogeneity,
            amplification_margin=0.2,
            heterogeneity_margin=0.1,
        )


def test_negative_nonnegative_heterogeneity_band_is_indeterminate_not_robust() -> None:
    result = classify_scale_trajectory(
        _band([-0.4, -0.4], [0.4, 0.4], coordinate="amplification"),
        _band([-0.5, -0.6], [-0.2, -0.3], coordinate="heterogeneity"),
        amplification_margin=0.2,
        heterogeneity_margin=0.1,
    )
    assert result.classification == "indeterminate"
    assert result.robust_transform_ids == ()


def test_transform_summary_round_trip_is_hash_bound_and_contains_no_phenotype_rows(
    tmp_path: Path,
) -> None:
    fixture = _fixture(seed=6115)
    summary = _build_transform_summary(fixture)
    manifest, arrays = write_context_transform_summary(summary, tmp_path / "scan")
    loaded = load_context_transform_summary(
        manifest,
        expected_transform_manifest_hash=summary.manifest["transform_manifest_hash"],
    )
    assert loaded.transform_records == summary.transform_records
    np.testing.assert_array_equal(loaded.genetic_rhs, summary.genetic_rhs)
    np.testing.assert_array_equal(
        loaded.rhs_numerator_contributions, summary.rhs_numerator_contributions
    )
    with np.load(arrays, allow_pickle=False) as payload:
        assert not any("phenotype" in name for name in payload.files)

    bytes_value = bytearray(arrays.read_bytes())
    bytes_value[-1] ^= 1
    arrays.write_bytes(bytes_value)
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_context_transform_summary(manifest)


def test_transform_manifest_tampering_is_detected_even_when_artifact_is_unchanged(
    tmp_path: Path,
) -> None:
    fixture = _fixture(seed=6116)
    summary = _build_transform_summary(fixture)
    manifest, _ = write_context_transform_summary(summary, tmp_path / "scan")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["transformations"][0]["spec"]["transform_id"] = "tampered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|mismatch"):
        load_context_transform_summary(
            manifest,
            expected_transform_manifest_hash=summary.manifest[
                "transform_manifest_hash"
            ],
        )


def test_transform_summary_load_rejects_rehashed_nested_spec_digest_tampering(
    tmp_path: Path,
) -> None:
    fixture = _fixture(seed=6125)
    summary = _build_transform_summary(fixture)
    manifest, _ = write_context_transform_summary(summary, tmp_path / "scan")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["transformations"][0]["spec_hash"] = "0" * 64
    clean_manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact", "performance"}
    }
    payload["transform_manifest_hash"] = canonical_sha256(
        {
            key: value
            for key, value in clean_manifest.items()
            if key != "transform_manifest_hash"
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest-bound"):
        load_context_transform_summary(manifest)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("negative_annotation", "non-negative"),
        ("annotation_hash", "annotation identity"),
        ("loo_grouping_hash", "LOO grouping identity"),
    ],
)
def test_transform_summary_load_rejects_annotation_and_group_identity_tampering(
    tmp_path: Path, mutation: str, error: str
) -> None:
    fixture = _fixture(seed=6123)
    summary = _build_transform_summary(fixture)
    manifest, arrays = write_context_transform_summary(
        summary, tmp_path / mutation / "scan"
    )
    with np.load(arrays, allow_pickle=False) as payload:
        values = {name: np.array(payload[name], copy=True) for name in payload.files}

    if mutation == "negative_annotation":
        values["annotation_weights"][0, 0] = -1.0
        values["annotation_weights"][1, 0] = 3.0
    elif mutation == "annotation_hash":
        values["annotation_weights"][0, 0] = 0.5
        values["annotation_weights"][1, 0] = 1.5
    else:
        values["loo_group_ids"][0] = "tampered-group"
    np.savez_compressed(arrays, **values)

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(arrays.read_bytes(), dtype=np.uint8)
    )
    load_kwargs = {
        "expected_transform_manifest_hash": summary.manifest["transform_manifest_hash"]
    }
    if mutation in {"negative_annotation", "annotation_hash"}:
        manifest_payload["array_hashes"]["annotation_weights"] = array_sha256(
            values["annotation_weights"]
        )
        clean_manifest = {
            key: value
            for key, value in manifest_payload.items()
            if key not in {"artifact", "performance", "transform_manifest_hash"}
        }
        manifest_payload["transform_manifest_hash"] = canonical_sha256(clean_manifest)
        load_kwargs = {}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_context_transform_summary(manifest, **load_kwargs)


def test_expected_transform_manifest_hash_binds_numeric_rows_to_transform_ids(
    tmp_path: Path,
) -> None:
    fixture = _fixture(seed=6126)
    summary = _build_transform_summary(fixture)
    manifest, arrays = write_context_transform_summary(summary, tmp_path / "scan")
    with np.load(arrays, allow_pickle=False) as payload:
        values = {name: np.array(payload[name], copy=True) for name in payload.files}

    for name in ("genetic_rhs", "residual_rhs"):
        values[name][[0, 1]] = values[name][[1, 0]]
    values["rhs_numerator_contributions"][:, [0, 1]] = values[
        "rhs_numerator_contributions"
    ][:, [1, 0]]
    np.savez_compressed(arrays, **values)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(arrays.read_bytes(), dtype=np.uint8)
    )
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest-bound content hashes"):
        load_context_transform_summary(
            manifest,
            expected_transform_manifest_hash=summary.manifest[
                "transform_manifest_hash"
            ],
        )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("decode_passes", 7),
        ("decoded_blocks", 1),
        ("maximum_rhs_columns", 999),
    ],
)
def test_transform_summary_load_rejects_persisted_decode_or_rhs_tampering(
    tmp_path: Path, field: str, tampered_value: int
) -> None:
    fixture = _fixture(seed=6124)
    summary = _build_transform_summary(fixture, block_size=5, transform_tile_size=2)
    manifest, _ = write_context_transform_summary(summary, tmp_path / field / "scan")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["performance"][field] = tampered_value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="decode/RHS evidence"):
        load_context_transform_summary(
            manifest,
            expected_transform_manifest_hash=summary.manifest[
                "transform_manifest_hash"
            ],
        )


def test_transform_summary_io_preserves_legal_annotation_names_with_colons(
    tmp_path: Path,
) -> None:
    fixture = _fixture(seed=6117, annotation_name="maf:low")
    summary = _build_transform_summary(fixture)
    manifest, _ = write_context_transform_summary(summary, tmp_path / "scan")
    loaded = load_context_transform_summary(manifest)
    assert loaded.component_index.annotation_names == ("maf:low",)
    assert loaded.component_index.names == summary.component_index.names
