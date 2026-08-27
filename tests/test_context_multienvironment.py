from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from summit.context import (
    ContextRankError,
    FixedEffectInteractionSpec,
    MultiEnvironmentSourceSpec,
    apply_multienvironment_calibration,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    calibrate_multienvironment_basis,
    canonical_sha256,
    common_scale_features,
    coefficients_to_omegas,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    derive_context_contrast,
    derive_covariance_modes,
    fit_context_model,
    fit_multienvironment_model,
    fit_multienvironment_pruning,
    omegas_to_coefficients,
    project_genetic_coefficients_psd,
    project_normalize_phenotype,
    validate_multienvironment_preset,
)


def _reference_sources(n: int = 42, *, seed: int = 7101) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = rng.normal(size=n)
    second = 0.35 * first + rng.normal(scale=0.9, size=n)
    category = np.asarray(["north", "south"] * (n // 2) + ["north"] * (n % 2))
    rng.shuffle(category)
    return {"first": first, "second": second, "region": category}


def _source_specs(*, q: int = 4) -> tuple[MultiEnvironmentSourceSpec, ...]:
    specs = [
        MultiEnvironmentSourceSpec("first", "continuous"),
        MultiEnvironmentSourceSpec("second", "continuous"),
    ]
    if q == 4:
        specs.append(
            MultiEnvironmentSourceSpec(
                "region",
                "categorical",
                categories=("north", "south"),
                reference_category="north",
            )
        )
    return tuple(specs)


def _calibration_and_preset(
    *,
    q: int = 4,
    seed: int = 7101,
    genotype_for_diagnostics: np.ndarray | None = None,
    interactions: tuple[FixedEffectInteractionSpec, ...] = (),
):
    sources = _reference_sources(seed=seed)
    mask = np.ones(len(sources["first"]), dtype=bool)
    calibration = calibrate_multienvironment_basis(
        sources, _source_specs(q=q), mask=mask, basis_id=f"fixture_q{q}"
    )
    rng = np.random.default_rng(seed + 1)
    covariates = {"pc1": rng.normal(size=mask.size)}
    context_grid_sources = {
        "first": np.asarray([-1.0, -0.2, 0.6, 1.4]),
        "second": np.asarray([0.8, -0.5, 0.1, 1.1]),
    }
    if q == 4:
        context_grid_sources["region"] = np.asarray(
            ["north", "south", "north", "south"]
        )
    preset = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=mask,
        covariates=covariates,
        interactions=interactions,
        context_grid_sources=context_grid_sources,
        genotype_for_diagnostics=genotype_for_diagnostics,
    )
    return sources, mask, covariates, calibration, preset


def _core_hash(preset: Any, name: str) -> str:
    direct = getattr(preset, name, None)
    if isinstance(direct, str):
        return direct
    aliases = {
        "basis_hash": ("basis_hash", "analysis_spec_hash", "calibration_hash"),
        "fixed_effect_hash": (
            "fixed_effect_hash",
            "fixed_effect_spec_hash",
            "fixed_effect_array_hash",
        ),
    }
    for key in aliases[name]:
        value = preset.manifest.get(key)
        if isinstance(value, str):
            return value
    raise AssertionError(f"Multi-environment preset does not expose {name}.")


def _engine_fixture(*, q: int, seed: int):
    n, m = 42, 36
    rng = np.random.default_rng(seed)
    genotype = rng.normal(size=(n, m))
    genotype -= np.mean(genotype, axis=0, keepdims=True)
    genotype /= np.std(genotype, axis=0, ddof=1, keepdims=True)
    interaction = (FixedEffectInteractionSpec("first", "pc1", "first_by_pc1"),)
    sources, mask, _, calibration, preset = _calibration_and_preset(
        q=q,
        seed=seed + 20,
        genotype_for_diagnostics=genotype,
        interactions=interaction,
    )
    context = np.asarray(sources["first"])
    phenotype = 0.4 * context + 0.15 * context * context + rng.normal(size=n)
    annotations = np.ones((m, 1), dtype=np.float64)
    groups = tuple(f"group:{index // (m // 6)}" for index in range(m))
    common = {
        "genotype": genotype,
        "basis": preset.basis,
        "projector": preset.projector,
        "annotations": annotations,
        "component_index": preset.component_index,
        "basis_hash": _core_hash(preset, "basis_hash"),
        "fixed_effect_hash": _core_hash(preset, "fixed_effect_hash"),
        "variant_hash": canonical_sha256({"variants": list(range(m))}),
        "loo_groups": groups,
        "genotype_scaling": "pre_scaled_input",
    }
    reference = build_context_reference(
        **common, gram_method="exact", same_person_method="exact"
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=phenotype,
        residual_basis=preset.residual_basis,
        residual_names=preset.residual_names,
        block_size=7,
    )
    fit = fit_context_model(
        reference,
        summary,
        context_grid=preset.context_grid,
        basis_metric=preset.basis_metric,
    )
    return {
        "genotype": genotype,
        "phenotype": phenotype,
        "annotations": annotations,
        "groups": groups,
        "calibration": calibration,
        "preset": preset,
        "reference": reference,
        "summary": summary,
        "fit": fit,
    }


def _fit_with_omega(fit, omega: np.ndarray, loo_omegas: np.ndarray):
    components = fit.component_index
    genetic = omegas_to_coefficients(omega[None], components)
    loo_genetic = np.stack(
        [omegas_to_coefficients(value[None], components) for value in loo_omegas]
    )
    raw = np.concatenate([genetic, fit.residual_coefficients])
    loo = np.concatenate(
        [
            loo_genetic,
            np.broadcast_to(
                fit.residual_coefficients,
                (loo_genetic.shape[0], fit.residual_coefficients.size),
            ),
        ],
        axis=1,
    )
    centered = loo - np.mean(loo, axis=0, keepdims=True)
    covariance = (loo.shape[0] - 1.0) / loo.shape[0] * (centered.T @ centered)
    manifest = copy.deepcopy(fit.manifest)
    manifest["dimensions"]["jackknife_groups"] = loo.shape[0]
    return replace(
        fit,
        manifest=manifest,
        raw_coefficients=raw,
        genetic_coefficients=genetic,
        raw_omegas=omega[None],
        jackknife_groups=tuple(f"group:{index}" for index in range(loo.shape[0])),
        loo_coefficients=loo,
        jackknife_covariance=covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        psd_projection=None,
    )


def test_reference_fixed_calibration_uses_one_declared_mask_and_no_local_refit() -> (
    None
):
    reference = _reference_sources(seed=7102)
    n_reference = len(reference["first"])
    reference_mask = np.ones(n_reference, dtype=bool)
    reference_mask[[1, 7]] = False
    calibration = calibrate_multienvironment_basis(
        reference, _source_specs(), mask=reference_mask, basis_id="fixed_reference"
    )
    reference_preset = apply_multienvironment_calibration(
        calibration, reference, mask=reference_mask
    )
    np.testing.assert_allclose(reference_preset.basis[:, 0], 1.0)
    np.testing.assert_allclose(
        np.mean(reference_preset.basis[:, 1:3], axis=0), 0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        np.mean(reference_preset.basis[:, 1:3] ** 2, axis=0), 1.0, atol=2e-15
    )
    np.testing.assert_allclose(
        reference_preset.basis_metric,
        reference_preset.basis.T
        @ reference_preset.basis
        / reference_preset.basis.shape[0],
        rtol=2e-15,
        atol=2e-15,
    )
    assert reference_preset.basis_metric[0, 0] == pytest.approx(1.0)

    study = _reference_sources(n=36, seed=7103)
    study["first"] = 2.0 + 1.7 * study["first"]
    study_mask = np.ones(36, dtype=bool)
    study_mask[[0, 5, 12]] = False
    study_preset = apply_multienvironment_calibration(
        calibration, study, mask=study_mask
    )
    assert not np.allclose(np.mean(study_preset.basis[:, 1], axis=0), 0.0)
    assert reference_preset.basis_metric.shape == (4, 4)
    np.testing.assert_allclose(study_preset.basis_metric, reference_preset.basis_metric)
    assert _core_hash(study_preset, "basis_hash") == _core_hash(
        reference_preset, "basis_hash"
    )
    assert array_sha256(study_preset.basis) != array_sha256(reference_preset.basis)


def test_mixed_continuous_categorical_basis_and_unknown_or_missing_fail_closed() -> (
    None
):
    sources = _reference_sources(seed=7104)
    mask = np.ones(len(sources["first"]), dtype=bool)
    calibration = calibrate_multienvironment_basis(
        sources, _source_specs(), mask=mask, basis_id="mixed"
    )
    preset = apply_multienvironment_calibration(calibration, sources, mask=mask)
    assert preset.basis.shape == (mask.size, 4)
    fitted_category = calibration.source_calibrations[-1]
    south_index = fitted_category.categories.index("south")
    expected_indicator = (sources["region"] == "south").astype(
        np.float64
    ) - fitted_category.category_probabilities[south_index]
    np.testing.assert_array_equal(preset.basis[:, 3], expected_indicator)

    unknown = {name: value.copy() for name, value in sources.items()}
    unknown["region"][3] = "east"
    with pytest.raises(ValueError, match="(?i)unknown|category|categorical"):
        apply_multienvironment_calibration(calibration, unknown, mask=mask)

    missing = {name: value.copy() for name, value in sources.items()}
    missing["first"][2] = np.nan
    with pytest.raises(ValueError, match="(?i)missing|finite|mask"):
        apply_multienvironment_calibration(calibration, missing, mask=mask)
    masked = mask.copy()
    masked[2] = False
    accepted = apply_multienvironment_calibration(calibration, missing, mask=masked)
    assert accepted.basis.shape[0] == int(np.sum(masked))

    missing_source = dict(sources)
    del missing_source["second"]
    with pytest.raises(ValueError, match="(?i)source|second|missing"):
        apply_multienvironment_calibration(calibration, missing_source, mask=mask)


def test_typed_categories_do_not_coerce_boolean_and_integer_identity() -> None:
    sources = {
        "value": np.asarray([False, True, False, True], dtype=object),
    }
    spec = MultiEnvironmentSourceSpec(
        "value",
        "categorical",
        categories=(False, True),
        reference_category=False,
    )
    calibration = calibrate_multienvironment_basis(
        sources, (spec,), mask=np.ones(4, dtype=bool)
    )
    malformed = {"value": np.asarray([False, True, 0, True], dtype=object)}
    with pytest.raises(ValueError, match="(?i)unknown|category|type"):
        apply_multienvironment_calibration(
            calibration, malformed, mask=np.ones(4, dtype=bool)
        )


def test_five_column_basis_calibrates_and_applies() -> None:
    sources = _reference_sources(seed=7105)
    mask = np.ones(len(sources["first"]), dtype=bool)
    five_columns = _source_specs() + (
        MultiEnvironmentSourceSpec("extra", "continuous"),
    )
    sources["extra"] = np.linspace(-1.0, 1.0, mask.size)
    calibration = calibrate_multienvironment_basis(
        sources, five_columns, mask=mask
    )
    preset = apply_multienvironment_calibration(calibration, sources, mask=mask)
    assert calibration.basis.shape == (mask.size, 5)
    assert preset.basis.shape == (mask.size, 5)
    assert len(preset.component_index.pair_index) == 15


def test_fixed_effect_main_effects_interactions_and_hashes_are_explicit() -> None:
    sources = _reference_sources(seed=7106)
    mask = np.ones(len(sources["first"]), dtype=bool)
    calibration = calibrate_multienvironment_basis(
        sources, _source_specs(), mask=mask, basis_id="fixed_effects"
    )
    rng = np.random.default_rng(7107)
    covariates = {"pc1": rng.normal(size=mask.size), "age": rng.normal(size=mask.size)}
    interaction = FixedEffectInteractionSpec("first", "pc1", "first_by_pc1")
    preset = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=mask,
        covariates=covariates,
        interactions=(interaction,),
    )
    assert "interaction:first_by_pc1" in preset.fixed_effect_names
    interaction_index = preset.fixed_effect_names.index("interaction:first_by_pc1")
    first_index = preset.pruning.retained_names.index("first")
    np.testing.assert_allclose(
        preset.fixed_effect_design[:, interaction_index],
        preset.basis[:, first_index] * covariates["pc1"],
    )
    assert preset.projector.rank == np.linalg.matrix_rank(preset.fixed_effect_design)
    genotype = rng.normal(size=(mask.size, 7))
    observed_features = common_scale_features(
        genotype, preset.basis, preset.projector.projector
    )
    explicit_features = np.stack(
        [
            preset.projector.projector @ (preset.basis[:, q, None] * genotype)
            for q in range(preset.basis.shape[1])
        ]
    )
    wrong_order = np.stack(
        [
            preset.basis[:, q, None] * (preset.projector.projector @ genotype)
            for q in range(preset.basis.shape[1])
        ]
    )
    np.testing.assert_allclose(observed_features, explicit_features, atol=3e-13)
    assert np.linalg.norm(observed_features - wrong_order) > 1.0e-2

    no_interaction = apply_multienvironment_calibration(
        calibration, sources, mask=mask, covariates=covariates
    )
    assert (
        preset.manifest["fixed_effect_spec_hash"]
        != no_interaction.manifest["fixed_effect_spec_hash"]
    )
    changed_values = {name: value[::-1].copy() for name, value in covariates.items()}
    same_spec = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=mask,
        covariates=changed_values,
        interactions=(interaction,),
    )
    assert (
        preset.manifest["fixed_effect_spec_hash"]
        == same_spec.manifest["fixed_effect_spec_hash"]
    )
    assert preset.fixed_effect_hash != same_spec.fixed_effect_hash

    with pytest.raises(ValueError, match="(?i)basis|unknown"):
        apply_multienvironment_calibration(
            calibration,
            sources,
            mask=mask,
            covariates=covariates,
            interactions=(
                FixedEffectInteractionSpec("absent", "pc1", "bad_interaction"),
            ),
        )
    with pytest.raises(ValueError, match="(?i)covariate|unknown"):
        apply_multienvironment_calibration(
            calibration,
            sources,
            mask=mask,
            covariates=covariates,
            interactions=(
                FixedEffectInteractionSpec("first", "absent", "bad_interaction"),
            ),
        )


def test_independent_cohorts_share_calibration_and_fixed_effect_specification_only() -> (
    None
):
    n_reference, n_study, m = 42, 36, 30
    reference_sources = _reference_sources(n_reference, seed=7126)
    study_sources = _reference_sources(n_study, seed=7127)
    study_sources["first"] = 1.5 + 1.3 * study_sources["first"]
    calibration = calibrate_multienvironment_basis(
        reference_sources,
        _source_specs(),
        mask=np.ones(n_reference, dtype=bool),
        basis_id="independent_cohorts",
    )
    interaction = (FixedEffectInteractionSpec("first", "pc1", "first_by_pc1"),)
    rng = np.random.default_rng(7128)
    reference_covariates = {"pc1": rng.normal(size=n_reference)}
    study_covariates = {"pc1": rng.normal(size=n_study)}
    reference_preset = apply_multienvironment_calibration(
        calibration,
        reference_sources,
        mask=np.ones(n_reference, dtype=bool),
        covariates=reference_covariates,
        interactions=interaction,
    )
    study_preset = apply_multienvironment_calibration(
        calibration,
        study_sources,
        mask=np.ones(n_study, dtype=bool),
        covariates=study_covariates,
        interactions=interaction,
    )
    assert (
        reference_preset.manifest["fixed_effect_spec_hash"]
        == study_preset.manifest["fixed_effect_spec_hash"]
    )
    assert reference_preset.fixed_effect_hash != study_preset.fixed_effect_hash
    assert _core_hash(reference_preset, "basis_hash") == _core_hash(
        study_preset, "basis_hash"
    )

    reference_genotype = rng.normal(size=(n_reference, m))
    study_genotype = rng.normal(size=(n_study, m))
    for genotype in (reference_genotype, study_genotype):
        genotype -= genotype.mean(axis=0, keepdims=True)
        genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    annotations = np.ones((m, 1))
    groups = tuple(f"group:{index // 5}" for index in range(m))
    variant_hash = canonical_sha256({"variants": list(range(m))})
    reference = build_context_reference(
        genotype=reference_genotype,
        basis=reference_preset.basis,
        projector=reference_preset.projector,
        annotations=annotations,
        component_index=reference_preset.component_index,
        basis_hash=_core_hash(reference_preset, "basis_hash"),
        fixed_effect_hash=_core_hash(reference_preset, "fixed_effect_hash"),
        variant_hash=variant_hash,
        loo_groups=groups,
        genotype_scaling="pre_scaled_input",
        gram_method="exact",
        same_person_method="exact",
    )
    summary = build_context_trait_summary(
        genotype=study_genotype,
        basis=study_preset.basis,
        phenotype=rng.normal(size=n_study),
        projector=study_preset.projector,
        annotations=annotations,
        component_index=study_preset.component_index,
        residual_basis=study_preset.residual_basis,
        residual_names=study_preset.residual_names,
        basis_hash=_core_hash(study_preset, "basis_hash"),
        fixed_effect_hash=_core_hash(study_preset, "fixed_effect_hash"),
        variant_hash=variant_hash,
        loo_groups=groups,
        genotype_scaling="pre_scaled_input",
    )
    fit = fit_multienvironment_model(
        reference,
        summary,
        reference_preset=reference_preset,
        study_preset=study_preset,
    )
    assert fit.manifest["dimensions"]["reference_n"] == n_reference
    assert fit.manifest["dimensions"]["n_samples"] == n_study

    wrong_components = replace(
        reference_preset.component_index, annotation_names=("other",)
    )
    wrong_reference_manifest = copy.deepcopy(reference.manifest)
    wrong_summary_manifest = copy.deepcopy(summary.manifest)
    for manifest in (wrong_reference_manifest, wrong_summary_manifest):
        manifest["component_index_hash"] = wrong_components.digest
        manifest["component_order"] = list(wrong_components.names)
        manifest["annotation_names"] = list(wrong_components.annotation_names)
    wrong_reference = replace(
        reference,
        component_index=wrong_components,
        manifest=wrong_reference_manifest,
    )
    wrong_summary = replace(
        summary,
        component_index=wrong_components,
        manifest=wrong_summary_manifest,
    )
    with pytest.raises(ValueError, match="(?i)component index"):
        fit_multienvironment_model(
            wrong_reference,
            wrong_summary,
            reference_preset=reference_preset,
            study_preset=study_preset,
        )

    wrong_residual_manifest = copy.deepcopy(summary.manifest)
    wrong_residual_manifest["residual_order"] = ["residual:other"]
    wrong_residual_summary = replace(
        summary,
        residual_names=("residual:other",),
        manifest=wrong_residual_manifest,
    )
    with pytest.raises(ValueError, match="(?i)residual order"):
        fit_multienvironment_model(
            reference,
            wrong_residual_summary,
            reference_preset=reference_preset,
            study_preset=study_preset,
        )

    incompatible_preset = apply_multienvironment_calibration(
        calibration,
        study_sources,
        mask=np.ones(n_study, dtype=bool),
        covariates=study_covariates,
        interactions=(),
    )
    incompatible_summary = build_context_trait_summary(
        genotype=study_genotype,
        basis=incompatible_preset.basis,
        phenotype=rng.normal(size=n_study),
        projector=incompatible_preset.projector,
        annotations=annotations,
        component_index=incompatible_preset.component_index,
        residual_basis=incompatible_preset.residual_basis,
        residual_names=incompatible_preset.residual_names,
        basis_hash=_core_hash(incompatible_preset, "basis_hash"),
        fixed_effect_hash=_core_hash(incompatible_preset, "fixed_effect_hash"),
        variant_hash=variant_hash,
        loo_groups=groups,
        genotype_scaling="pre_scaled_input",
    )
    with pytest.raises(ValueError, match="(?i)compatib|basis|fixed"):
        fit_multienvironment_model(
            reference,
            incompatible_summary,
            reference_preset=reference_preset,
            study_preset=incompatible_preset,
        )


def test_explicit_pruning_records_pinned_constant_subset_and_reconstruction() -> None:
    rng = np.random.default_rng(7108)
    first = rng.normal(size=40)
    sources = {"first": first, "duplicate": 2.0 * first}
    specs = (
        MultiEnvironmentSourceSpec("first", "continuous"),
        MultiEnvironmentSourceSpec("duplicate", "continuous"),
    )
    mask = np.ones(first.size, dtype=bool)
    calibration = calibrate_multienvironment_basis(
        sources, specs, mask=mask, basis_id="redundant"
    )
    pruning = fit_multienvironment_pruning(calibration, rtol=1e-10)
    assert pruning.retained_indices[0] == 0
    assert len(pruning.retained_indices) == 2
    assert len(pruning.dropped_indices) == 1

    full = apply_multienvironment_calibration(calibration, sources, mask=mask)
    reduced = apply_multienvironment_calibration(
        calibration, sources, mask=mask, pruning=pruning
    )
    np.testing.assert_allclose(
        reduced.basis @ pruning.reconstruction,
        full.basis,
        rtol=2e-13,
        atol=2e-13,
    )
    assert reduced.pruning.requested_names == full.pruning.requested_names
    assert reduced.pruning.retained_names == tuple(
        full.pruning.requested_names[index] for index in pruning.retained_indices
    )


def test_near_collinearity_is_diagnosed_without_silent_pruning() -> None:
    rng = np.random.default_rng(7109)
    n, m = 40, 24
    first = rng.normal(size=n)
    sources = {
        "first": first,
        "near_duplicate": first + 1.0e-4 * rng.normal(size=n),
    }
    specs = (
        MultiEnvironmentSourceSpec("first", "continuous"),
        MultiEnvironmentSourceSpec("near_duplicate", "continuous"),
    )
    mask = np.ones(n, dtype=bool)
    calibration = calibrate_multienvironment_basis(sources, specs, mask=mask)
    genotype = rng.normal(size=(n, m))
    preset = apply_multienvironment_calibration(
        calibration, sources, mask=mask, genotype_for_diagnostics=genotype
    )
    assert preset.basis.shape[1] == 3
    assert preset.conditioning.context_rank == 3
    assert preset.conditioning.context_condition_number > 1.0e6
    assert preset.conditioning.high_correlation_pairs
    assert preset.conditioning.projected_feature_rank == 3
    assert preset.pruning.dropped_indices == ()
    assert preset.pruning.manifest["method"] == "explicit_identity_no_pruning_v1"


@pytest.mark.parametrize("q", [3, 4])
def test_q3_q4_end_to_end_matches_explicit_dense_normal_equations(q: int) -> None:
    fixture = _engine_fixture(q=q, seed=7110 + q)
    preset = fixture["preset"]
    fit = fixture["fit"]
    normalized = project_normalize_phenotype(fixture["phenotype"], preset.projector)
    features = common_scale_features(
        fixture["genotype"], preset.basis, preset.projector.projector
    )
    genetic = dense_genetic_kernels(
        features,
        fixture["annotations"],
        preset.component_index,
    )
    residual = dense_residual_kernels(preset.projector.projector, preset.residual_basis)
    dense = dense_normal_equations(
        genetic,
        residual,
        normalized,
        preset.component_index.names,
        preset.residual_names,
    )
    np.testing.assert_allclose(
        fit.equations.matrix, dense.matrix, rtol=2e-12, atol=2e-10
    )
    np.testing.assert_allclose(fit.equations.rhs, dense.rhs, rtol=2e-12, atol=2e-10)
    expected = np.linalg.solve(dense.matrix, dense.rhs)
    np.testing.assert_allclose(fit.raw_coefficients, expected, rtol=3e-11, atol=3e-10)
    assert preset.component_index.pair_index.num_basis == q
    assert len(preset.component_index) == q * (q + 1) // 2
    assert all(
        entry.kernel_factor == 2
        for entry in preset.component_index.entries
        if entry.q != entry.r
    )
    annotation = fit.context_outputs["annotations"][0]
    np.testing.assert_allclose(annotation["omega"], fit.raw_omegas[0])
    np.testing.assert_allclose(
        annotation["covariance_surface"],
        preset.context_grid @ fit.raw_omegas[0] @ preset.context_grid.T,
        rtol=2e-13,
        atol=2e-13,
    )


def test_redundant_unpruned_basis_reaches_declared_rank_failure() -> None:
    rng = np.random.default_rng(7115)
    n, m = 36, 30
    first = rng.normal(size=n)
    sources = {"first": first, "duplicate": first.copy()}
    specs = (
        MultiEnvironmentSourceSpec("first", "continuous"),
        MultiEnvironmentSourceSpec("duplicate", "continuous"),
    )
    mask = np.ones(n, dtype=bool)
    calibration = calibrate_multienvironment_basis(sources, specs, mask=mask)
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    preset = apply_multienvironment_calibration(
        calibration, sources, mask=mask, genotype_for_diagnostics=genotype
    )
    assert preset.conditioning.context_rank < preset.basis.shape[1]
    assert preset.conditioning.projected_feature_rank < preset.basis.shape[1]

    annotations = np.ones((m, 1))
    groups = tuple(f"group:{index // 5}" for index in range(m))
    common = {
        "genotype": genotype,
        "basis": preset.basis,
        "projector": preset.projector,
        "annotations": annotations,
        "component_index": preset.component_index,
        "basis_hash": _core_hash(preset, "basis_hash"),
        "fixed_effect_hash": _core_hash(preset, "fixed_effect_hash"),
        "variant_hash": canonical_sha256({"variants": list(range(m))}),
        "loo_groups": groups,
        "genotype_scaling": "pre_scaled_input",
    }
    reference = build_context_reference(
        **common, gram_method="exact", same_person_method="exact"
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=rng.normal(size=n),
        residual_basis=preset.residual_basis,
        residual_names=preset.residual_names,
    )
    with pytest.raises(ContextRankError):
        fit_context_model(reference, summary)


def _omega_with_metric_spectrum(
    metric: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric_values, metric_vectors = np.linalg.eigh(metric)
    assert np.min(metric_values) > 0.0
    metric_sqrt = (metric_vectors * np.sqrt(metric_values)) @ metric_vectors.T
    metric_inverse_sqrt = (
        metric_vectors * (1.0 / np.sqrt(metric_values))
    ) @ metric_vectors.T
    omega = (
        metric_inverse_sqrt
        @ eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
        @ metric_inverse_sqrt
    )
    return omega, metric_sqrt, metric_inverse_sqrt


@pytest.mark.parametrize(
    "operator_eigenvalues, expected_fraction",
    [
        (np.asarray([4.0, 0.0, 0.0]), 1.0),
        (np.asarray([3.0, 1.0, 0.0]), 0.75),
    ],
)
def test_metric_modes_recover_rank_one_rank_two_equation_and_loo_uncertainty(
    operator_eigenvalues: np.ndarray, expected_fraction: float
) -> None:
    fixture = _engine_fixture(q=3, seed=7116)
    fit = fixture["fit"]
    preset = fixture["preset"].without_individual_data()
    metric = np.asarray(preset.basis_metric)
    rotation, _ = np.linalg.qr(
        np.asarray(
            [
                [1.0, 0.3, -0.2],
                [0.4, 1.0, 0.1],
                [-0.3, 0.2, 1.0],
            ]
        )
    )
    omega, _, metric_inverse_sqrt = _omega_with_metric_spectrum(
        metric, operator_eigenvalues, rotation
    )
    perturbations = np.asarray(
        [
            [-0.12, 0.04, 0.01],
            [0.08, -0.03, -0.01],
            [-0.05, 0.06, 0.02],
            [0.10, -0.04, -0.02],
            [-0.02, -0.01, 0.01],
            [0.04, 0.02, -0.01],
        ]
    )
    loo_omegas = np.stack(
        [
            _omega_with_metric_spectrum(
                metric,
                np.maximum(operator_eigenvalues + delta, 0.0),
                rotation,
            )[0]
            for delta in perturbations
        ]
    )
    synthetic_fit = _fit_with_omega(fit, omega, loo_omegas)

    modes = derive_covariance_modes(synthetic_fit, preset, annotation="all")
    np.testing.assert_allclose(
        modes.eigenvalues, operator_eigenvalues, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(modes.metric, metric, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        modes.eigenfunctions.T @ metric @ modes.eigenfunctions,
        np.eye(3),
        rtol=3e-12,
        atol=3e-12,
    )
    for index, eigenvalue in enumerate(modes.eigenvalues):
        coefficient = modes.eigenfunctions[:, index]
        np.testing.assert_allclose(
            omega @ metric @ coefficient,
            eigenvalue * coefficient,
            rtol=3e-11,
            atol=3e-11,
        )
        tied = any(
            other != index
            and abs(operator_eigenvalues[other] - eigenvalue)
            <= 1.0e-10 * max(1.0, abs(eigenvalue))
            for other in range(operator_eigenvalues.size)
        )
        if not tied:
            expected = metric_inverse_sqrt @ rotation[:, index]
            sign = np.sign(expected @ metric @ coefficient) or 1.0
            np.testing.assert_allclose(
                coefficient, sign * expected, rtol=3e-11, atol=3e-11
            )
    assert modes.rank_one_fraction == pytest.approx(expected_fraction, abs=3e-13)
    centered = modes.loo_eigenvalues - np.mean(
        modes.loo_eigenvalues, axis=0, keepdims=True
    )
    expected_se = np.sqrt(
        (centered.shape[0] - 1.0)
        / centered.shape[0]
        * np.sum(centered * centered, axis=0)
    )
    np.testing.assert_allclose(modes.eigenvalue_standard_errors, expected_se)

    values = preset.context_grid @ modes.eigenfunctions
    reconstructed = (values * modes.eigenvalues[None]) @ values.T
    direct = preset.context_grid @ omega @ preset.context_grid.T
    np.testing.assert_allclose(reconstructed, direct, rtol=3e-11, atol=3e-11)


def test_mode_loo_alignment_tracks_functions_through_eigenvalue_crossing() -> None:
    fixture = _engine_fixture(q=3, seed=7117)
    preset = fixture["preset"].without_individual_data()
    metric = np.asarray(preset.basis_metric)
    rotation, _ = np.linalg.qr(
        np.asarray([[1.0, 0.2, 0.1], [-0.2, 1.0, 0.3], [0.1, -0.3, 1.0]])
    )
    omega = _omega_with_metric_spectrum(metric, np.asarray([3.0, 2.0, 0.5]), rotation)[
        0
    ]
    replicate_spectra = np.asarray(
        [
            [1.8, 3.2, 0.5],
            [3.1, 1.9, 0.45],
            [2.9, 2.1, 0.55],
            [3.2, 1.8, 0.50],
            [2.8, 2.2, 0.48],
            [3.05, 1.95, 0.52],
        ]
    )
    loo_omegas = np.stack(
        [
            _omega_with_metric_spectrum(metric, values, rotation)[0]
            for values in replicate_spectra
        ]
    )
    synthetic_fit = _fit_with_omega(fixture["fit"], omega, loo_omegas)
    modes = derive_covariance_modes(synthetic_fit, preset, annotation="all")
    np.testing.assert_allclose(
        modes.loo_eigenvalues, replicate_spectra, rtol=3e-11, atol=3e-11
    )
    for group in range(replicate_spectra.shape[0]):
        alignment = modes.eigenfunctions.T @ metric @ modes.loo_eigenfunctions[group]
        assert np.min(np.diag(alignment)) > 1.0 - 3e-11
        np.testing.assert_allclose(
            alignment - np.diag(np.diag(alignment)), 0.0, atol=3e-11
        )

    permutation = np.asarray([4, 0, 5, 2, 1, 3])
    permuted_fit = replace(
        synthetic_fit,
        jackknife_groups=tuple(
            synthetic_fit.jackknife_groups[index] for index in permutation
        ),
        loo_coefficients=synthetic_fit.loo_coefficients[permutation],
    )
    permuted = derive_covariance_modes(permuted_fit, preset, annotation="all")
    np.testing.assert_allclose(
        permuted.loo_eigenvalues, modes.loo_eigenvalues[permutation]
    )
    np.testing.assert_allclose(
        permuted.loo_eigenfunctions, modes.loo_eigenfunctions[permutation]
    )


def test_exact_tied_modes_report_unidentified_individual_eigenfunctions() -> None:
    fixture = _engine_fixture(q=3, seed=7118)
    preset = fixture["preset"].without_individual_data()
    metric = np.asarray(preset.basis_metric)
    rotation, _ = np.linalg.qr(
        np.asarray([[1.0, 0.1, -0.2], [0.2, 1.0, 0.1], [-0.1, 0.3, 1.0]])
    )
    omega = _omega_with_metric_spectrum(metric, np.asarray([2.0, 2.0, 0.5]), rotation)[
        0
    ]
    rng = np.random.default_rng(7119)
    loo_omegas = np.stack(
        [omega + 0.005 * (value + value.T) for value in rng.normal(size=(6, 3, 3))]
    )
    synthetic_fit = _fit_with_omega(fixture["fit"], omega, loo_omegas)
    modes = derive_covariance_modes(
        synthetic_fit, preset, annotation="all", eigengap_rtol=1e-6
    )
    unstable_indices = {
        index for index, unstable in enumerate(modes.unstable_modes) if unstable
    }
    assert unstable_indices.issuperset({0, 1})
    assert "unstable" in modes.status or "tie" in modes.status


def test_indefinite_raw_modes_are_not_relabelled_psd_and_projection_is_separate() -> (
    None
):
    fixture = _engine_fixture(q=3, seed=7120)
    preset = fixture["preset"].without_individual_data()
    metric = np.asarray(preset.basis_metric)
    rotation = np.eye(3)
    raw_omega = _omega_with_metric_spectrum(
        metric, np.asarray([2.0, 1.0, -0.3]), rotation
    )[0]
    rng = np.random.default_rng(7121)
    loo_omegas = np.stack(
        [raw_omega + 0.025 * (value + value.T) for value in rng.normal(size=(9, 3, 3))]
    )
    synthetic_fit = _fit_with_omega(fixture["fit"], raw_omega, loo_omegas)
    raw_before = synthetic_fit.raw_coefficients.copy()
    raw_modes = derive_covariance_modes(synthetic_fit, preset, annotation="all")
    assert "indefinite" in raw_modes.status
    assert np.isnan(raw_modes.rank_one_fraction)
    assert np.min(raw_modes.eigenvalues) < 0.0
    np.testing.assert_array_equal(synthetic_fit.raw_coefficients, raw_before)

    genetic_count = len(synthetic_fit.component_index)
    projection = project_genetic_coefficients_psd(
        synthetic_fit.genetic_coefficients,
        synthetic_fit.jackknife_covariance[:genetic_count, :genetic_count],
        synthetic_fit.component_index,
    )
    projected_fit = replace(synthetic_fit, psd_projection=projection)
    projected_modes = derive_covariance_modes(
        projected_fit, preset, annotation="all", use_psd=True
    )
    assert np.min(projected_modes.eigenvalues) >= -1e-10
    assert projected_modes.loo_eigenvalues.shape == loo_omegas.shape[:1] + (3,)
    assert np.all(np.isfinite(projected_modes.loo_rank_one_fractions))
    metric_values, metric_vectors = np.linalg.eigh(metric)
    metric_sqrt = (metric_vectors * np.sqrt(metric_values)) @ metric_vectors.T
    covariance = synthetic_fit.jackknife_covariance[:genetic_count, :genetic_count]
    annotations_disjoint = synthetic_fit.manifest["psd_projection"][
        "annotations_disjoint"
    ]
    for index, raw_coefficients in enumerate(
        synthetic_fit.loo_coefficients[:, :genetic_count]
    ):
        replicate_projection = project_genetic_coefficients_psd(
            raw_coefficients,
            covariance,
            synthetic_fit.component_index,
            annotations_disjoint=annotations_disjoint,
        )
        replicate_omega = coefficients_to_omegas(
            replicate_projection.projected_coefficients,
            synthetic_fit.component_index,
        )[0]
        expected = np.linalg.eigvalsh(metric_sqrt @ replicate_omega @ metric_sqrt)[::-1]
        np.testing.assert_allclose(
            projected_modes.loo_eigenvalues[index], expected, rtol=3e-10, atol=3e-10
        )
    assert np.isfinite(projected_modes.rank_one_fraction)
    assert projection.distance > 0.0
    np.testing.assert_array_equal(projected_fit.raw_coefficients, raw_before)


@pytest.mark.parametrize("kind", ["covariance", "variance_difference"])
def test_context_contrast_matches_direct_matrix_and_manual_joint_loo_uncertainty(
    kind: str,
) -> None:
    fixture = _engine_fixture(q=3, seed=7122)
    preset = fixture["preset"].without_individual_data()
    omega = np.asarray([[1.4, -0.35, 0.2], [-0.35, 0.9, -0.15], [0.2, -0.15, 0.7]])
    rng = np.random.default_rng(7123)
    perturbations = rng.normal(scale=0.03, size=(8, 3, 3))
    loo_omegas = np.stack([omega + 0.5 * (value + value.T) for value in perturbations])
    synthetic_fit = _fit_with_omega(fixture["fit"], omega, loo_omegas)
    left = np.asarray(preset.context_grid[0])
    right = np.asarray(preset.context_grid[-1])
    result = derive_context_contrast(
        synthetic_fit,
        preset,
        left,
        right,
        annotation="all",
        kind=kind,
    )
    if kind == "covariance":
        expected = float(left @ omega @ right)
        expected_loo = np.asarray([left @ value @ right for value in loo_omegas])
    else:
        expected = float(left @ omega @ left - right @ omega @ right)
        expected_loo = np.asarray(
            [left @ value @ left - right @ value @ right for value in loo_omegas]
        )
    assert result.estimate == pytest.approx(expected, rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(result.loo_values, expected_loo, rtol=2e-13, atol=2e-13)
    centered = expected_loo - np.mean(expected_loo)
    expected_se = np.sqrt(
        (expected_loo.size - 1.0) / expected_loo.size * (centered @ centered)
    )
    assert result.standard_error == pytest.approx(expected_se, rel=2e-13, abs=2e-13)
    assert result.loo_groups == synthetic_fit.jackknife_groups

    reverse = derive_context_contrast(
        synthetic_fit,
        preset,
        right,
        left,
        annotation="all",
        kind=kind,
    )
    expected_sign = 1.0 if kind == "covariance" else -1.0
    assert reverse.estimate == pytest.approx(expected_sign * result.estimate)
    np.testing.assert_allclose(reverse.loo_values, expected_sign * result.loo_values)


def test_nonorthogonal_basis_mixing_preserves_surfaces_modes_and_contrasts() -> None:
    n, m = 42, 36
    rng = np.random.default_rng(7129)
    original_sources = _reference_sources(n, seed=7130)
    original_sources = {
        "first": original_sources["first"],
        "second": original_sources["second"],
    }
    mixed_sources = {
        "mixed_first": original_sources["first"] + 0.7 * original_sources["second"],
        "mixed_second": -0.4 * original_sources["first"]
        + 1.2 * original_sources["second"],
    }
    mask = np.ones(n, dtype=bool)
    original_calibration = calibrate_multienvironment_basis(
        original_sources,
        (
            MultiEnvironmentSourceSpec("first", "continuous"),
            MultiEnvironmentSourceSpec("second", "continuous"),
        ),
        mask=mask,
        basis_id="original_coordinates",
    )
    mixed_calibration = calibrate_multienvironment_basis(
        mixed_sources,
        (
            MultiEnvironmentSourceSpec("mixed_first", "continuous"),
            MultiEnvironmentSourceSpec("mixed_second", "continuous"),
        ),
        mask=mask,
        basis_id="mixed_coordinates",
    )
    covariates = {"pc1": rng.normal(size=n)}
    original_grid = {
        "first": np.asarray([-1.0, 0.2, 1.3, -0.4]),
        "second": np.asarray([0.5, -0.7, 0.1, 1.1]),
    }
    mixed_grid = {
        "mixed_first": original_grid["first"] + 0.7 * original_grid["second"],
        "mixed_second": -0.4 * original_grid["first"] + 1.2 * original_grid["second"],
    }
    original_preset = apply_multienvironment_calibration(
        original_calibration,
        original_sources,
        mask=mask,
        covariates=covariates,
        context_grid_sources=original_grid,
    )
    mixed_preset = apply_multienvironment_calibration(
        mixed_calibration,
        mixed_sources,
        mask=mask,
        covariates=covariates,
        context_grid_sources=mixed_grid,
    )
    basis_map_transpose = np.linalg.lstsq(
        original_preset.basis, mixed_preset.basis, rcond=None
    )[0]
    basis_map = basis_map_transpose.T
    assert np.linalg.cond(basis_map) < 20.0
    np.testing.assert_allclose(
        mixed_preset.basis,
        original_preset.basis @ basis_map.T,
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        mixed_preset.basis_metric,
        basis_map @ original_preset.basis_metric @ basis_map.T,
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        original_preset.projector.projector,
        mixed_preset.projector.projector,
        rtol=3e-12,
        atol=3e-12,
    )

    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    phenotype = rng.normal(size=n)
    annotations = np.ones((m, 1))
    groups = tuple(f"group:{index // 6}" for index in range(m))
    variant_hash = canonical_sha256({"variants": list(range(m))})

    def fit_template(preset):
        common = {
            "genotype": genotype,
            "basis": preset.basis,
            "projector": preset.projector,
            "annotations": annotations,
            "component_index": preset.component_index,
            "basis_hash": _core_hash(preset, "basis_hash"),
            "fixed_effect_hash": _core_hash(preset, "fixed_effect_hash"),
            "variant_hash": variant_hash,
            "loo_groups": groups,
            "genotype_scaling": "pre_scaled_input",
        }
        reference = build_context_reference(
            **common, gram_method="exact", same_person_method="exact"
        )
        summary = build_context_trait_summary(
            **common,
            phenotype=phenotype,
            residual_basis=preset.residual_basis,
            residual_names=preset.residual_names,
        )
        return fit_context_model(reference, summary)

    original_fit = fit_template(original_preset)
    mixed_fit = fit_template(mixed_preset)
    original_omega = np.asarray(
        [[1.3, -0.2, 0.15], [-0.2, 0.8, -0.1], [0.15, -0.1, 0.5]]
    )
    rng = np.random.default_rng(7131)
    deviations = rng.normal(scale=0.02, size=(6, 3, 3))
    original_loo = np.stack(
        [original_omega + 0.5 * (value + value.T) for value in deviations]
    )
    inverse = np.linalg.inv(basis_map)
    mixed_omega = inverse.T @ original_omega @ inverse
    mixed_loo = np.stack([inverse.T @ value @ inverse for value in original_loo])
    original_fit = _fit_with_omega(original_fit, original_omega, original_loo)
    mixed_fit = _fit_with_omega(mixed_fit, mixed_omega, mixed_loo)

    original_modes = derive_covariance_modes(
        original_fit, original_preset.without_individual_data(), annotation="all"
    )
    mixed_modes = derive_covariance_modes(
        mixed_fit, mixed_preset.without_individual_data(), annotation="all"
    )
    np.testing.assert_allclose(
        mixed_modes.eigenvalues,
        original_modes.eigenvalues,
        rtol=3e-11,
        atol=3e-11,
    )
    np.testing.assert_allclose(
        mixed_modes.loo_eigenvalues,
        original_modes.loo_eigenvalues,
        rtol=3e-10,
        atol=3e-10,
    )
    for index in range(3):
        original_values = (
            original_preset.context_grid @ original_modes.eigenfunctions[:, index]
        )
        mixed_values = mixed_preset.context_grid @ mixed_modes.eigenfunctions[:, index]
        sign = np.sign(original_values @ mixed_values) or 1.0
        np.testing.assert_allclose(
            original_values, sign * mixed_values, rtol=5e-10, atol=5e-10
        )
    original_surface = (
        original_preset.context_grid @ original_omega @ original_preset.context_grid.T
    )
    mixed_surface = (
        mixed_preset.context_grid @ mixed_omega @ mixed_preset.context_grid.T
    )
    np.testing.assert_allclose(mixed_surface, original_surface, rtol=3e-12, atol=3e-12)

    original_contrast = derive_context_contrast(
        original_fit,
        original_preset.without_individual_data(),
        original_preset.context_grid[0],
        original_preset.context_grid[2],
        annotation="all",
        kind="variance_difference",
    )
    mixed_contrast = derive_context_contrast(
        mixed_fit,
        mixed_preset.without_individual_data(),
        mixed_preset.context_grid[0],
        mixed_preset.context_grid[2],
        annotation="all",
        kind="variance_difference",
    )
    assert mixed_contrast.estimate == pytest.approx(
        original_contrast.estimate, rel=5e-11, abs=5e-11
    )
    np.testing.assert_allclose(
        mixed_contrast.loo_values,
        original_contrast.loo_values,
        rtol=5e-10,
        atol=5e-10,
    )


def test_derived_outputs_require_matching_summary_only_preset_identity() -> None:
    fixture = _engine_fixture(q=3, seed=7124)
    fit = fixture["fit"]
    stripped = fixture["preset"].without_individual_data()
    assert stripped.basis.shape == (0, 3)
    assert stripped.fixed_effect_design.shape[0] == 0
    modes = derive_covariance_modes(fit, stripped, annotation="all")
    assert modes.metric.shape == (3, 3)
    assert validate_multienvironment_preset(stripped)["has_individual_data"] is False

    altered_grid = stripped.context_grid.copy()
    altered_grid[0, 1] += 0.25
    with pytest.raises(ValueError, match="(?i)identity|grid|preset"):
        derive_covariance_modes(
            fit,
            replace(stripped, context_grid=altered_grid),
            annotation="all",
        )

    altered_metric = stripped.reference_basis_metric.copy()
    altered_metric[1, 1] += 0.25
    with pytest.raises(ValueError, match="(?i)identity|metric|preset"):
        derive_covariance_modes(
            fit,
            replace(stripped, reference_basis_metric=altered_metric),
            annotation="all",
        )

    other = _engine_fixture(q=3, seed=7125)["preset"].without_individual_data()
    with pytest.raises(ValueError, match="(?i)basis|preset|manifest|compatib"):
        derive_covariance_modes(fit, other, annotation="all")
