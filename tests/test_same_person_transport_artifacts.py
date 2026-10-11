"""Artifacts and fits that carry the study same-person term.

Covers loading pre-fix artifacts (no context fourth moments), the new
optional arrays, and an end-to-end selectively missing study whose
transported fit should approach the fit that uses the study's own Gram.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from generalized_gxe_variant_ldscore_oracle import orthonormalize
from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    build_context_reference,
    build_context_trait_summary,
    fit_context_model,
    load_context_reference,
    load_context_trait_summary,
    rank_revealing_projector,
    write_context_reference,
    write_context_trait_summary,
)
from summit.context.oracle import (
    SamePersonTransportWarning,
    approximate_study_same_person,
    context_fourth_moments,
    dense_genetic_kernels,
    exact_study_same_person,
    kernel_gram,
    transfer_reference_gram,
)
from summit.context.spec import canonical_sha256
from summit.ldscore.generalized_gxe_fit_v1 import (
    assemble_generalized_gxe_normal_equation_batch_v1,
    fit_generalized_gxe_variant_model_v1,
)
from summit.ldscore.generalized_gxe_pass1 import (
    ArraySequentialGenotypeOperator,
    GeneralizedGxEPass1Executor,
    NumpyNNOperator,
)
from summit.ldscore.generalized_gxe_pass2 import (
    GeneralizedGxEPass2Executor,
    NumpyTNOperator,
)
from summit.ldscore.generalized_gxe_reference_v1 import (
    build_generalized_gxe_variant_reference_v1,
    compose_generalized_gxe_variant_references_v1,
    load_generalized_gxe_variant_reference_v1,
    reduce_generalized_gxe_reference_for_inference,
    reference_moments_after_deleting_variant_blocks_v1,
    serialize_generalized_gxe_inference_axes,
    write_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_trait_summary import (
    aggregate_generalized_gxe_trait_statistics,
    generalized_gxe_per_variant_trait_statistics,
    load_generalized_gxe_trait_summary,
    reaggregate_generalized_gxe_trait_summary,
    write_generalized_gxe_trait_summary,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)

FIXTURES = Path(__file__).parent / "fixtures" / "same_person_transport"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "same_person_legacy_generator", FIXTURES / "generate_legacy_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep the fixture directory clean
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _expected() -> dict:
    return json.loads((FIXTURES / "legacy_expected.json").read_text())


def _reference(genotype, basis, fixed, annotations, names, block_ids, labels, probes, mode):
    """Pure-Python two-pass reference with published context fourth moments."""
    genotype = np.asfortranarray(genotype)
    basis = np.asfortranarray(basis)
    plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=basis.shape[1],
            num_annotations=annotations.shape[1],
            num_probes=probes.probe_count,
            memory_limit_bytes=512 * 1024**2,
            genotype_format="bed",
            fixed_effect_rank=fixed.shape[1],
            preferred_variant_block_width=16,
            preferred_rhs_tile_columns=3 * basis.shape[1] ** 2,
            rhs_policy="tiled",
        )
    )
    operator = ArraySequentialGenotypeOperator(genotype)
    masses = annotations.sum(axis=0)
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        annotation_masses=masses,
        probe_spec=probes,
        work_plan=plan,
        nn_operator=NumpyNNOperator(),
        annotation_tile_width=annotations.shape[1],
        probe_tile_width=8,
        native_probe_module=False,
    ).execute()
    result = GeneralizedGxEPass2Executor(
        pass1_result=pass1,
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        work_plan=plan,
        tn_operator=NumpyTNOperator(),
        probe_tile_width=3,
        component_diagonal_sample_tile_width=64,
    ).execute()
    block_directed, block_masses, error = reduce_generalized_gxe_reference_for_inference(
        directional_ldscores=result.directional_ldscores,
        annotations=annotations,
        variant_block_ids=block_ids,
        block_labels=labels,
    )
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=genotype.shape[1],
        num_samples=genotype.shape[0],
        basis_names=tuple(f"basis_{index}" for index in range(basis.shape[1])),
        fixed_effect_rank=fixed.shape[1],
        annotation_names=names,
        annotation_masses=masses,
        variant_block_ids=block_ids,
        block_labels=labels,
        residual_component_names=("identity",),
    )
    arrays = {
        "directed_numerator": result.directed_numerator,
        "symmetric_numerator": result.symmetric_numerator,
        "genetic_gram": result.genetic_gram,
        "block_directed_numerator": block_directed,
        "block_annotation_mass": block_masses,
        "same_person": result.same_person,
        "context_fourth_moments": context_fourth_moments(basis),
    }
    if mode == "composable":
        arrays.update(
            directional_ldscores=result.directional_ldscores,
            annotations=annotations,
            component_kernel_diagonal=result.component_kernel_diagonal,
        )
    phases = {name: 0.0 for name in ("pass1", "barrier", "pass2", "finalize")}
    return build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probes,
        genotype_scale_plan={
            "genotype_scale_policy": "fixture_standardized_v1",
            "allele_orientation": "fixture_a1_v1",
        },
        arrays=arrays,
        pass_ledger=result.ledger.to_dict(),
        performance_ledger={
            "backend": "numpy_fixture",
            "threads": 1,
            "affinity": {},
            "numa_evidence": {},
            "phase_wall_seconds": phases,
            "phase_cpu_seconds": phases,
            "bytes_read": 0,
            "gemm_dimensions": [],
            "peak_rss_bytes": 0,
            "output_bytes": 0,
        },
        provenance={"fixture": "same_person_transport"},
        diagnostics={"block_reconstruction_error": error},
        mode=mode,
    )


def _trait(genotype, basis, fixed_design, phenotype, annotations, names, block_ids, labels):
    statistics = generalized_gxe_per_variant_trait_statistics(
        genotype=genotype,
        basis=basis,
        fixed_basis=orthonormalize(fixed_design),
        phenotypes=phenotype[:, None],
        residual_basis=np.ones((genotype.shape[0], 1)),
    )
    return aggregate_generalized_gxe_trait_statistics(
        statistics,
        annotations=annotations,
        annotation_names=names,
        variant_group_ids=block_ids,
        group_labels=labels,
        trait_ids=("trait",),
        residual_names=("identity",),
        n_samples=genotype.shape[0],
    )


def test_pre_fix_generalized_artifacts_load_and_fit_unchanged() -> None:
    reference = load_generalized_gxe_variant_reference_v1(
        FIXTURES / "legacy_reference.generalized-gxe-variant-ldscore-v1.npz"
    )
    trait = load_generalized_gxe_trait_summary(
        FIXTURES / "legacy_trait.generalized-gxe-trait-summary-v1.npz"
    )
    assert reference.context_fourth_moments is None
    assert trait.context_fourth_moments is None
    assert trait.n_samples < reference.n_samples
    with pytest.warns(SamePersonTransportWarning):
        fit = fit_generalized_gxe_variant_model_v1(reference, trait)
    assert fit.manifest["same_person_transport"] == "scaled_legacy"
    expected = _expected()
    np.testing.assert_allclose(
        fit.raw_coefficients, expected["generalized_raw_coefficients"], rtol=1e-12
    )
    np.testing.assert_allclose(
        fit.raw_standard_errors,
        expected["generalized_raw_standard_errors"],
        rtol=1e-12,
    )


def test_pre_fix_development_artifacts_load_and_fit_unchanged() -> None:
    reference = load_context_reference(
        FIXTURES / "legacy_context.context-reference.json"
    )
    summary = load_context_trait_summary(FIXTURES / "legacy_context.context-trait.json")
    assert reference.context_fourth_moments is None
    assert summary.context_fourth_moments is None
    with pytest.warns(SamePersonTransportWarning):
        fit = fit_context_model(reference, summary)
    assert fit.manifest["assembly"]["same_person_transport"] == "scaled_legacy"
    assert fit.equations.same_person_transport == "scaled_legacy"
    expected = _expected()
    np.testing.assert_allclose(
        fit.raw_coefficients, expected["context_raw_coefficients"], rtol=1e-12
    )
    np.testing.assert_allclose(
        fit.standard_errors, expected["context_standard_errors"], rtol=1e-12
    )


def test_development_artifacts_round_trip_fourth_moments_and_change_the_fit(
    tmp_path: Path,
) -> None:
    generator = _generator()
    design = generator.legacy_design()
    reference, summary = generator.context_objects(design)
    np.testing.assert_array_equal(
        reference.context_fourth_moments, context_fourth_moments(design["basis"])
    )
    np.testing.assert_array_equal(
        summary.context_fourth_moments,
        context_fourth_moments(design["basis"], design["rows"]),
    )
    manifest_path, _ = write_context_reference(reference, tmp_path / "new")
    trait_path, _ = write_context_trait_summary(summary, tmp_path / "new")
    loaded_reference = load_context_reference(manifest_path)
    loaded_summary = load_context_trait_summary(trait_path)
    np.testing.assert_array_equal(
        loaded_reference.context_fourth_moments, reference.context_fourth_moments
    )
    np.testing.assert_array_equal(
        loaded_summary.context_fourth_moments, summary.context_fourth_moments
    )
    fit = fit_context_model(loaded_reference, loaded_summary)
    assert fit.manifest["assembly"]["same_person_transport"] == (
        "fourth_moment_difference"
    )
    study_same = approximate_study_same_person(
        reference.same_person,
        reference.context_fourth_moments,
        summary.context_fourth_moments,
        reference_n=reference.n_samples,
        study_n=summary.n_samples,
        annotation_count=1,
    )
    np.testing.assert_allclose(
        fit.equations.transferred_genetic_gram,
        transfer_reference_gram(
            reference.gram,
            reference.same_person,
            reference_n=reference.n_samples,
            study_n=summary.n_samples,
            study_same_person=study_same,
        ),
        rtol=1e-14,
    )
    legacy = _expected()["context_raw_coefficients"]
    assert not np.allclose(fit.raw_coefficients, legacy, rtol=1e-6)


def test_generalized_artifacts_round_trip_and_compose_fourth_moments(
    tmp_path: Path,
) -> None:
    generator = _generator()
    design = generator.legacy_design()
    fixed = orthonormalize(design["fixed_design"])
    probes = GlobalVariantProbeSpec(4417, 0, 8)
    annotations = np.column_stack(
        [design["annotations"][:, 0], (np.arange(design["m"]) % 2).astype(float)]
    )
    composable = _reference(
        design["genotype"],
        design["basis"],
        fixed,
        annotations,
        ("all", "odd"),
        design["block_ids"],
        design["block_labels"],
        probes,
        "composable",
    )
    summary = compose_generalized_gxe_variant_references_v1(
        ((composable, ("all", "odd")),),
        variant_block_ids=design["block_ids"],
        block_labels=design["block_labels"],
        mode="summary",
    )
    assert summary.component_kernel_diagonal is None
    np.testing.assert_array_equal(
        summary.context_fourth_moments, context_fourth_moments(design["basis"])
    )
    path = write_generalized_gxe_variant_reference_v1(summary, tmp_path / "public")
    loaded = load_generalized_gxe_variant_reference_v1(path)
    np.testing.assert_array_equal(
        loaded.context_fourth_moments, summary.context_fourth_moments
    )
    rows = design["rows"]
    trait = _trait(
        design["genotype"][rows],
        design["basis"][rows],
        design["fixed_design"][rows],
        design["phenotype"],
        annotations[:, :1],
        ("all",),
        design["block_ids"],
        design["block_labels"],
    )
    trait_path = write_generalized_gxe_trait_summary(trait, tmp_path / "trait")
    loaded_trait = load_generalized_gxe_trait_summary(trait_path)
    np.testing.assert_array_equal(
        loaded_trait.context_fourth_moments, context_fourth_moments(design["basis"], rows)
    )
    reaggregated = reaggregate_generalized_gxe_trait_summary(
        loaded_trait,
        annotations=annotations,
        annotation_names=("all", "odd"),
        variant_group_ids=design["block_ids"],
        group_labels=design["block_labels"],
    )
    np.testing.assert_array_equal(
        reaggregated.context_fourth_moments, loaded_trait.context_fourth_moments
    )
    fit = fit_generalized_gxe_variant_model_v1(loaded, reaggregated)
    assert fit.manifest["same_person_transport"] == "fourth_moment_difference"
    exact = fit_generalized_gxe_variant_model_v1(
        composable, reaggregated, study_rows=rows
    )
    assert exact.manifest["same_person_transport"] == "own_rows_exact"
    np.testing.assert_allclose(
        exact.transferred_genetic_gram,
        transfer_reference_gram(
            composable.genetic_gram,
            composable.same_person,
            reference_n=composable.n_samples,
            study_n=len(rows),
            study_same_person=exact_study_same_person(
                composable.component_kernel_diagonal, rows
            ),
        ),
        rtol=1e-14,
    )
    # A source without fourth moments (an older bundle) leaves them out.
    older = dataclasses.replace(composable, context_fourth_moments=None)
    mixed = compose_generalized_gxe_variant_references_v1(
        ((older, ("all",)), (composable, ("odd",))),
        variant_block_ids=design["block_ids"],
        block_labels=design["block_labels"],
        mode="summary",
    )
    assert mixed.context_fourth_moments is None


def test_trait_summary_fourth_moment_declaration_is_checked(tmp_path: Path) -> None:
    generator = _generator()
    design = generator.legacy_design()
    rows = design["rows"]
    trait = _trait(
        design["genotype"][rows],
        design["basis"][rows],
        design["fixed_design"][rows],
        design["phenotype"],
        design["annotations"],
        ("all",),
        design["block_ids"],
        design["block_labels"],
    )
    path = write_generalized_gxe_trait_summary(trait, tmp_path / "trait")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    del arrays["context_fourth_moments"]
    tampered = tmp_path / "tampered.generalized-gxe-trait-summary-v1.npz"
    with tampered.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(ValueError, match="fourth-moment declaration"):
        load_generalized_gxe_trait_summary(tampered)


def _selective_cohort(seed: int):
    rng = np.random.default_rng(seed)
    n, m = 200, 300
    sex = rng.integers(0, 2, size=n).astype(float)
    age = rng.normal(size=n)
    basis = np.column_stack([np.ones(n), sex - 0.5, age])
    frequencies = rng.uniform(0.05, 0.5, size=m)
    raw = rng.binomial(2, frequencies, size=(n, m)).astype(float)
    genotype = (raw - 2 * frequencies) / np.sqrt(2 * frequencies * (1 - frequencies))
    fixed_design = np.column_stack([np.ones(n), sex, age, rng.normal(size=n)])
    # Missingness depends strongly on sex and weakly on age.
    keep = 0.15 + 0.7 * (sex > 0) + 0.1 * np.tanh(age)
    rows = np.flatnonzero(rng.random(n) < keep)
    phenotype = rng.normal(size=len(rows))
    return rng, basis, genotype, fixed_design, rows, phenotype


def test_selectively_missing_study_recovers_its_own_gram_fit_with_the_fix() -> None:
    _, basis, genotype, fixed_design, rows, phenotype = _selective_cohort(3)
    m = genotype.shape[1]
    annotations = np.ones((m, 1))
    names = ("all",)
    block_ids = np.arange(m, dtype=np.int64) % 10
    labels = tuple(f"b{index}" for index in range(10))
    reference = _reference(
        genotype,
        basis,
        orthonormalize(fixed_design),
        annotations,
        names,
        block_ids,
        labels,
        GlobalVariantProbeSpec(3, 0, 64),
        "composable",
    )
    trait = _trait(
        genotype[rows],
        basis[rows],
        fixed_design[rows],
        phenotype,
        annotations,
        names,
        block_ids,
        labels,
    )
    # Known answer: the same study moments with the study's own exact Gram
    # (its own people and projector), so no transport is involved at all.
    projector = rank_revealing_projector(fixed_design[rows]).projector
    features = np.stack(
        [projector @ (basis[rows, q, None] * genotype[rows]) for q in range(3)]
    )
    own_gram = kernel_gram(
        dense_genetic_kernels(
            features, annotations, ContextComponentIndex(names, ContextPairIndex(3))
        )
    )
    fits = {
        "scaled_legacy": fit_generalized_gxe_variant_model_v1(
            reference, trait, same_person_transport="scaled_legacy"
        ),
        "fourth_moment_difference": fit_generalized_gxe_variant_model_v1(
            reference, trait
        ),
        "own_rows_exact": fit_generalized_gxe_variant_model_v1(
            reference, trait, study_rows=rows
        ),
    }
    c = len(reference.component_index)
    matrix = np.array(fits["own_rows_exact"].normal_matrix)
    matrix[:c, :c] = own_gram
    target = np.linalg.solve(matrix, fits["own_rows_exact"].normal_rhs)[:c]
    se = fits["own_rows_exact"].raw_standard_errors[:c]
    gram_error, deviation = {}, {}
    for label, fit in fits.items():
        assert fit.manifest["same_person_transport"] == label
        gram_error[label] = np.linalg.norm(
            fit.transferred_genetic_gram - own_gram
        ) / np.linalg.norm(own_gram)
        deviation[label] = float(np.max(np.abs(fit.raw_coefficients[:c] - target) / se))
    assert gram_error["scaled_legacy"] > 0.3
    for label in ("fourth_moment_difference", "own_rows_exact"):
        assert gram_error[label] < 0.3 * gram_error["scaled_legacy"]
        assert deviation[label] < 0.6 * deviation["scaled_legacy"]


def test_deleted_systems_reuse_the_full_study_same_person_term() -> None:
    _, basis, genotype, fixed_design, rows, phenotype = _selective_cohort(5)
    genotype = genotype[:, :60]
    m = genotype.shape[1]
    annotations = np.ones((m, 1))
    block_ids = np.arange(m, dtype=np.int64) % 4
    labels = tuple(f"b{index}" for index in range(4))
    reference = _reference(
        genotype,
        basis,
        orthonormalize(fixed_design),
        annotations,
        ("all",),
        block_ids,
        labels,
        GlobalVariantProbeSpec(5, 0, 8),
        "summary",
    )
    trait = _trait(
        genotype[rows], basis[rows], fixed_design[rows], phenotype, annotations,
        ("all",), block_ids, labels,
    )
    systems = assemble_generalized_gxe_normal_equation_batch_v1(
        reference, trait, deleted_block_sets=((), *((label,) for label in labels))
    )
    study_same = approximate_study_same_person(
        reference.same_person,
        reference.context_fourth_moments,
        trait.context_fourth_moments,
        reference_n=reference.n_samples,
        study_n=trait.n_samples,
        annotation_count=1,
    )
    for deleted, system in zip(((), *((label,) for label in labels)), systems):
        assert system.same_person_transport == "fourth_moment_difference"
        moments = reference_moments_after_deleting_variant_blocks_v1(reference, deleted)
        np.testing.assert_array_equal(
            system.transferred_genetic_gram,
            transfer_reference_gram(
                moments.gram,
                reference.same_person,
                reference_n=reference.n_samples,
                study_n=trait.n_samples,
                study_same_person=study_same,
            ),
        )


def _development_selective_fits(seed: int):
    _, basis, genotype, fixed_design, rows, phenotype = _selective_cohort(seed)
    m = genotype.shape[1]
    components = ContextComponentIndex(("all",), ContextPairIndex(3))
    common = dict(
        annotations=np.ones((m, 1)),
        component_index=components,
        loo_groups=tuple(f"group:{index % 6}" for index in range(m)),
        basis_hash=canonical_sha256({"basis": "intercept_sex_age"}),
        variant_hash=canonical_sha256({"variants": list(range(m))}),
    )
    reference = build_context_reference(
        genotype=genotype,
        basis=basis,
        projector=rank_revealing_projector(fixed_design),
        fixed_effect_hash=canonical_sha256({"fixed": "reference"}),
        **common,
    )
    study_projector = rank_revealing_projector(fixed_design[rows])
    summary = build_context_trait_summary(
        genotype=genotype[rows],
        basis=basis[rows],
        projector=study_projector,
        phenotype=phenotype,
        residual_basis=np.ones((len(rows), 1)),
        residual_names=("residual:constant",),
        fixed_effect_hash=canonical_sha256({"fixed": "study"}),
        **common,
    )
    # Known answer: a reference built on the study's own people and projector,
    # whose Gram needs no transport (n_S = n_R and G_S = G_R).
    matched = build_context_reference(
        genotype=genotype[rows],
        basis=basis[rows],
        projector=study_projector,
        fixed_effect_hash=canonical_sha256({"fixed": "study"}),
        **common,
    )
    target = fit_context_model(matched, summary)
    fixed = fit_context_model(reference, summary)
    with pytest.warns(SamePersonTransportWarning):
        legacy = fit_context_model(
            dataclasses.replace(reference, context_fourth_moments=None), summary
        )
    np.testing.assert_array_equal(
        target.equations.transferred_genetic_gram, matched.gram
    )
    assert fixed.manifest["assembly"]["same_person_transport"] == (
        "fourth_moment_difference"
    )
    assert legacy.manifest["assembly"]["same_person_transport"] == "scaled_legacy"
    return target, fixed, legacy, len(components)


def test_development_fit_of_selective_study_approaches_matched_reference() -> None:
    gram_errors, deviations = [], []
    for seed in (3, 4, 5, 6):
        target, fixed, legacy, c = _development_selective_fits(seed)
        own = target.equations.transferred_genetic_gram
        se = target.standard_errors[:c]
        row_gram, row_deviation = [], []
        for fit in (legacy, fixed):
            row_gram.append(
                np.linalg.norm(fit.equations.transferred_genetic_gram - own)
                / np.linalg.norm(own)
            )
            row_deviation.append(
                np.max(np.abs(fit.raw_coefficients[:c] - target.raw_coefficients[:c]) / se)
            )
        gram_errors.append(row_gram)
        deviations.append(row_deviation)
    gram_errors = np.asarray(gram_errors)
    deviations = np.asarray(deviations)
    # Every draw: the Gram error falls several-fold (remaining error is the
    # unchanged different-person transport and the reference projector).
    assert np.all(gram_errors[:, 1] < 0.3 * gram_errors[:, 0])
    # Coefficients, in the matched fit's jackknife s.e., averaged over draws.
    assert deviations[:, 1].mean() < 0.5 * deviations[:, 0].mean()


def test_transform_scan_carries_study_fourth_moments(tmp_path: Path) -> None:
    from summit.context.transform import (
        PhenotypeTransformSpec,
        build_context_transform_summary,
        fit_context_transform_scan,
        load_context_transform_summary,
        write_context_transform_summary,
    )

    generator = _generator()
    design = generator.legacy_design()
    reference, _ = generator.context_objects(design)
    rows = design["rows"]
    summary = build_context_transform_summary(
        genotype=design["genotype"][rows],
        basis=design["basis"][rows],
        projector=rank_revealing_projector(design["fixed_design"][rows]),
        annotations=design["annotations"],
        component_index=reference.component_index,
        basis_hash=reference.manifest["basis_hash"],
        fixed_effect_hash=canonical_sha256({"fixed": "study"}),
        variant_hash=reference.manifest["variant_hash"],
        loo_groups=tuple(f"group:{index % 3}" for index in range(design["m"])),
        original_phenotype=np.exp(0.2 * design["phenotype"]),
        transformations=(
            PhenotypeTransformSpec("identity", "identity"),
            PhenotypeTransformSpec("log", "log", shift=0.0),
        ),
        residual_basis=np.ones((len(rows), 1)),
        residual_names=("residual:constant",),
        original_trait="fixture_trait",
    )
    np.testing.assert_array_equal(
        summary.context_fourth_moments, context_fourth_moments(design["basis"], rows)
    )
    np.testing.assert_array_equal(
        summary.to_single_summary(0).context_fourth_moments, summary.context_fourth_moments
    )
    manifest_path, _ = write_context_transform_summary(summary, tmp_path / "scan")
    loaded = load_context_transform_summary(manifest_path)
    np.testing.assert_array_equal(
        loaded.context_fourth_moments, summary.context_fourth_moments
    )
    fit = fit_context_transform_scan(reference, loaded)
    assert fit.equations.same_person_transport == "fourth_moment_difference"
    assert fit.manifest["same_person_transport"] == "fourth_moment_difference"
