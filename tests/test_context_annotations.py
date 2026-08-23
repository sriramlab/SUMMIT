from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from summit.context import (
    AnnotationContrastResult,
    AnnotationGroupBalance,
    AnnotationResourceEstimate,
    AnnotationResourceRequest,
    AnnotationTotalResult,
    ContextComponentIndex,
    ContextPairIndex,
    ContextRankError,
    DisjointAnnotationPartition,
    array_sha256,
    assemble_context_normal_equations,
    build_context_reference,
    build_context_trait_summary,
    build_disjoint_annotation_partition,
    build_grouped_context_reference,
    build_grouped_context_trait_summary,
    build_maf_ld_partition,
    canonical_sha256,
    coefficients_to_omegas,
    combine_genetic_kernels,
    common_scale_features,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    derive_annotation_contrast,
    derive_annotation_total,
    estimate_annotation_resources,
    fit_annotation_context_model,
    fit_context_model,
    group_context_reference,
    group_context_trait_summary,
    load_context_reference,
    load_context_trait_summary,
    load_disjoint_annotation_partition,
    omegas_to_coefficients,
    project_normalize_phenotype,
    rank_revealing_projector,
    reference_moments_after_deleting_groups,
    trait_moments_after_deleting_groups,
    validate_disjoint_annotation_partition,
    write_context_reference,
    write_context_trait_summary,
    write_disjoint_annotation_partition,
)


def _fixture(
    *,
    seed: int = 8301,
    n: int = 40,
    q: int = 2,
    k: int = 4,
    group_count: int = 8,
) -> dict[str, object]:
    """Balanced disjoint bins with deliberately nonorthogonal bin kernels."""
    rng = np.random.default_rng(seed)
    m = k * group_count
    shared = rng.normal(size=(n, group_count))
    genotype = np.empty((n, m), dtype=np.float64)
    for group in range(group_count):
        for annotation in range(k):
            column = group * k + annotation
            genotype[:, column] = 0.72 * shared[:, group] + 0.69 * rng.normal(size=n)
    genotype -= np.mean(genotype, axis=0, keepdims=True)
    genotype /= np.std(genotype, axis=0, ddof=1, keepdims=True)

    environment = rng.normal(size=n)
    basis = np.ones((n, q), dtype=np.float64)
    if q > 1:
        basis[:, 1] = environment
    if q > 2:
        basis[:, 2:] = rng.normal(size=(n, q - 2))
    fixed = np.column_stack([np.ones(n), basis[:, 1:], rng.normal(size=n)])
    projector = rank_revealing_projector(fixed)
    phenotype = rng.normal(size=n)
    residual_basis = np.column_stack([np.ones(n), environment * environment + 0.25])
    residual_names = ("residual:constant", "residual:environment_squared")

    annotations = np.zeros((m, k), dtype=np.float64)
    annotations[np.arange(m), np.arange(m) % k] = 1.0
    annotation_names = tuple(f"bin:{index}" for index in range(k))
    loo_groups = tuple(f"group:{index // k}" for index in range(m))
    variant_hash = canonical_sha256({"variant_ids": [f"v{j}" for j in range(m)]})
    definitions = tuple(
        {
            "kind": "synthetic_fixed_bin",
            "index": index,
            "label": annotation_names[index],
        }
        for index in range(k)
    )
    partition = build_disjoint_annotation_partition(
        annotations,
        annotation_names,
        definitions=definitions,
        variant_hash=variant_hash,
        loo_groups=loo_groups,
        source="synthetic_external_variant_table",
        source_digest=array_sha256(annotations),
    )
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q))
    return {
        "genotype": genotype,
        "basis": basis,
        "fixed": fixed,
        "projector": projector,
        "phenotype": phenotype,
        "residual_basis": residual_basis,
        "residual_names": residual_names,
        "annotations": annotations,
        "annotation_names": annotation_names,
        "definitions": definitions,
        "loo_groups": loo_groups,
        "variant_hash": variant_hash,
        "partition": partition,
        "components": components,
    }


def _core_build_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "genotype": fixture["genotype"],
        "basis": fixture["basis"],
        "projector": fixture["projector"],
        "annotations": fixture["annotations"],
        "component_index": fixture["components"],
        "loo_groups": fixture["loo_groups"],
        "basis_hash": array_sha256(fixture["basis"]),
        "fixed_effect_hash": array_sha256(fixture["fixed"]),
        "variant_hash": fixture["variant_hash"],
        "genotype_scaling": "pre_scaled_input",
    }


def _build_core_objects(fixture: dict[str, object]):
    common = _core_build_kwargs(fixture)
    reference = build_context_reference(
        **common,
        gram_method="exact",
        same_person_method="exact",
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=fixture["phenotype"],
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        block_size=5,
    )
    return reference, summary


def _dense_equations(fixture: dict[str, object]):
    y = project_normalize_phenotype(fixture["phenotype"], fixture["projector"])
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    genetic = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    residual = dense_residual_kernels(
        fixture["projector"].projector, fixture["residual_basis"]
    )
    return dense_normal_equations(
        genetic,
        residual,
        y,
        fixture["components"].names,
        fixture["residual_names"],
    )


def _group_masks(groups: tuple[str, ...]) -> tuple[tuple[str, ...], list[np.ndarray]]:
    ordered = tuple(dict.fromkeys(groups))
    values = np.asarray(groups, dtype=object)
    return ordered, [values == group for group in ordered]


def _jackknife_covariance(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    count = values.shape[0]
    return (count - 1.0) / count * (centered.T @ centered)


def _annotation_indices(
    components: ContextComponentIndex, annotation: str
) -> np.ndarray:
    return np.asarray(
        [
            entry.index
            for entry in components.entries
            if entry.annotation_name == annotation
        ],
        dtype=np.int64,
    )


def _assert_no_variant_axis(value: object, n_variants: int) -> None:
    """Grouped fit artifacts may not retain any M-axis numeric array."""
    if isinstance(value, np.ndarray):
        assert n_variants not in value.shape
        return
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_variant_axis(nested, n_variants)
        return
    if isinstance(value, (tuple, list)):
        for nested in value:
            _assert_no_variant_axis(nested, n_variants)
        return
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        for name in fields:
            _assert_no_variant_axis(getattr(value, name), n_variants)


def test_strict_partition_is_ordered_hash_stable_and_reports_group_balance() -> None:
    fixture = _fixture()
    partition = fixture["partition"]
    assert isinstance(partition, DisjointAnnotationPartition)
    validation = validate_disjoint_annotation_partition(partition)
    assert validation["status"] == "valid_strict_disjoint_partition"
    assert partition.annotation_names == fixture["annotation_names"]
    np.testing.assert_array_equal(partition.annotations, fixture["annotations"])
    np.testing.assert_array_equal(partition.annotation_masses, 8.0)
    assert isinstance(partition.group_balance, AnnotationGroupBalance)
    np.testing.assert_array_equal(partition.group_balance.group_variant_counts, 4)
    np.testing.assert_array_equal(partition.group_balance.group_annotation_masses, 1.0)
    np.testing.assert_array_equal(
        partition.group_balance.minimum_retained_annotation_mass, 7.0
    )
    assert partition.group_balance.total_groups_balanced
    assert partition.group_balance.every_deletion_retains_each_annotation
    assert partition.manifest["mode"] == "disjoint_partition"
    assert partition.manifest["annotation_names"] == list(fixture["annotation_names"])

    rebuilt = build_disjoint_annotation_partition(
        fixture["annotations"],
        fixture["annotation_names"],
        definitions=fixture["definitions"],
        variant_hash=fixture["variant_hash"],
        loo_groups=fixture["loo_groups"],
        source="synthetic_external_variant_table",
        source_digest=array_sha256(fixture["annotations"]),
    )
    assert rebuilt.manifest == partition.manifest

    permuted = np.asarray(fixture["annotations"])[:, ::-1]
    reordered = build_disjoint_annotation_partition(
        permuted,
        tuple(reversed(fixture["annotation_names"])),
        definitions=tuple(reversed(fixture["definitions"])),
        variant_hash=fixture["variant_hash"],
        loo_groups=fixture["loo_groups"],
        source="synthetic_external_variant_table",
        source_digest=array_sha256(permuted),
    )
    assert reordered.digest != partition.digest


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda x: x.__setitem__((0, 1), 1.0), "overlap|exactly one|row"),
        (lambda x: x.__setitem__((0, slice(None)), 0.0), "unassigned|exactly one|row"),
        (lambda x: x.__setitem__((0, 0), 0.5), "0/1|binary|exactly"),
        (lambda x: x.__setitem__((0, 0), np.nan), "finite|NaN"),
    ],
)
def test_strict_partition_rejects_overlap_unassigned_fractional_and_nonfinite(
    mutator, match: str
) -> None:
    fixture = _fixture()
    annotations = np.asarray(fixture["annotations"]).copy()
    mutator(annotations)
    with pytest.raises(ValueError, match=f"(?i){match}"):
        build_disjoint_annotation_partition(
            annotations,
            fixture["annotation_names"],
            definitions=fixture["definitions"],
            variant_hash=fixture["variant_hash"],
            loo_groups=fixture["loo_groups"],
            source="strict_test",
        )


def test_maf_ld_fixed_edges_are_half_open_with_closed_terminal_bins() -> None:
    maf = np.asarray([0.0, 0.049, 0.0, 0.049, 0.05, 0.50, 0.05, 0.50])
    ld_score = np.asarray([0.0, 0.99, 1.0, 2.0, 0.0, 0.99, 1.0, 2.0])
    groups = tuple(f"g{index % 4}" for index in range(maf.size))
    variant_hash = canonical_sha256({"variants": list(range(maf.size))})
    partition = build_maf_ld_partition(
        maf,
        ld_score,
        maf_edges=(0.0, 0.05, 0.5),
        ld_edges=(0.0, 1.0, 2.0),
        variant_hash=variant_hash,
        loo_groups=groups,
        source="fixed_reference_maf_ld_v1",
        source_digest=array_sha256(np.column_stack([maf, ld_score])),
    )
    assert partition.annotation_names == (
        "maf0__ld0",
        "maf0__ld1",
        "maf1__ld0",
        "maf1__ld1",
    )
    expected = np.zeros((maf.size, 4), dtype=np.float64)
    expected[np.arange(maf.size), [0, 0, 1, 1, 2, 2, 3, 3]] = 1.0
    np.testing.assert_array_equal(partition.annotations, expected)
    assert partition.manifest["boundary_convention"] == (
        "lower_inclusive_upper_exclusive_final_upper_closed"
    )
    assert partition.manifest["source"] == "fixed_reference_maf_ld_v1"
    assert partition.definitions[0]["maf"] == {
        "lower": 0.0,
        "upper": 0.05,
        "upper_closed": False,
    }
    assert partition.definitions[-1]["ld_score"] == {
        "lower": 1.0,
        "upper": 2.0,
        "upper_closed": True,
    }

    for bad_maf, bad_ld in ((maf + 0.51, ld_score), (maf, ld_score + 2.01)):
        with pytest.raises(ValueError, match="(?i)edge|range|boundar|unassigned"):
            build_maf_ld_partition(
                bad_maf,
                bad_ld,
                maf_edges=(0.0, 0.05, 0.5),
                ld_edges=(0.0, 1.0, 2.0),
                variant_hash=variant_hash,
                loo_groups=groups,
                source="fixed_reference_maf_ld_v1",
            )


def test_grouped_contributions_are_exact_sums_and_lossless_for_every_deletion() -> None:
    fixture = _fixture(q=3, k=4, n=44)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    ordered, masks = _group_masks(fixture["loo_groups"])
    assert grouped_reference.loo_group_ids == ordered
    assert grouped_summary.loo_group_ids == ordered
    np.testing.assert_array_equal(
        grouped_reference.group_variant_counts, [int(np.sum(mask)) for mask in masks]
    )
    expected_masses = np.asarray(
        [np.sum(np.asarray(fixture["annotations"])[mask], axis=0) for mask in masks]
    )
    np.testing.assert_array_equal(
        grouped_reference.group_annotation_masses, expected_masses
    )
    np.testing.assert_array_equal(
        grouped_summary.group_annotation_masses, expected_masses
    )
    np.testing.assert_allclose(
        grouped_reference.group_gram_numerator_contributions,
        np.asarray(
            [
                np.sum(reference.gram_numerator_contributions[mask], axis=0)
                for mask in masks
            ]
        ),
        rtol=2e-15,
        atol=2e-12,
    )
    for grouped, original, field in (
        (grouped_summary, summary, "rhs_numerator_contributions"),
        (grouped_summary, summary, "trace_numerator_contributions"),
        (grouped_summary, summary, "genetic_residual_numerator_contributions"),
    ):
        expected = np.asarray(
            [np.sum(getattr(original, field)[mask], axis=0) for mask in masks]
        )
        np.testing.assert_allclose(
            getattr(grouped, f"group_{field}"), expected, rtol=2e-15, atol=2e-12
        )

    np.testing.assert_allclose(grouped_reference.gram, reference.gram, atol=0, rtol=0)
    np.testing.assert_allclose(
        grouped_reference.same_person, reference.same_person, atol=0, rtol=0
    )
    np.testing.assert_allclose(grouped_summary.genetic_rhs, summary.genetic_rhs)
    np.testing.assert_allclose(grouped_summary.genetic_traces, summary.genetic_traces)
    np.testing.assert_allclose(
        grouped_summary.genetic_residual, summary.genetic_residual
    )
    for group in ordered:
        observed_reference = reference_moments_after_deleting_groups(
            grouped_reference, (group,)
        )
        expected_reference = reference_moments_after_deleting_groups(
            reference, (group,)
        )
        np.testing.assert_allclose(observed_reference.gram, expected_reference.gram)
        np.testing.assert_allclose(
            observed_reference.annotation_masses, expected_reference.annotation_masses
        )
        observed_summary = trait_moments_after_deleting_groups(
            grouped_summary, (group,)
        )
        expected_summary = trait_moments_after_deleting_groups(summary, (group,))
        for field in (
            "annotation_masses",
            "genetic_rhs",
            "genetic_traces",
            "genetic_residual",
        ):
            np.testing.assert_allclose(
                getattr(observed_summary, field), getattr(expected_summary, field)
            )
    assert (
        grouped_reference.manifest["approximate_loo"]["exact_deleted_kernels"] is False
    )
    assert grouped_reference.manifest["approximate_loo"]["same_person_deletion"] == (
        "reuse_full_reference_D"
    )
    assert (
        grouped_reference.manifest["approximate_loo"]["contribution_storage"]
        == "loo_grouped"
    )


def test_grouped_artifacts_have_no_variant_axis_and_direct_builders_match_adapters() -> (
    None
):
    fixture = _fixture(q=2, k=4)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    m = np.asarray(fixture["genotype"]).shape[1]
    _assert_no_variant_axis(grouped_reference, m)
    _assert_no_variant_axis(grouped_summary, m)

    common = _core_build_kwargs(fixture)
    common.pop("annotations")
    common.pop("component_index")
    common.pop("loo_groups")
    direct_reference = build_grouped_context_reference(
        partition=fixture["partition"],
        **common,
        gram_method="exact",
        same_person_method="exact",
    )
    direct_summary = build_grouped_context_trait_summary(
        partition=fixture["partition"],
        **common,
        phenotype=fixture["phenotype"],
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        block_size=5,
    )
    np.testing.assert_allclose(direct_reference.gram, grouped_reference.gram)
    np.testing.assert_allclose(
        direct_reference.group_gram_numerator_contributions,
        grouped_reference.group_gram_numerator_contributions,
    )
    np.testing.assert_allclose(direct_summary.genetic_rhs, grouped_summary.genetic_rhs)
    np.testing.assert_allclose(
        direct_summary.group_rhs_numerator_contributions,
        grouped_summary.group_rhs_numerator_contributions,
    )
    _assert_no_variant_axis(direct_reference, m)
    _assert_no_variant_axis(direct_summary, m)


def test_grouped_full_and_loo_fit_match_snp_storage_and_dense_joint_system() -> None:
    fixture = _fixture(q=2, k=4, n=44)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    dense = _dense_equations(fixture)
    grouped_equations = assemble_context_normal_equations(
        grouped_reference, grouped_summary
    )
    np.testing.assert_allclose(
        grouped_equations.matrix, dense.matrix, rtol=2e-13, atol=2e-11
    )
    np.testing.assert_allclose(grouped_equations.rhs, dense.rhs, rtol=2e-13, atol=2e-11)
    np.testing.assert_allclose(
        grouped_equations.traces, dense.traces, rtol=2e-13, atol=2e-11
    )

    pair_count = len(fixture["components"].pair_index)
    off_block = grouped_equations.matrix[:pair_count, pair_count : 2 * pair_count]
    assert np.max(np.abs(off_block)) > 1e-5

    snp_fit = fit_context_model(reference, summary)
    grouped_fit = fit_annotation_context_model(
        grouped_reference,
        grouped_summary,
        partition=fixture["partition"],
    )
    np.testing.assert_allclose(grouped_fit.raw_coefficients, snp_fit.raw_coefficients)
    np.testing.assert_allclose(grouped_fit.loo_coefficients, snp_fit.loo_coefficients)
    np.testing.assert_allclose(
        grouped_fit.jackknife_covariance, snp_fit.jackknife_covariance
    )
    assert grouped_fit.manifest["annotation_mode"] == "disjoint_partition"
    assert (
        grouped_fit.manifest["annotation_partition_hash"] == fixture["partition"].digest
    )
    assert grouped_fit.manifest["annotation_interpretation"] == {
        "omega": "total_bin_covariance_contribution",
        "per_annotation_mass": "omega_divided_by_annotation_mass",
        "overlapping_annotations": "unsupported",
    }


def test_annotation_kernels_use_bin_mass_and_single_symmetric_offdiagonal() -> None:
    fixture = _fixture(q=2, k=2, n=30, group_count=6)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    entries = fixture["components"].entries
    for annotation_index in range(2):
        weight = np.asarray(fixture["annotations"])[:, annotation_index]
        mass = float(np.sum(weight))
        f0 = features[0] * weight[None, :]
        f1 = features[1] * weight[None, :]
        expected = {
            (0, 0): (f0 @ features[0].T) / mass,
            (1, 1): (f1 @ features[1].T) / mass,
            (0, 1): (f0 @ features[1].T + f1 @ features[0].T) / mass,
        }
        for component in entries:
            if component.annotation_index == annotation_index:
                np.testing.assert_allclose(
                    kernels[component.index],
                    expected[(component.q, component.r)],
                    rtol=2e-14,
                    atol=2e-13,
                )

    reference, _ = _build_core_objects(fixture)
    expected_gram = np.einsum("aij,bij->ab", kernels, kernels, optimize=True)
    np.testing.assert_allclose(reference.gram, expected_gram, rtol=2e-13, atol=2e-11)
    pair_count = len(fixture["components"].pair_index)
    assert np.max(np.abs(expected_gram[:pair_count, pair_count:])) > 1.0e-5


def test_unequal_partition_masses_obey_refinement_and_common_per_snp_scaling() -> None:
    rng = np.random.default_rng(8617)
    n, m, q = 24, 10, 2
    genotype = rng.normal(size=(n, m))
    genotype -= np.mean(genotype, axis=0, keepdims=True)
    genotype /= np.std(genotype, axis=0, ddof=1, keepdims=True)
    environment = rng.normal(size=n)
    basis = np.column_stack([np.ones(n), environment])
    projector = rank_revealing_projector(
        np.column_stack([np.ones(n), environment, rng.normal(size=n)])
    )
    features = common_scale_features(genotype, basis, projector.projector)
    annotations = np.zeros((m, 2), dtype=np.float64)
    annotations[:3, 0] = 1.0
    annotations[3:, 1] = 1.0
    split_components = ContextComponentIndex(("small", "large"), ContextPairIndex(q))
    all_components = ContextComponentIndex(("all",), ContextPairIndex(q))
    split_kernels = dense_genetic_kernels(features, annotations, split_components)
    all_kernels = dense_genetic_kernels(
        features, np.ones((m, 1), dtype=np.float64), all_components
    )
    masses = np.sum(annotations, axis=0)
    pair_count = len(all_components)
    for pair in range(pair_count):
        expected = (
            masses[0] / m * split_kernels[pair]
            + masses[1] / m * split_kernels[pair_count + pair]
        )
        np.testing.assert_allclose(all_kernels[pair], expected, rtol=2e-14, atol=2e-13)

    common_per_snp_omega = np.asarray([[0.8, -0.17], [-0.17, 0.45]])
    all_coefficients = omegas_to_coefficients(
        common_per_snp_omega[None, :, :], all_components
    )
    split_coefficients = omegas_to_coefficients(
        np.asarray(
            [
                masses[0] / m * common_per_snp_omega,
                masses[1] / m * common_per_snp_omega,
            ]
        ),
        split_components,
    )
    np.testing.assert_allclose(
        combine_genetic_kernels(split_kernels, split_coefficients),
        combine_genetic_kernels(all_kernels, all_coefficients),
        rtol=2e-14,
        atol=2e-13,
    )


def test_annotation_fit_requires_all_frozen_groups_but_accepts_reordered_set() -> None:
    fixture = _fixture(q=2, k=4, n=44)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    groups = grouped_summary.loo_group_ids
    canonical = fit_annotation_context_model(
        grouped_reference, grouped_summary, partition=fixture["partition"]
    )
    reversed_fit = fit_annotation_context_model(
        grouped_reference,
        grouped_summary,
        partition=fixture["partition"],
        loo_groups=tuple(reversed(groups)),
    )
    np.testing.assert_allclose(
        reversed_fit.loo_coefficients, canonical.loo_coefficients[::-1]
    )
    np.testing.assert_allclose(
        reversed_fit.jackknife_covariance, canonical.jackknife_covariance
    )
    reversed_contrast = derive_annotation_contrast(
        reversed_fit,
        "bin:0",
        "bin:1",
        reference=grouped_reference,
        summary=grouped_summary,
    )
    pair_count = len(fixture["components"].pair_index)
    expected_trace = []
    for row, group in zip(
        reversed_fit.loo_coefficients, reversed_fit.jackknife_groups, strict=True
    ):
        traces = assemble_context_normal_equations(
            grouped_reference, grouped_summary, (group,)
        ).traces[: len(fixture["components"])]
        expected_trace.append(
            row[:pair_count] @ traces[:pair_count]
            - row[pair_count : 2 * pair_count] @ traces[pair_count : 2 * pair_count]
        )
    np.testing.assert_allclose(reversed_contrast.loo_trace_differences, expected_trace)
    with pytest.raises(ValueError, match="(?i)all|available|exhaust|group"):
        fit_annotation_context_model(
            grouped_reference,
            grouped_summary,
            partition=fixture["partition"],
            loo_groups=groups[:-1],
        )


def test_collinear_annotation_kernels_report_rank_failure_and_null_direction() -> None:
    fixture = _fixture(q=1, k=2, n=30, group_count=6)
    genotype = np.asarray(fixture["genotype"]).copy()
    genotype[:, 1::2] = genotype[:, 0::2]
    fixture["genotype"] = genotype
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    equations = assemble_context_normal_equations(grouped_reference, grouped_summary)
    null = np.zeros(equations.matrix.shape[0])
    null[0] = 1.0
    null[1] = -1.0
    np.testing.assert_allclose(equations.matrix @ null, 0.0, rtol=0.0, atol=2e-11)
    with pytest.raises(ContextRankError) as caught:
        fit_annotation_context_model(
            grouped_reference,
            grouped_summary,
            partition=fixture["partition"],
        )
    diagnostics = caught.value.diagnostics
    assert diagnostics.rank < equations.matrix.shape[0]
    assert np.max(np.abs(diagnostics.null_space.T @ null)) > 0.99 * np.linalg.norm(null)


def test_partition_mismatch_and_overlap_claims_fail_closed() -> None:
    fixture = _fixture()
    reference, summary = _build_core_objects(fixture)
    annotations = np.asarray(fixture["annotations"])[:, ::-1]
    wrong_partition = build_disjoint_annotation_partition(
        annotations,
        tuple(reversed(fixture["annotation_names"])),
        definitions=tuple(reversed(fixture["definitions"])),
        variant_hash=fixture["variant_hash"],
        loo_groups=fixture["loo_groups"],
        source="wrong_order",
        source_digest=array_sha256(annotations),
    )
    with pytest.raises(ValueError, match="(?i)annotation|partition|order|hash"):
        group_context_reference(reference, wrong_partition)
    with pytest.raises(ValueError, match="(?i)annotation|partition|order|hash"):
        group_context_trait_summary(summary, wrong_partition)

    wrong_variant_partition = build_disjoint_annotation_partition(
        fixture["annotations"],
        fixture["annotation_names"],
        definitions=fixture["definitions"],
        variant_hash=canonical_sha256({"variant_ids": ["wrong-order"]}),
        loo_groups=fixture["loo_groups"],
        source="wrong_variant_universe",
        source_digest=array_sha256(fixture["annotations"]),
    )
    with pytest.raises(ValueError, match="(?i)variant|order|hash"):
        group_context_reference(reference, wrong_variant_partition)
    with pytest.raises(ValueError, match="(?i)variant|order|hash"):
        group_context_trait_summary(summary, wrong_variant_partition)

    changed_definitions = tuple(
        {**definition, "calibration_version": "silently_changed"}
        for definition in fixture["definitions"]
    )
    changed_definition_partition = build_disjoint_annotation_partition(
        fixture["annotations"],
        fixture["annotation_names"],
        definitions=changed_definitions,
        variant_hash=fixture["variant_hash"],
        loo_groups=fixture["loo_groups"],
        source="synthetic_external_variant_table",
        source_digest=array_sha256(fixture["annotations"]),
    )
    grouped_reference = group_context_reference(reference, fixture["partition"])
    changed_summary = group_context_trait_summary(summary, changed_definition_partition)
    with pytest.raises(ValueError, match="(?i)definition|partition|annotation|hash"):
        fit_annotation_context_model(grouped_reference, changed_summary)

    overlapping = np.asarray(fixture["annotations"]).copy()
    overlapping[0, 1] = 1.0
    with pytest.raises(ValueError, match="(?i)overlap|exactly one"):
        build_disjoint_annotation_partition(
            overlapping,
            fixture["annotation_names"],
            definitions=fixture["definitions"],
            variant_hash=fixture["variant_hash"],
            loo_groups=fixture["loo_groups"],
            source="false_disjoint_claim",
        )

    # Generic nonnegative annotation weights remain useful internally, but they
    # may not be laundered into the strict disjoint public interpretation merely
    # because their SNP contributions were grouped.
    overlapping_reference = build_context_reference(
        **{**_core_build_kwargs(fixture), "annotations": overlapping},
        contribution_storage="loo_grouped",
        gram_method="exact",
        same_person_method="exact",
    )
    overlapping_summary = build_context_trait_summary(
        **{**_core_build_kwargs(fixture), "annotations": overlapping},
        phenotype=fixture["phenotype"],
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        contribution_storage="loo_grouped",
        block_size=5,
    )
    with pytest.raises(ValueError, match="(?i)strict|disjoint|partition|bound"):
        fit_annotation_context_model(overlapping_reference, overlapping_summary)


def test_empty_or_single_group_annotation_fails_before_misleading_fit() -> None:
    fixture = _fixture(k=2, group_count=6)
    annotations = np.asarray(fixture["annotations"]).copy()
    annotations[:, 0] = 1.0
    annotations[:, 1] = 0.0
    with pytest.raises(ValueError, match="(?i)empty|positive mass|annotation"):
        build_disjoint_annotation_partition(
            annotations,
            fixture["annotation_names"],
            variant_hash=fixture["variant_hash"],
            loo_groups=fixture["loo_groups"],
            source="empty_bin",
        )

    annotations = np.asarray(fixture["annotations"]).copy()
    groups = np.asarray(fixture["loo_groups"], dtype=object)
    first_group = groups[0]
    annotation_one = annotations[:, 1] == 1.0
    groups[annotation_one] = first_group
    with pytest.raises(ValueError, match="(?i)delete|retained|group|support|mass"):
        build_disjoint_annotation_partition(
            annotations,
            fixture["annotation_names"],
            variant_hash=fixture["variant_hash"],
            loo_groups=tuple(groups),
            source="single_group_bin",
        )


def test_group_balance_exposes_annotation_concentration_despite_balanced_totals() -> (
    None
):
    group_count = 4
    variants_per_group = 4
    groups = tuple(
        f"g{group}" for group in range(group_count) for _ in range(variants_per_group)
    )
    annotations = np.zeros((len(groups), 2), dtype=np.float64)
    annotations[: len(groups) // 2, 0] = 1.0
    annotations[len(groups) // 2 :, 1] = 1.0
    partition = build_disjoint_annotation_partition(
        annotations,
        ("low", "high"),
        variant_hash=canonical_sha256({"variants": list(range(len(groups)))}),
        loo_groups=groups,
        source="balanced_groups_concentrated_bins",
    )
    balance = partition.group_balance
    assert balance.total_groups_balanced
    assert balance.every_deletion_retains_each_annotation
    np.testing.assert_array_equal(balance.group_variant_counts, 4)
    np.testing.assert_array_equal(
        balance.group_annotation_masses,
        [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]],
    )
    np.testing.assert_array_equal(balance.zero_support_group_counts, [2, 2])
    np.testing.assert_allclose(balance.maximum_deleted_mass_share, [0.5, 0.5])
    np.testing.assert_allclose(balance.effective_group_count, [2.0, 2.0])


def test_annotation_contrast_uses_full_joint_jackknife_and_deleted_traces() -> None:
    fixture = _fixture(q=2, k=4, n=44)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    grid = np.asarray([[1.0, -1.0], [1.0, 0.0], [1.0, 1.5]])
    metric = np.asarray(fixture["basis"]).T @ np.asarray(fixture["basis"]) / 44.0
    fit = fit_annotation_context_model(
        grouped_reference,
        grouped_summary,
        partition=fixture["partition"],
        context_grid=grid,
        basis_metric=metric,
    )
    result = derive_annotation_contrast(
        fit,
        "bin:0",
        "bin:1",
        scale="total_component",
        reference=grouped_reference,
        summary=grouped_summary,
        context_grid=grid,
        basis_metric=metric,
    )
    assert isinstance(result, AnnotationContrastResult)
    left = _annotation_indices(fixture["components"], "bin:0")
    right = _annotation_indices(fixture["components"], "bin:1")
    expected = fit.genetic_coefficients[left] - fit.genetic_coefficients[right]
    expected_loo = fit.loo_coefficients[:, left] - fit.loo_coefficients[:, right]
    expected_covariance = _jackknife_covariance(expected_loo)
    np.testing.assert_allclose(result.coefficient_difference, expected)
    np.testing.assert_allclose(result.loo_coefficient_differences, expected_loo)
    np.testing.assert_allclose(result.covariance, expected_covariance)
    np.testing.assert_allclose(
        result.standard_errors, np.sqrt(np.maximum(np.diag(expected_covariance), 0.0))
    )

    full_covariance = fit.jackknife_covariance
    expected_from_joint = (
        full_covariance[np.ix_(left, left)]
        + full_covariance[np.ix_(right, right)]
        - full_covariance[np.ix_(left, right)]
        - full_covariance[np.ix_(right, left)]
    )
    np.testing.assert_allclose(result.covariance, expected_from_joint)

    expected_omega = coefficients_to_omegas(
        np.concatenate([expected, np.zeros(3 * expected.size)]),
        fixture["components"],
    )[0]
    np.testing.assert_allclose(result.omega_difference, expected_omega)
    np.testing.assert_allclose(
        result.surface_difference, grid @ expected_omega @ grid.T
    )

    full_traces = fit.equations.traces[: len(fixture["components"])]
    expected_trace = float(
        fit.genetic_coefficients[left] @ full_traces[left]
        - fit.genetic_coefficients[right] @ full_traces[right]
    )
    assert result.trace_difference == pytest.approx(expected_trace)
    replicate_trace = []
    for row, group in zip(fit.loo_coefficients, fit.jackknife_groups, strict=True):
        equations = assemble_context_normal_equations(
            grouped_reference, grouped_summary, (group,)
        )
        traces = equations.traces[: len(fixture["components"])]
        replicate_trace.append(row[left] @ traces[left] - row[right] @ traces[right])
    expected_trace_se = float(
        np.sqrt(_jackknife_covariance(np.asarray(replicate_trace)[:, None])[0, 0])
    )
    assert result.trace_standard_error == pytest.approx(expected_trace_se)
    assert "trace=defined" in result.status

    without_grouped_inputs = derive_annotation_contrast(
        fit, "bin:0", "bin:1", scale="total_component"
    )
    assert without_grouped_inputs.trace_difference is None
    assert without_grouped_inputs.trace_standard_error is None
    assert "trace=unavailable" in without_grouped_inputs.status


def test_annotation_derivations_reject_grouped_moments_from_another_partition() -> None:
    fixture = _fixture(q=2, k=4, n=44)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    fit = fit_annotation_context_model(
        grouped_reference, grouped_summary, partition=fixture["partition"]
    )

    alternative = dict(fixture)
    alternative_annotations = np.roll(
        np.asarray(fixture["annotations"]), shift=1, axis=1
    )
    alternative_partition = build_disjoint_annotation_partition(
        alternative_annotations,
        fixture["annotation_names"],
        definitions=fixture["definitions"],
        variant_hash=fixture["variant_hash"],
        loo_groups=fixture["loo_groups"],
        source="alternative_partition",
        source_digest=array_sha256(alternative_annotations),
    )
    alternative["annotations"] = alternative_annotations
    alternative["partition"] = alternative_partition
    alt_reference, alt_summary = _build_core_objects(alternative)
    alt_grouped_reference = group_context_reference(
        alt_reference, alternative_partition
    )
    alt_grouped_summary = group_context_trait_summary(
        alt_summary, alternative_partition
    )

    with pytest.raises(ValueError, match="(?i)partition|manifest|identity|mismatch"):
        derive_annotation_contrast(
            fit,
            "bin:0",
            "bin:1",
            reference=alt_grouped_reference,
            summary=alt_grouped_summary,
        )
    with pytest.raises(ValueError, match="(?i)partition|manifest|identity|mismatch"):
        derive_annotation_total(
            fit,
            reference=alt_grouped_reference,
            summary=alt_grouped_summary,
        )


def test_annotation_total_and_per_mass_contrast_have_exact_loo_maps() -> None:
    fixture = _fixture(q=2, k=4, n=44)
    reference, summary = _build_core_objects(fixture)
    grouped_reference = group_context_reference(reference, fixture["partition"])
    grouped_summary = group_context_trait_summary(summary, fixture["partition"])
    fit = fit_annotation_context_model(
        grouped_reference, grouped_summary, partition=fixture["partition"]
    )
    total = derive_annotation_total(
        fit, reference=grouped_reference, summary=grouped_summary
    )
    assert isinstance(total, AnnotationTotalResult)
    pair_count = len(fixture["components"].pair_index)
    reshaped = fit.genetic_coefficients.reshape(4, pair_count)
    loo_reshaped = fit.loo_coefficients[:, : 4 * pair_count].reshape(
        len(fit.jackknife_groups), 4, pair_count
    )
    np.testing.assert_allclose(total.coefficient_total, np.sum(reshaped, axis=0))
    np.testing.assert_allclose(
        total.loo_coefficient_totals, np.sum(loo_reshaped, axis=1)
    )
    np.testing.assert_allclose(
        total.covariance,
        _jackknife_covariance(np.sum(loo_reshaped, axis=1)),
    )
    assert total.trace_total == pytest.approx(
        float(
            fit.genetic_coefficients
            @ fit.equations.traces[: len(fixture["components"])]
        )
    )

    per_mass = derive_annotation_contrast(
        fit,
        "bin:0",
        "bin:1",
        scale="per_annotation_mass",
        reference=grouped_reference,
        summary=grouped_summary,
    )
    masses = np.asarray(fixture["partition"].annotation_masses)
    expected = (
        fit.genetic_coefficients[:pair_count] / masses[0]
        - fit.genetic_coefficients[pair_count : 2 * pair_count] / masses[1]
    )
    np.testing.assert_allclose(per_mass.coefficient_difference, expected)
    expected_loo = []
    for row, group in zip(fit.loo_coefficients, fit.jackknife_groups, strict=True):
        retained = reference_moments_after_deleting_groups(
            grouped_reference, (group,)
        ).annotation_masses
        expected_loo.append(
            row[:pair_count] / retained[0]
            - row[pair_count : 2 * pair_count] / retained[1]
        )
    np.testing.assert_allclose(per_mass.loo_coefficient_differences, expected_loo)
    assert per_mass.scale == "per_annotation_mass"
    with pytest.raises(ValueError, match="(?i)scale"):
        derive_annotation_contrast(fit, "bin:0", "bin:1", scale="enrichment")


def test_resource_estimate_matches_declared_dimensions_storage_and_probe_costs() -> (
    None
):
    scalar = 8
    n_reference = 300_000
    n_study = 250_000
    n_variants = 1_000_000
    k_count = 8
    q_count = 4
    h_count = 3
    group_count = 100
    tile_size = 32
    request = AnnotationResourceRequest(
        n_reference=n_reference,
        n_study=n_study,
        n_variants=n_variants,
        q=q_count,
        k=k_count,
        h=h_count,
        gram_probes=128,
        same_person_probes=64,
        probe_tile_size=tile_size,
        loo_groups=group_count,
        dtype_bytes=scalar,
        memory_cap_bytes=10_000_000_000,
    )
    result = estimate_annotation_resources(request)
    assert isinstance(result, AnnotationResourceEstimate)
    pair_count = q_count * (q_count + 1) // 2
    assert pair_count == 10
    assert result.p_genetic == 80
    assert result.p_total == 83
    assert result.selected_probe_tile_size == tile_size
    assert result.buffer_bytes["normal_matrix"] == scalar * 83 * 83
    assert result.buffer_bytes["operator_action_tile"] == (
        scalar * n_reference * tile_size * 80
    )
    # U_q=G' D_q PZ is M x b for each source basis q.  Target H_kr
    # may be streamed one r at a time, leaving K N x b targets resident.
    assert result.buffer_bytes["operator_source_tile"] == (
        scalar * n_variants * tile_size * q_count
    )
    assert result.buffer_bytes["operator_annotation_target_tile"] == (
        scalar * n_reference * tile_size * k_count
    )
    assert result.buffer_bytes["same_person_persistent"] == scalar * (
        n_reference * 80 + 80 * 80
    )

    storage = result.storage_bytes
    grouped_reference_contribution = scalar * group_count * 80 * 80
    snp_reference_contribution = scalar * n_variants * 80 * 80
    grouped_trait_contribution = scalar * group_count * 80 * (2 + h_count)
    snp_trait_contribution = scalar * n_variants * 80 * (2 + h_count)
    assert storage["grouped_reference_contribution_bytes"] == (
        grouped_reference_contribution
    )
    assert storage["snp_reference_contribution_bytes"] == snp_reference_contribution
    assert storage["grouped_trait_contribution_bytes"] == grouped_trait_contribution
    assert storage["snp_trait_contribution_bytes"] == snp_trait_contribution
    # Complete numeric sufficient-statistic storage also includes masses,
    # group counts, full moments, and the residual block.
    assert storage["grouped_reference_total_bytes"] == (
        grouped_reference_contribution
        + scalar * (group_count * k_count + group_count + k_count + 2 * 80 * 80)
    )
    assert storage["snp_reference_total_bytes"] == (
        snp_reference_contribution
        + scalar * (n_variants * k_count + k_count + 2 * 80 * 80)
    )
    assert storage["grouped_trait_total_bytes"] == (
        grouped_trait_contribution
        + scalar
        * (
            group_count * k_count
            + group_count
            + k_count
            + 80 * (2 + h_count)
            + h_count * h_count
            + 2 * h_count
        )
    )
    assert storage["snp_trait_total_bytes"] == (
        snp_trait_contribution
        + scalar
        * (
            n_variants * k_count
            + k_count
            + 80 * (2 + h_count)
            + h_count * h_count
            + 2 * h_count
        )
    )
    assert result.operation_counts["probe_tiles"] == 4
    assert result.operation_counts["source_genotype_products"] == q_count * 4
    assert result.operation_counts["annotation_target_genotype_products"] == (
        k_count * q_count * 4
    )
    assert result.operation_counts["grouped_numerator_source_products"] == (
        q_count * 80 * 4
    )
    assert result.dominant_flop_proxies["grouped_numerator_source_products"] == (
        2 * n_reference * n_variants * q_count * 80 * request.gram_probes
    )
    assert result.condition_status == "unknown_without_pilot"
    assert result.pilot_condition_number is None
    assert result.within_memory_cap

    too_small = estimate_annotation_resources(
        replace(request, memory_cap_bytes=result.buffer_bytes["normal_matrix"])
    )
    assert not too_small.within_memory_cap
    assert too_small.selected_probe_tile_size == 0
    assert too_small.verdict == "operator_memory_cap_exceeded"

    pilot = np.diag(np.geomspace(1.0, 1.0e-7, 83))
    with_pilot = estimate_annotation_resources(request, pilot_normal_matrix=pilot)
    assert with_pilot.condition_status == "pilot_full_rank"
    assert with_pilot.pilot_condition_number == pytest.approx(1.0e7)
    scaled_pilot = estimate_annotation_resources(
        request, pilot_normal_matrix=1.0e-24 * pilot
    )
    assert scaled_pilot.pilot_rank == with_pilot.pilot_rank
    assert scaled_pilot.pilot_condition_number == pytest.approx(
        with_pilot.pilot_condition_number
    )
    with pytest.raises(ValueError, match="(?i)shape|dimension"):
        estimate_annotation_resources(
            request, pilot_normal_matrix=np.eye(request.q + request.h)
        )
    with pytest.raises(ValueError, match="(?i)q|at most 4"):
        estimate_annotation_resources(replace(request, q=5))


def test_partition_round_trip_is_schema_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    stem = tmp_path / "partition"
    manifest_path, arrays_path = write_disjoint_annotation_partition(
        fixture["partition"], stem
    )
    loaded = load_disjoint_annotation_partition(
        manifest_path, expected_partition_hash=fixture["partition"].digest
    )
    assert loaded.manifest == fixture["partition"].manifest
    np.testing.assert_array_equal(loaded.annotations, fixture["partition"].annotations)
    validate_disjoint_annotation_partition(loaded)
    with pytest.raises(ValueError, match="(?i)identity|partition|unexpected"):
        load_disjoint_annotation_partition(
            manifest_path, expected_partition_hash=canonical_sha256({"wrong": True})
        )

    schema_manifest, _ = write_disjoint_annotation_partition(
        fixture["partition"], tmp_path / "partition-schema"
    )
    payload = json.loads(schema_manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "summit.context.annotations.invalid"
    schema_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)schema|version"):
        load_disjoint_annotation_partition(schema_manifest)

    tamper_manifest, tamper_arrays = write_disjoint_annotation_partition(
        fixture["partition"], tmp_path / "partition-array"
    )
    payload = json.loads(tamper_manifest.read_text(encoding="utf-8"))
    arrays_path = tamper_arrays
    with np.load(arrays_path, allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    arrays["weights"][0, 0] = 0.0
    np.savez_compressed(arrays_path, **arrays)
    payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)
    )
    tamper_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)hash|partition|annotation|reconstruct"):
        load_disjoint_annotation_partition(tamper_manifest)

    extra_manifest, extra_arrays = write_disjoint_annotation_partition(
        fixture["partition"], tmp_path / "partition-extra"
    )
    payload = json.loads(extra_manifest.read_text(encoding="utf-8"))
    with np.load(extra_arrays, allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    arrays["undeclared"] = np.asarray([1.0])
    np.savez_compressed(extra_arrays, **arrays)
    payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(extra_arrays.read_bytes(), dtype=np.uint8)
    )
    extra_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)array|artifact|schema|invalid"):
        load_disjoint_annotation_partition(extra_manifest)


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("masses", lambda payload: payload["masses"].__setitem__(0, 999.0)),
        (
            "group_variant_counts",
            lambda payload: payload["group_variant_counts"].__setitem__(0, 999),
        ),
        (
            "boundary_convention",
            lambda payload: payload.__setitem__(
                "boundary_convention", "silently_changed"
            ),
        ),
        (
            "balance",
            lambda payload: payload["balance"].__setitem__(
                "maximum_deleted_mass_share", [0.0] * len(payload["masses"])
            ),
        ),
    ],
)
def test_partition_loader_rejects_tampered_scientific_manifest_claims(
    tmp_path: Path, field: str, mutate
) -> None:
    fixture = _fixture()
    manifest_path, _ = write_disjoint_annotation_partition(
        fixture["partition"], tmp_path / f"partition-{field}"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="(?i)manifest|mass|group|balance|boundary|partition|mismatch"
    ):
        load_disjoint_annotation_partition(manifest_path)


@pytest.mark.parametrize("artifact", ["reference", "summary"])
def test_grouped_artifact_round_trip_and_array_tamper_detection(
    tmp_path: Path, artifact: str
) -> None:
    fixture = _fixture(q=2, k=4)
    reference, summary = _build_core_objects(fixture)
    if artifact == "reference":
        value = group_context_reference(reference, fixture["partition"])
        prefix = tmp_path / "grouped-reference"
        manifest_path, arrays_path = write_context_reference(value, prefix)
        loaded = load_context_reference(
            manifest_path,
            expected={"annotation_partition_hash": fixture["partition"].digest},
        )
        observed = loaded.group_gram_numerator_contributions
        expected = value.group_gram_numerator_contributions
    else:
        value = group_context_trait_summary(summary, fixture["partition"])
        prefix = tmp_path / "grouped-summary"
        manifest_path, arrays_path = write_context_trait_summary(value, prefix)
        loaded = load_context_trait_summary(
            manifest_path,
            expected={"annotation_partition_hash": fixture["partition"].digest},
        )
        observed = loaded.group_rhs_numerator_contributions
        expected = value.group_rhs_numerator_contributions
    assert loaded.manifest == value.manifest
    np.testing.assert_array_equal(
        loaded.group_variant_counts, value.group_variant_counts
    )
    np.testing.assert_array_equal(
        loaded.group_annotation_masses, value.group_annotation_masses
    )
    np.testing.assert_array_equal(observed, expected)
    _assert_no_variant_axis(loaded, np.asarray(fixture["genotype"]).shape[1])
    wrong_expected = {"annotation_partition_hash": canonical_sha256({"wrong": True})}
    loader = (
        load_context_reference
        if artifact == "reference"
        else load_context_trait_summary
    )
    with pytest.raises(ValueError, match="(?i)manifest|mismatch|partition|expected"):
        loader(manifest_path, expected=wrong_expected)

    with np.load(arrays_path, allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    target = (
        "gram_numerator_contributions"
        if artifact == "reference"
        else "rhs_numerator_contributions"
    )
    arrays[target].flat[0] += 1.0
    np.savez_compressed(arrays_path, **arrays)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)hash|digest|reconstruct|contribution"):
        loader(manifest_path)

    payload["schema_version"] = "invalid-grouped-schema"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)schema|version|hash|digest"):
        loader(manifest_path)


@pytest.mark.parametrize("artifact", ["reference", "summary"])
def test_grouped_artifact_rejects_extra_arrays_and_asymmetric_moments(
    tmp_path: Path, artifact: str
) -> None:
    fixture = _fixture(q=2, k=4)
    reference, summary = _build_core_objects(fixture)
    value = (
        group_context_reference(reference, fixture["partition"])
        if artifact == "reference"
        else group_context_trait_summary(summary, fixture["partition"])
    )
    writer = (
        write_context_reference
        if artifact == "reference"
        else write_context_trait_summary
    )
    loader = (
        load_context_reference
        if artifact == "reference"
        else load_context_trait_summary
    )

    extra_manifest, extra_arrays = writer(value, tmp_path / f"{artifact}-extra")
    with np.load(extra_arrays, allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    arrays["undeclared"] = np.asarray([1.0])
    np.savez_compressed(extra_arrays, **arrays)
    payload = json.loads(extra_manifest.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(extra_arrays.read_bytes(), dtype=np.uint8)
    )
    extra_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)array|artifact|schema|unexpected"):
        loader(extra_manifest)

    symmetry_manifest, symmetry_arrays = writer(
        value, tmp_path / f"{artifact}-asymmetric"
    )
    with np.load(symmetry_arrays, allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    if artifact == "reference":
        delta = 0.125
        arrays["gram"][0, 1] += delta
        component_mass = float(value.annotation_masses[0])
        arrays["gram_numerator_contributions"][0, 0, 1] += (
            delta * component_mass * component_mass
        )
    else:
        arrays["residual_gram"][0, 1] += 0.125
    np.savez_compressed(symmetry_arrays, **arrays)
    payload = json.loads(symmetry_manifest.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = array_sha256(
        np.frombuffer(symmetry_arrays.read_bytes(), dtype=np.uint8)
    )
    symmetry_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)symmetr|matrix|gram|moment"):
        loader(symmetry_manifest)


def test_in_memory_partition_manifest_and_balance_tampering_is_rejected() -> None:
    fixture = _fixture()
    partition = fixture["partition"]
    altered_manifest = copy.deepcopy(partition.manifest)
    altered_manifest["annotation_names"] = list(reversed(partition.annotation_names))
    with pytest.raises(ValueError, match="(?i)order|hash|manifest"):
        validate_disjoint_annotation_partition(
            replace(partition, manifest=altered_manifest)
        )

    altered_manifest = copy.deepcopy(partition.manifest)
    altered_manifest["ordered_bins"][0]["definition"]["index"] = 999
    with pytest.raises(ValueError, match="(?i)definition|bin|manifest|hash"):
        validate_disjoint_annotation_partition(
            replace(partition, manifest=altered_manifest)
        )
    altered_balance = replace(
        partition.group_balance,
        group_variant_counts=partition.group_balance.group_variant_counts + 1,
    )
    with pytest.raises(ValueError, match="(?i)group|balance|count|hash"):
        validate_disjoint_annotation_partition(
            replace(partition, balance=altered_balance)
        )
