from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from generalized_gxe_variant_ldscore_oracle import (
    exact_component_kernel_diagonal,
    exact_same_person_matrix,
    orthonormalize,
)
from summit.context.oracle import transfer_reference_gram
from summit.ldscore.generalized_gxe_fit_v1 import (
    assemble_generalized_gxe_normal_equations_v1,
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


def _inputs(seed: int = 20260831):
    rng = np.random.default_rng(seed)
    n, m = 19, 24
    environment = rng.normal(size=n)
    genotype = np.asfortranarray(
        rng.normal(size=(n, m)) + 0.25 * environment[:, None]
    )
    basis = np.asfortranarray(
        np.column_stack((np.ones(n), environment))
    )
    fixed = orthonormalize(
        np.column_stack((np.ones(n), rng.normal(size=n)))
    )
    annotations = np.column_stack(
        (
            np.ones(m),
            (np.arange(m) % 3 != 0).astype(np.float64),
            rng.uniform(0.15, 1.25, size=m),
        )
    )
    names = ("baseline", "tissue_a", "tissue_b")
    block_ids = np.arange(m, dtype=np.int64) % 4
    block_labels = tuple(f"block_{index}" for index in range(4))
    probe_spec = GlobalVariantProbeSpec(88173, 5, 17)
    return (
        rng,
        genotype,
        basis,
        fixed,
        annotations,
        names,
        block_ids,
        block_labels,
        probe_spec,
    )


def _run_reference(
    genotype: np.ndarray,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    names: tuple[str, ...],
    block_ids: np.ndarray,
    block_labels: tuple[str, ...],
    probe_spec: GlobalVariantProbeSpec,
):
    plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=basis.shape[1],
            num_annotations=annotations.shape[1],
            num_probes=probe_spec.probe_count,
            memory_limit_bytes=512 * 1024**2,
            genotype_format="bed",
            fixed_effect_rank=fixed.shape[1],
            preferred_variant_block_width=5,
            preferred_rhs_tile_columns=3 * basis.shape[1] ** 2,
            rhs_policy="tiled",
        )
    )
    operator = ArraySequentialGenotypeOperator(genotype)
    masses = np.sum(annotations, axis=0, dtype=np.float64)
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        annotation_masses=masses,
        probe_spec=probe_spec,
        work_plan=plan,
        nn_operator=NumpyNNOperator(),
        annotation_tile_width=annotations.shape[1],
        probe_tile_width=4,
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
        component_diagonal_sample_tile_width=6,
    ).execute()
    block_directed, block_masses, reconstruction_error = (
        reduce_generalized_gxe_reference_for_inference(
            directional_ldscores=result.directional_ldscores,
            annotations=annotations,
            variant_block_ids=block_ids,
            block_labels=block_labels,
        )
    )
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=genotype.shape[1],
        num_samples=genotype.shape[0],
        basis_names=tuple(f"basis_{index}" for index in range(basis.shape[1])),
        fixed_effect_rank=fixed.shape[1],
        annotation_names=names,
        annotation_masses=masses,
        variant_block_ids=block_ids,
        block_labels=block_labels,
        residual_component_names=("identity",),
    )
    phases = {
        name: 0.0 for name in ("pass1", "barrier", "pass2", "finalize")
    }
    artifact = build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe_spec,
        genotype_scale_plan={
            "genotype_scale_policy": "fixture_standardized_v1",
            "allele_orientation": "fixture_a1_v1",
        },
        arrays={
            "directed_numerator": result.directed_numerator,
            "symmetric_numerator": result.symmetric_numerator,
            "genetic_gram": result.genetic_gram,
            "block_directed_numerator": block_directed,
            "block_annotation_mass": block_masses,
            "same_person": result.same_person,
            "directional_ldscores": result.directional_ldscores,
            "annotations": annotations,
            "component_kernel_diagonal": (
                result.component_kernel_diagonal
            ),
        },
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
        provenance={"fixture": "annotation_composition"},
        diagnostics={"block_reconstruction_error": reconstruction_error},
        mode="composable",
    )
    return artifact, result


@pytest.mark.parametrize("q_count", (1, 2, 3))
@pytest.mark.parametrize("annotation_kind", ("disjoint", "overlap", "continuous"))
def test_exact_component_diagonals_cover_annotation_geometries(
    q_count: int, annotation_kind: str
) -> None:
    rng = np.random.default_rng(9100 + 10 * q_count)
    n, m = 13, 15
    genotype = np.asfortranarray(rng.normal(size=(n, m)))
    columns = [np.ones(n), rng.normal(size=n), rng.normal(size=n)]
    basis = np.asfortranarray(np.column_stack(columns[:q_count]))
    fixed = orthonormalize(
        np.column_stack((np.ones(n), rng.normal(size=n)))
    )
    index = np.arange(m)
    if annotation_kind == "disjoint":
        annotations = np.column_stack((index % 2 == 0, index % 2 == 1))
    elif annotation_kind == "overlap":
        annotations = np.column_stack((index % 2 == 0, index % 3 != 0))
    else:
        annotations = rng.uniform(0.1, 1.3, size=(m, 2))
    annotations = np.asarray(annotations, dtype=np.float64)
    names = ("a", "b")
    blocks = index % 3
    artifact, result = _run_reference(
        genotype,
        basis,
        fixed,
        annotations,
        names,
        blocks,
        ("b0", "b1", "b2"),
        GlobalVariantProbeSpec(9119, 0, 9),
    )
    expected = exact_component_kernel_diagonal(
        genotype, basis, fixed, annotations
    )
    np.testing.assert_allclose(
        result.component_kernel_diagonal,
        expected,
        rtol=1.2e-13,
        atol=1.2e-13,
    )
    np.testing.assert_allclose(
        artifact.same_person,
        exact_same_person_matrix(genotype, basis, fixed, annotations),
        rtol=1.5e-13,
        atol=1.5e-13,
    )


def test_separate_annotation_bundles_compose_to_joint_fit_without_genotypes(
    tmp_path: Path,
) -> None:
    (
        rng,
        genotype,
        basis,
        fixed,
        annotations,
        names,
        block_ids,
        block_labels,
        probe_spec,
    ) = _inputs()
    joint, joint_result = _run_reference(
        genotype,
        basis,
        fixed,
        annotations,
        names,
        block_ids,
        block_labels,
        probe_spec,
    )
    baseline, _ = _run_reference(
        genotype,
        basis,
        fixed,
        annotations[:, :1],
        names[:1],
        block_ids,
        block_labels,
        probe_spec,
    )
    tissues, _ = _run_reference(
        genotype,
        basis,
        fixed,
        annotations[:, 1:],
        names[1:],
        block_ids,
        block_labels,
        probe_spec,
    )
    composed = compose_generalized_gxe_variant_references_v1(
        ((baseline, ("baseline",)), (tissues, ("tissue_a", "tissue_b"))),
        variant_block_ids=block_ids,
        block_labels=block_labels,
        mode="composable",
    )
    for name in (
        "directional_ldscores",
        "directed_numerator",
        "symmetric_numerator",
        "genetic_gram",
        "component_kernel_diagonal",
        "same_person",
        "block_directed_numerator",
        "block_annotation_mass",
    ):
        np.testing.assert_allclose(
            getattr(composed, name),
            getattr(joint, name),
            rtol=2.0e-13,
            atol=2.0e-13,
        )
    assert composed.manifest["pass_ledger"][
        "genotype_passes_during_composition"
    ] == 0
    assert joint_result.ledger.observed_reference_genotype_passes == 2
    assert joint_result.ledger.observed_retained_variant_visits == (
        2 * genotype.shape[1]
    )

    per_variant = generalized_gxe_per_variant_trait_statistics(
        genotype=genotype,
        basis=basis,
        fixed_basis=fixed,
        phenotypes=rng.normal(size=(genotype.shape[0], 1)),
        residual_basis=np.ones((genotype.shape[0], 1)),
    )
    baseline_trait = aggregate_generalized_gxe_trait_statistics(
        per_variant,
        annotations=annotations[:, :1],
        annotation_names=names[:1],
        variant_group_ids=block_ids,
        group_labels=block_labels,
        trait_ids=("trait",),
        residual_names=("identity",),
        n_samples=genotype.shape[0],
    )
    trait_path = write_generalized_gxe_trait_summary(
        baseline_trait, tmp_path / "trait"
    )
    reaggregated = reaggregate_generalized_gxe_trait_summary(
        load_generalized_gxe_trait_summary(trait_path),
        annotations=annotations,
        annotation_names=names,
        variant_group_ids=block_ids,
        group_labels=block_labels,
    )
    direct_trait = aggregate_generalized_gxe_trait_statistics(
        per_variant,
        annotations=annotations,
        annotation_names=names,
        variant_group_ids=block_ids,
        group_labels=block_labels,
        trait_ids=("trait",),
        residual_names=("identity",),
        n_samples=genotype.shape[0],
    )
    np.testing.assert_allclose(
        reaggregated.genetic_rhs, direct_trait.genetic_rhs
    )
    composed_fit = fit_generalized_gxe_variant_model_v1(
        composed, reaggregated
    )
    joint_fit = fit_generalized_gxe_variant_model_v1(joint, direct_trait)
    np.testing.assert_allclose(
        composed_fit.raw_coefficients,
        joint_fit.raw_coefficients,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    equal_n_equations = assemble_generalized_gxe_normal_equations_v1(
        composed, reaggregated
    )
    np.testing.assert_allclose(
        equal_n_equations.reference_genetic_gram,
        composed.genetic_gram,
        rtol=0.0,
        atol=0.0,
    )
    study_n = composed.n_samples + 7
    expected_transport = (
        (study_n / composed.n_samples) * composed.same_person
        + study_n
        * (study_n - 1)
        / (composed.n_samples * (composed.n_samples - 1))
        * (composed.genetic_gram - composed.same_person)
    )
    np.testing.assert_allclose(
        transfer_reference_gram(
            composed.genetic_gram,
            composed.same_person,
            reference_n=composed.n_samples,
            study_n=study_n,
        ),
        expected_transport,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    for block in block_labels:
        left = reference_moments_after_deleting_variant_blocks_v1(
            composed, (block,)
        )
        right = reference_moments_after_deleting_variant_blocks_v1(
            joint, (block,)
        )
        np.testing.assert_allclose(left.gram, right.gram, rtol=2.0e-13)
        np.testing.assert_array_equal(left.same_person, composed.same_person)
        np.testing.assert_array_equal(right.same_person, joint.same_person)


def test_summary_mode_omits_private_arrays_and_composable_writer_warns(
    tmp_path: Path,
) -> None:
    (
        _rng,
        genotype,
        basis,
        fixed,
        annotations,
        names,
        block_ids,
        block_labels,
        probe_spec,
    ) = _inputs(20260901)
    composable, _ = _run_reference(
        genotype,
        basis,
        fixed,
        annotations,
        names,
        block_ids,
        block_labels,
        probe_spec,
    )
    with pytest.warns(UserWarning, match="should not be publicly shared"):
        private_path = write_generalized_gxe_variant_reference_v1(
            composable, tmp_path / "private"
        )
    with np.load(private_path, allow_pickle=False) as archive:
        assert "annotations" in archive.files
        assert "component_kernel_diagonal" in archive.files

    summary = compose_generalized_gxe_variant_references_v1(
        ((composable, names),),
        variant_block_ids=block_ids,
        block_labels=block_labels,
        mode="summary",
    )
    assert summary.annotations is None
    assert summary.component_kernel_diagonal is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        public_path = write_generalized_gxe_variant_reference_v1(
            summary, tmp_path / "public"
        )
    with np.load(public_path, allow_pickle=False) as archive:
        assert "annotations" not in archive.files
        assert "component_kernel_diagonal" not in archive.files
