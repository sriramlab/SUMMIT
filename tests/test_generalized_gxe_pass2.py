from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from generalized_gxe_variant_ldscore_oracle import (
    exact_component_kernel_diagonal,
    exact_recomputed_delete_block_grams,
    exact_same_person_matrix,
    frozen_ldscore_delete_block,
    orthonormalize,
    randomized_two_pass_ldscores,
)
from summit.ldscore.generalized_gxe_pass1 import (
    ArraySequentialGenotypeOperator,
    GeneralizedGxEPass1Executor,
    NumpyNNOperator,
    ProtectedNNOperator,
)
from summit.ldscore.generalized_gxe_pass2 import (
    GeneralizedGxEPass2Executor,
    NumpyTNOperator,
    ProtectedTNOperator,
    build_pair_product_plan,
)
from summit.ldscore.generalized_gxe_reference_v1 import (
    reduce_generalized_gxe_reference_for_inference,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


def _fixture(
    num_basis: int,
    num_annotations: int,
    *,
    seed: int,
    probe_count: int = 23,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n_samples, n_variants = 11, 10
    environment = rng.normal(size=n_samples)
    covariate = rng.normal(size=n_samples)
    fixed = orthonormalize(
        np.column_stack([np.ones(n_samples), covariate])
    )
    genotype = np.asfortranarray(
        rng.normal(size=(n_samples, n_variants))
        + 0.4 * environment[:, None]
        + 0.2 * covariate[:, None]
    )
    basis_columns = [np.ones(n_samples), environment]
    if num_basis >= 3:
        basis_columns.append(
            0.3 * environment**2 + rng.normal(size=n_samples)
        )
    basis = np.asfortranarray(np.column_stack(basis_columns[:num_basis]))
    annotations = rng.uniform(
        0.1, 1.4, size=(n_variants, num_annotations)
    )
    spec = GlobalVariantProbeSpec(
        root_seed=seed,
        probe_offset=7,
        probe_count=probe_count,
    )
    probes = spec.generate(np.arange(n_variants, dtype=np.int64))
    return genotype, basis, fixed, annotations, spec, probes


def _block_ids(num_variants: int, block_count: int) -> np.ndarray:
    if block_count < 2 or block_count > num_variants:
        raise ValueError("invalid fixture block count")
    return np.repeat(
        np.arange(block_count, dtype=np.int64),
        np.diff(
            np.linspace(0, num_variants, block_count + 1, dtype=np.int64)
        ),
    )


def _plan(
    genotype: np.ndarray,
    basis: np.ndarray,
    annotations: np.ndarray,
    probe_count: int,
    *,
    variant_width: int,
    rhs_probe_width: int,
    rhs_policy: str,
    threads: int = 1,
):
    q_squared = basis.shape[1] ** 2
    preferred_rhs = q_squared * rhs_probe_width
    return plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=basis.shape[1],
            num_annotations=annotations.shape[1],
            num_probes=probe_count,
            memory_limit_bytes=512 * 1024**2,
            genotype_format="bed",
            threads=threads,
            preferred_variant_block_width=variant_width,
            preferred_rhs_tile_columns=preferred_rhs,
            rhs_policy=rhs_policy,
        )
    )


def _run_two_pass(
    *,
    genotype: np.ndarray,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    spec: GlobalVariantProbeSpec,
    variant_width: int,
    rhs_probe_width: int,
    rhs_policy: str = "tiled",
    pass1_probe_width: int = 5,
    threads: int = 1,
    row_complete_sink=None,
):
    plan = _plan(
        genotype,
        basis,
        annotations,
        spec.probe_count,
        variant_width=variant_width,
        rhs_probe_width=rhs_probe_width,
        rhs_policy=rhs_policy,
        threads=threads,
    )
    operator = ArraySequentialGenotypeOperator(genotype)
    names = tuple(f"annotation_{index}" for index in range(annotations.shape[1]))
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        annotation_masses=np.sum(annotations, axis=0),
        probe_spec=spec,
        work_plan=plan,
        nn_operator=NumpyNNOperator(threads=threads),
        annotation_tile_width=annotations.shape[1],
        probe_tile_width=min(pass1_probe_width, spec.probe_count),
        same_person_sample_tile_width=4,
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
        tn_operator=NumpyTNOperator(threads=threads),
        probe_tile_width=(
            spec.probe_count if rhs_policy == "precompute" else rhs_probe_width
        ),
        row_complete_sink=row_complete_sink,
    ).execute()
    return result, pass1, operator, plan


def test_pair_product_plan_derives_one_two_four_from_orientations_only() -> None:
    plan = build_pair_product_plan(3)
    orientation_counts = np.asarray([1, 1, 1, 2, 2, 2])
    expected = orientation_counts[:, None] * orientation_counts[None, :]
    np.testing.assert_array_equal(np.asarray(plan.multiplicities), expected)
    assert plan.pairs == (
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 1),
        (0, 2),
        (1, 2),
    )
    for target_pair, row in enumerate(plan.terms):
        for source_pair, terms in enumerate(row):
            assert len(terms) == expected[target_pair, source_pair]
            assert len(set(term.to_tuple() for term in terms)) == len(terms)


@pytest.mark.parametrize(
    ("num_basis", "num_annotations", "seed"),
    ((1, 1, 5101), (2, 1, 5201), (3, 1, 5301), (3, 2, 5302)),
)
def test_complete_pass2_outputs_match_every_dense_oracle_layer(
    num_basis: int,
    num_annotations: int,
    seed: int,
) -> None:
    genotype, basis, fixed, annotations, spec, probes = _fixture(
        num_basis, num_annotations, seed=seed
    )
    blocks = _block_ids(genotype.shape[1], 3)
    observed, pass1, operator, _plan_value = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=4,
        rhs_probe_width=3,
    )
    expected, expected_sources, _cross = randomized_two_pass_ldscores(
        genotype, basis, fixed, annotations, probes
    )
    np.testing.assert_allclose(
        observed.directional_ldscores,
        expected.directional_ldscores,
        rtol=8.0e-14,
        atol=8.0e-14,
    )
    np.testing.assert_allclose(
        observed.directed_numerator,
        expected.directed_numerator,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        observed.symmetric_numerator,
        expected.symmetric_numerator,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        observed.genetic_gram, expected.gram, rtol=1.2e-13, atol=1.2e-13
    )
    block_directed, block_masses, error = (
        reduce_generalized_gxe_reference_for_inference(
            directional_ldscores=observed.directional_ldscores,
            annotations=annotations,
            variant_block_ids=blocks,
            block_labels=("left", "middle", "right"),
        )
    )
    _deleted, expected_blocks, expected_masses = frozen_ldscore_delete_block(
        expected, annotations, blocks
    )
    np.testing.assert_allclose(block_directed, expected_blocks, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(block_masses, expected_masses, rtol=3.0e-15, atol=3.0e-15)
    assert error < 1.0e-12
    np.testing.assert_allclose(
        pass1.contextual_sources,
        expected_sources,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    expected_diagonal = exact_component_kernel_diagonal(
        genotype, basis, fixed, annotations
    )
    np.testing.assert_allclose(
        observed.component_kernel_diagonal,
        expected_diagonal,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        observed.same_person,
        exact_same_person_matrix(genotype, basis, fixed, annotations),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    observed.ledger.validate_clean_completion()
    assert observed.ledger.observed_reference_genotype_passes == 2
    assert observed.ledger.observed_retained_variant_visits == 2 * genotype.shape[1]
    assert operator.observed_passes == 2
    assert operator.observed_variant_visits == 2 * genotype.shape[1]


@pytest.mark.parametrize(
    ("variant_width", "rhs_probe_width", "rhs_policy"),
    ((2, 1, "tiled"), (4, 4, "tiled"), (10, 23, "precompute")),
)
def test_pass2_is_invariant_to_rhs_plans(
    variant_width: int,
    rhs_probe_width: int,
    rhs_policy: str,
) -> None:
    genotype, basis, fixed, annotations, spec, probes = _fixture(
        3, 2, seed=5402
    )
    observed, _pass1, operator, plan = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=variant_width,
        rhs_probe_width=rhs_probe_width,
        rhs_policy=rhs_policy,
    )
    expected, _sources, _cross = randomized_two_pass_ldscores(
        genotype, basis, fixed, annotations, probes
    )
    np.testing.assert_allclose(
        observed.directional_ldscores,
        expected.directional_ldscores,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert operator.observed_variant_visits == 2 * genotype.shape[1]
    assert observed.telemetry["backend"]["rhs_precomputed"] is (
        rhs_policy == "precompute"
    )
    assert plan.tiling["rhs_tile_columns"] >= basis.shape[1] ** 2


def test_q2_fixture_matches_mature_xw_orientation_multiplicities() -> None:
    genotype, basis, fixed, annotations, spec, probes = _fixture(
        2, 1, seed=5522
    )
    blocks = _block_ids(genotype.shape[1], 2)
    observed, _pass1, _operator, _plan_value = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=6,
        rhs_probe_width=5,
    )
    expected, _sources, _cross = randomized_two_pass_ldscores(
        genotype, basis, fixed, annotations, probes
    )
    assert build_pair_product_plan(2).multiplicities == (
        (1, 1, 2),
        (1, 1, 2),
        (2, 2, 4),
    )
    np.testing.assert_allclose(
        observed.directional_ldscores,
        expected.directional_ldscores,
        rtol=8.0e-14,
        atol=8.0e-14,
    )


def test_j_changes_only_posthoc_reductions_without_rescanning() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture(
        3, 2, seed=5602
    )
    result, _pass1, operator, _plan_value = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=4,
        rhs_probe_width=3,
    )
    reductions = []
    for block_count in (2, 5):
        blocks = _block_ids(genotype.shape[1], block_count)
        block_directed, block_masses, error = reduce_generalized_gxe_reference_for_inference(
            directional_ldscores=result.directional_ldscores,
            annotations=annotations,
            variant_block_ids=blocks,
            block_labels=tuple(f"block_{index}" for index in range(block_count)),
        )
        reductions.append((block_directed, block_masses))
        assert error < 1.0e-12
    assert reductions[0][0].shape[0] == 2
    assert reductions[1][0].shape[0] == 5
    for block_directed, block_masses in reductions:
        np.testing.assert_allclose(np.sum(block_directed, axis=0), result.directed_numerator)
        np.testing.assert_allclose(np.sum(block_masses, axis=0), np.sum(annotations, axis=0))
    assert result.ledger.observed_reference_genotype_passes == 2
    assert operator.observed_variant_visits == 2 * genotype.shape[1]


def test_frozen_row_deletion_keeps_full_scores_sources_and_same_person() -> None:
    genotype, basis, fixed, annotations, spec, probes = _fixture(
        3, 2, seed=5702, probe_count=37
    )
    blocks = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    observed, pass1, _operator, _plan_value = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=4,
        rhs_probe_width=4,
    )
    expected, _sources, _cross = randomized_two_pass_ldscores(
        genotype, basis, fixed, annotations, probes
    )
    fixed_deleted, expected_block_num, expected_block_mass = frozen_ldscore_delete_block(
        expected, annotations, blocks
    )
    block_num, block_mass, error = reduce_generalized_gxe_reference_for_inference(
        directional_ldscores=observed.directional_ldscores,
        annotations=annotations,
        variant_block_ids=blocks,
        block_labels=("left", "middle", "right"),
    )
    np.testing.assert_allclose(block_num, expected_block_num, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(block_mass, expected_block_mass, rtol=3.0e-15, atol=3.0e-15)
    assert error < 1.0e-12
    exact_deleted = exact_recomputed_delete_block_grams(
        genotype, basis, fixed, annotations, blocks
    )
    assert np.max(np.abs(fixed_deleted - exact_deleted)) > 1.0e-4
    for block in range(3):
        retained = blocks != block
        np.testing.assert_allclose(
            observed.directional_ldscores[retained],
            expected.directional_ldscores[retained],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
    np.testing.assert_allclose(
        observed.same_person,
        exact_same_person_matrix(genotype, basis, fixed, annotations),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

def test_signed_per_snp_scores_are_not_clamped() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture(
        3, 2, seed=5802
    )
    observed, _pass1, _operator, _plan_value = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=5,
        rhs_probe_width=3,
    )
    assert float(np.min(observed.directional_ldscores)) < 0.0


def test_row_complete_sink_round_trips_without_driving_aggregates() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture(
        3, 2, seed=5902
    )
    captured = []

    def sink(row_start: int, row_stop: int, values: np.ndarray) -> None:
        assert values.flags.writeable is False
        assert values.shape[0] == row_stop - row_start
        captured.append((row_start, row_stop, np.array(values, copy=True)))

    observed, _pass1, _operator, _plan_value = _run_two_pass(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=4,
        rhs_probe_width=3,
        row_complete_sink=sink,
    )
    assert [(start, stop) for start, stop, _value in captured] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    roundtrip = np.concatenate([value for _start, _stop, value in captured])
    np.testing.assert_array_equal(roundtrip, observed.directional_ldscores)


def test_pass2_rejects_structural_annotation_mismatches() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture(
        3, 2, seed=6002
    )
    plan = _plan(
        genotype,
        basis,
        annotations,
        spec.probe_count,
        variant_width=4,
        rhs_probe_width=3,
        rhs_policy="tiled",
    )
    operator = ArraySequentialGenotypeOperator(genotype)
    names = ("annotation_0", "annotation_1")
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        annotation_masses=np.sum(annotations, axis=0),
        probe_spec=spec,
        work_plan=plan,
        nn_operator=NumpyNNOperator(),
        native_probe_module=False,
    ).execute()
    with pytest.raises(ValueError, match="basis must have shape"):
        GeneralizedGxEPass2Executor(
            pass1_result=pass1,
            genotype_operator=operator,
            basis=basis[:-1],
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=names,
            work_plan=plan,
            tn_operator=NumpyTNOperator(),
        )
    changed_annotations = annotations.copy()
    changed_annotations[0, 0] += 0.1
    with pytest.raises(RuntimeError, match="annotation masses changed"):
        GeneralizedGxEPass2Executor(
            pass1_result=pass1,
            genotype_operator=operator,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=changed_annotations,
            annotation_names=names,
            work_plan=plan,
            tn_operator=NumpyTNOperator(),
        )
    assert operator.observed_passes == 1


def test_planner_never_admits_rhs_narrower_than_q_squared() -> None:
    genotype, basis, _fixed, annotations, spec, _probes = _fixture(
        3, 1, seed=6101
    )
    plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=3,
            num_annotations=1,
            num_probes=spec.probe_count,
            memory_limit_bytes=512 * 1024**2,
            genotype_format="bed",
            preferred_rhs_tile_columns=1,
            rhs_policy="tiled",
        )
    )
    assert plan.tiling["rhs_tile_columns"] == 9


def _configured_native_threads(native_module) -> int:
    desired = min(2, len(os.sched_getaffinity(0)))
    try:
        return int(native_module.configure_blas_threads(desired))
    except RuntimeError as exc:
        assert (
            "different thread count" in str(exc)
            or "BLIS_NUM_THREADS" in str(exc)
        )
        configured = int(native_module.build_info()["blas_runtime_threads"])
        assert native_module.configure_blas_threads(configured) == configured
        return configured


def test_protected_native_tn_matches_dense_backend_and_uses_two_passes() -> None:
    from summit import gxeldcore

    genotype, basis, fixed, annotations, spec, probes = _fixture(
        3, 2, seed=6202
    )
    threads = _configured_native_threads(gxeldcore)
    plan = _plan(
        genotype,
        basis,
        annotations,
        spec.probe_count,
        variant_width=4,
        rhs_probe_width=4,
        rhs_policy="tiled",
        threads=threads,
    )
    operator = ArraySequentialGenotypeOperator(genotype)
    names = ("annotation_0", "annotation_1")
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        annotation_masses=np.sum(annotations, axis=0),
        probe_spec=spec,
        work_plan=plan,
        nn_operator=ProtectedNNOperator(threads=threads, native_module=gxeldcore),
        annotation_tile_width=2,
        probe_tile_width=5,
        native_probe_module=gxeldcore,
    ).execute()
    observed = GeneralizedGxEPass2Executor(
        pass1_result=pass1,
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        work_plan=plan,
        tn_operator=ProtectedTNOperator(threads=threads, native_module=gxeldcore),
        probe_tile_width=4,
    ).execute()
    expected, _sources, _cross = randomized_two_pass_ldscores(
        genotype, basis, fixed, annotations, probes
    )
    np.testing.assert_allclose(
        observed.directional_ldscores,
        expected.directional_ldscores,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        observed.genetic_gram,
        expected.gram,
        rtol=1.5e-13,
        atol=1.5e-13,
    )
    observed.ledger.validate_clean_completion()
    assert observed.telemetry["native"]["available"] is True
    assert observed.telemetry["native"]["gemm_status"]["dropped_records"] == 0
    assert observed.telemetry["native"]["output_numa_status"]["failed_calls"] == 0
    expected_calls = 3 * annotations.shape[1] * 6
    assert observed.telemetry["target_tn"]["calls"] == expected_calls


def test_native_pass2_one_and_multiple_threads_match_in_fresh_processes(
    tmp_path: Path,
) -> None:
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("native multi-thread comparison requires two available CPUs")
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from summit import gxeldcore
        from summit.ldscore.generalized_gxe_pass1 import (
            ArraySequentialGenotypeOperator, GeneralizedGxEPass1Executor,
            ProtectedNNOperator,
        )
        from summit.ldscore.generalized_gxe_pass2 import (
            GeneralizedGxEPass2Executor, ProtectedTNOperator,
        )
        from summit.ldscore.generalized_gxe_variant import (
            GeneralizedGxEPlanInputs, GlobalVariantProbeSpec,
            plan_generalized_gxe_variant_work,
        )

        threads = int(sys.argv[1])
        output = sys.argv[2]
        rng = np.random.default_rng(77125)
        n, m, q, k, b = 31, 29, 2, 2, 17
        genotype = np.asfortranarray(rng.normal(size=(n, m)))
        environment = rng.normal(size=n)
        basis = np.asfortranarray(np.column_stack([np.ones(n), environment]))
        fixed, _ = np.linalg.qr(
            np.column_stack([np.ones(n), rng.normal(size=n)]), mode="reduced"
        )
        fixed = np.asfortranarray(fixed)
        annotations = rng.uniform(0.2, 1.1, size=(m, k))
        spec = GlobalVariantProbeSpec(77125, 3, b)
        plan = plan_generalized_gxe_variant_work(GeneralizedGxEPlanInputs(
            num_samples=n, num_variants=m, num_basis=q, num_annotations=k,
            num_probes=b,
            memory_limit_bytes=512 * 1024**2, genotype_format="bed",
            threads=threads, preferred_variant_block_width=8,
            preferred_rhs_tile_columns=q*q*5, rhs_policy="tiled",
        ))
        operator = ArraySequentialGenotypeOperator(genotype)
        names = ("a", "b")
        pass1 = GeneralizedGxEPass1Executor(
            genotype_operator=operator, basis=basis, fixed_effect_basis=fixed,
            annotations=annotations, annotation_names=names,
            annotation_masses=np.sum(annotations, axis=0), probe_spec=spec,
            work_plan=plan,
            nn_operator=ProtectedNNOperator(threads=threads, native_module=gxeldcore),
            probe_tile_width=5, native_probe_module=gxeldcore,
        ).execute()
        result = GeneralizedGxEPass2Executor(
            pass1_result=pass1, genotype_operator=operator, basis=basis,
            fixed_effect_basis=fixed, annotations=annotations,
            annotation_names=names, work_plan=plan,
            tn_operator=ProtectedTNOperator(threads=threads, native_module=gxeldcore),
            probe_tile_width=5,
        ).execute()
        np.savez(
            output, directional=result.directional_ldscores,
            directed=result.directed_numerator,
        )
        """
    )
    outputs = []
    for threads in (1, 2):
        output = tmp_path / f"pass2-threads-{threads}.npz"
        environment = dict(os.environ)
        environment["BLIS_NUM_THREADS"] = str(threads)
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["OMP_THREAD_LIMIT"] = str(threads)
        completed = subprocess.run(
            [sys.executable, "-c", script, str(threads), str(output)],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(np.load(output))
    for name in ("directional", "directed"):
        np.testing.assert_allclose(
            outputs[0][name], outputs[1][name], rtol=8.0e-14, atol=8.0e-14
        )
