from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from generalized_gxe_variant_ldscore_oracle import (
    orthonormalize,
    pair_order,
    pass1_sources,
    same_person_ustatistic,
)
from summit.ldscore.generalized_gxe_pass1 import (
    ArraySequentialGenotypeOperator,
    DecodedGenotypeBlock,
    GeneralizedGxEPass1Executor,
    MatureSequentialGenotypeOperator,
    NumpyNNOperator,
    ProtectedNNOperator,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


def _fixture(*, probe_count: int = 37) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(2026082204)
    n_samples, n_variants = 11, 10
    environment = rng.normal(size=n_samples)
    covariate = rng.normal(size=n_samples)
    fixed = orthonormalize(
        np.column_stack([np.ones(n_samples), covariate])
    )
    genotype = np.asfortranarray(
        rng.normal(size=(n_samples, n_variants))
        + 0.35 * environment[:, None]
        + 0.20 * covariate[:, None]
    )
    basis = np.asfortranarray(
        np.column_stack(
            [
                np.ones(n_samples),
                environment,
                0.4 * environment**2 + rng.normal(size=n_samples),
            ]
        )
    )
    annotations = rng.uniform(0.1, 1.3, size=(n_variants, 2))
    spec = GlobalVariantProbeSpec(
        root_seed=2026082204,
        probe_offset=13,
        probe_count=probe_count,
    )
    probes = spec.generate(np.arange(n_variants, dtype=np.int64))
    return genotype, basis, fixed, annotations, spec, probes


def _plan(
    genotype: np.ndarray,
    basis: np.ndarray,
    annotations: np.ndarray,
    probe_count: int,
    *,
    variant_width: int,
    rhs_width: int,
) -> object:
    return plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=basis.shape[1],
            num_annotations=annotations.shape[1],
            num_probes=probe_count,
            memory_limit_bytes=512 * 1024**2,
            genotype_format="bed",
            threads=1,
            preferred_variant_block_width=variant_width,
            preferred_rhs_tile_columns=rhs_width,
            rhs_policy="tiled",
        )
    )


def _execute_numpy(
    *,
    genotype: np.ndarray,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    spec: GlobalVariantProbeSpec,
    variant_width: int,
    probe_width: int,
    annotation_width: int,
    threads: int = 1,
    retain_base: bool = True,
):
    operator = ArraySequentialGenotypeOperator(genotype)
    plan = _plan(
        genotype,
        basis,
        annotations,
        spec.probe_count,
        variant_width=variant_width,
        rhs_width=probe_width,
    )
    result = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=("baseline", "secondary")[: annotations.shape[1]],
        annotation_masses=np.sum(annotations, axis=0),
        probe_spec=spec,
        work_plan=plan,
        nn_operator=NumpyNNOperator(threads=threads),
        annotation_tile_width=annotation_width,
        probe_tile_width=probe_width,
        same_person_sample_tile_width=4,
        retain_base_sources=retain_base,
        native_probe_module=False,
    ).execute()
    return result, operator


def test_pass1_fixed_global_probes_match_dense_oracle_and_seal_barrier() -> None:
    genotype, basis, fixed, annotations, spec, probes = _fixture()
    expected_base, expected_contextual = pass1_sources(
        genotype, basis, fixed, annotations, probes
    )
    expected_same = same_person_ustatistic(
        expected_contextual, annotations, pair_order(basis.shape[1])
    )
    observed, operator = _execute_numpy(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=4,
        probe_width=7,
        annotation_width=2,
    )

    assert observed.base_sources is not None
    np.testing.assert_allclose(
        observed.base_sources, expected_base, rtol=3.0e-14, atol=3.0e-14
    )
    np.testing.assert_allclose(
        observed.contextual_sources,
        expected_contextual,
        rtol=4.0e-14,
        atol=4.0e-14,
    )
    np.testing.assert_allclose(
        observed.same_person, expected_same, rtol=8.0e-14, atol=8.0e-14
    )
    expected_leakage = max(
        float(np.max(np.abs(fixed.T @ expected_contextual[k, q])))
        for k in range(annotations.shape[1])
        for q in range(basis.shape[1])
    )
    assert observed.maximum_projection_leakage == pytest.approx(
        expected_leakage, rel=5.0e-2, abs=2.0e-14
    )
    assert observed.maximum_relative_projection_leakage < 2.0e-14
    assert observed.pass1_barrier_sealed is True
    observed.ledger.validate_pass1_barrier()
    assert observed.ledger.to_dict() == {
        "planned_reference_genotype_passes": 2,
        "observed_reference_genotype_passes": 1,
        "planned_retained_variant_visits": 20,
        "observed_retained_variant_visits": 10,
        "duplicate_retained_variant_visits": 0,
        "pass1_decoded_blocks": 3,
        "pass2_decoded_blocks": 0,
        "retry_count": 0,
        "repair_count": 0,
        "fallback_count": 0,
        "integrity_failure_count": 0,
    }
    assert operator.observed_passes == 1
    assert operator.observed_variant_visits == genotype.shape[1]
    assert operator.blocks_read == 3
    assert observed.pair_table == pair_order(3)
    assert observed.component_table == tuple(
        (annotation, pair)
        for annotation in range(2)
        for pair in range(6)
    )


@pytest.mark.parametrize(
    ("variant_width", "probe_width", "annotation_width"),
    ((1, 1, 1), (4, 7, 2), (10, 37, 1)),
)
def test_pass1_is_invariant_to_all_logical_tile_widths(
    variant_width: int,
    probe_width: int,
    annotation_width: int,
) -> None:
    genotype, basis, fixed, annotations, spec, probes = _fixture()
    expected_base, expected_contextual = pass1_sources(
        genotype, basis, fixed, annotations, probes
    )
    expected_same = same_person_ustatistic(
        expected_contextual, annotations, pair_order(3)
    )
    observed, operator = _execute_numpy(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=variant_width,
        probe_width=probe_width,
        annotation_width=annotation_width,
    )
    np.testing.assert_allclose(
        observed.base_sources, expected_base, rtol=4.0e-14, atol=4.0e-14
    )
    np.testing.assert_allclose(
        observed.contextual_sources,
        expected_contextual,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        observed.same_person, expected_same, rtol=1.0e-13, atol=1.0e-13
    )
    assert operator.observed_variant_visits == genotype.shape[1]
    assert observed.ledger.observed_retained_variant_visits == genotype.shape[1]
    assert observed.ledger.duplicate_retained_variant_visits == 0


def test_pass1_numpy_thread_request_is_scientifically_invariant() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture()
    results = []
    for threads in (1, 4):
        result, _operator = _execute_numpy(
            genotype=genotype,
            basis=basis,
            fixed=fixed,
            annotations=annotations,
            spec=spec,
            variant_width=4,
            probe_width=7,
            annotation_width=2,
            threads=threads,
        )
        results.append(result)
    np.testing.assert_array_equal(
        results[0].contextual_sources, results[1].contextual_sources
    )
    np.testing.assert_array_equal(results[0].same_person, results[1].same_person)


def test_pass1_releases_base_scratch_and_seals_scientific_outputs() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture()
    result, _operator = _execute_numpy(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        spec=spec,
        variant_width=10,
        probe_width=37,
        annotation_width=2,
        retain_base=False,
    )
    assert result.base_sources is None
    assert result.contextual_sources.flags.writeable is False
    assert result.same_person.flags.writeable is False
    assert result.annotation_masses.flags.writeable is False
    assert result.telemetry["allocation_ledger"][
        "base_source_scratch_released"
    ] is True
    expected_barrier = {
        "pass1_sealed": True,
        "target_scoring_started": False,
        "contextual_sources_readonly": True,
        "same_person_cross_tile_finalized": True,
    }
    assert all(
        result.telemetry["barrier"][name] == value
        for name, value in expected_barrier.items()
    )
    with pytest.raises(ValueError):
        result.contextual_sources[0, 0, 0, 0] = 0.0


@pytest.mark.parametrize("mutation", ("negative", "nonfinite", "zero_mass", "wrong_mass"))
def test_pass1_rejects_invalid_annotations_before_descriptor_entry(
    mutation: str,
) -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture()
    changed = annotations.copy()
    masses = np.sum(changed, axis=0)
    if mutation == "negative":
        changed[0, 0] = -1.0
    elif mutation == "nonfinite":
        changed[0, 0] = np.nan
    elif mutation == "zero_mass":
        changed[:, 0] = 0.0
        masses = np.sum(changed, axis=0)
    else:
        masses[0] += 1.0
    operator = ArraySequentialGenotypeOperator(genotype)
    with pytest.raises(ValueError, match="annotation"):
        GeneralizedGxEPass1Executor(
            genotype_operator=operator,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=changed,
            annotation_names=("baseline", "secondary"),
            annotation_masses=masses,
            probe_spec=spec,
            work_plan=_plan(
                genotype,
                basis,
                changed,
                spec.probe_count,
                variant_width=4,
                rhs_width=7,
            ),
            nn_operator=NumpyNNOperator(),
            native_probe_module=False,
        )
    assert operator.observed_passes == 0
    assert operator.observed_variant_visits == 0


def test_pass1_fails_closed_on_common_scale_identity_mismatch() -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture()

    class WrongScaleOperator(ArraySequentialGenotypeOperator):
        def read_block(self, row_start: int, row_stop: int) -> DecodedGenotypeBlock:
            block = super().read_block(row_start, row_stop)
            return DecodedGenotypeBlock(
                block.row_start, block.row_stop, block.values, "different_scale"
            )

    operator = WrongScaleOperator(genotype)
    executor = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=("baseline", "secondary"),
        annotation_masses=np.sum(annotations, axis=0),
        probe_spec=spec,
        work_plan=_plan(
            genotype,
            basis,
            annotations,
            spec.probe_count,
            variant_width=4,
            rhs_width=7,
        ),
        nn_operator=NumpyNNOperator(),
        native_probe_module=False,
    )
    with pytest.raises(RuntimeError, match="identity/scale"):
        executor.execute()


def test_mature_adapter_detects_in_place_descriptor_mutation(tmp_path: Path) -> None:
    genotype, basis, fixed, annotations, spec, _probes = _fixture()
    source = tmp_path / "sealed.bed"
    source.write_bytes(b"0123456789abcdef")
    descriptor = os.open(source, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))

    def mutating_reader(row_start: int, row_stop: int) -> np.ndarray:
        os.pwrite(descriptor, b"X", 0)
        os.fsync(descriptor)
        return genotype[:, row_start:row_stop]

    try:
        operator = MatureSequentialGenotypeOperator(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            genotype_format="bed",
            genotype_scale_id="mean_imputed_sample_ddof=1_fp64_v1",
            stable_descriptors={".bed": descriptor},
            read_block=mutating_reader,
            backend_name="mutation_test_callback",
        )
        executor = GeneralizedGxEPass1Executor(
            genotype_operator=operator,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=("baseline", "secondary"),
            annotation_masses=np.sum(annotations, axis=0),
            probe_spec=spec,
            work_plan=_plan(
                genotype,
                basis,
                annotations,
                spec.probe_count,
                variant_width=4,
                rhs_width=7,
            ),
            nn_operator=NumpyNNOperator(),
            native_probe_module=False,
        )
        with pytest.raises(RuntimeError, match="descriptors changed"):
            executor.execute()
        assert operator.observed_variant_visits < genotype.shape[1]
        assert operator.observed_passes == 1
    finally:
        os.close(descriptor)


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


def test_native_protected_nn_pass1_matches_dense_oracle_and_publishes_telemetry() -> None:
    from summit import gxeldcore

    genotype, basis, fixed, annotations, spec, probes = _fixture()
    expected_base, expected_contextual = pass1_sources(
        genotype, basis, fixed, annotations, probes
    )
    expected_same = same_person_ustatistic(
        expected_contextual, annotations, pair_order(3)
    )
    threads = _configured_native_threads(gxeldcore)
    operator = ArraySequentialGenotypeOperator(genotype)
    result = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=("baseline", "secondary"),
        annotation_masses=np.sum(annotations, axis=0),
        probe_spec=spec,
        work_plan=_plan(
            genotype,
            basis,
            annotations,
            spec.probe_count,
            variant_width=4,
            rhs_width=7,
        ),
        nn_operator=ProtectedNNOperator(threads=threads, native_module=gxeldcore),
        annotation_tile_width=2,
        probe_tile_width=7,
        retain_base_sources=True,
        native_probe_module=gxeldcore,
    ).execute()
    np.testing.assert_allclose(
        result.base_sources, expected_base, rtol=4.0e-14, atol=4.0e-14
    )
    np.testing.assert_allclose(
        result.contextual_sources,
        expected_contextual,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        result.same_person, expected_same, rtol=1.0e-13, atol=1.0e-13
    )
    assert result.telemetry["native"]["available"] is True
    assert result.telemetry["source_nn"]["calls"] == 36
    assert result.telemetry["source_nn"]["repaired_columns"] == 0
    assert result.telemetry["native"]["gemm_status"]["dropped_records"] == 0
    assert result.telemetry["native"]["output_numa_status"]["failed_calls"] == 0


def test_native_pass1_one_and_multiple_threads_match_in_fresh_processes(
    tmp_path: Path,
) -> None:
    from summit import gxeldcore

    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("native multi-thread comparison requires two available CPUs")
    script = textwrap.dedent(
        """
        import os
        import sys
        import numpy as np
        from summit import gxeldcore
        from summit.ldscore.generalized_gxe_pass1 import (
            ArraySequentialGenotypeOperator, GeneralizedGxEPass1Executor,
            ProtectedNNOperator,
        )
        from summit.ldscore.generalized_gxe_variant import (
            GeneralizedGxEPlanInputs, GlobalVariantProbeSpec,
            plan_generalized_gxe_variant_work,
        )

        threads = int(sys.argv[1])
        output = sys.argv[2]
        rng = np.random.default_rng(77123)
        n, m, q, k, b = 31, 29, 2, 2, 17
        genotype = np.asfortranarray(rng.normal(size=(n, m)))
        environment = rng.normal(size=n)
        basis = np.asfortranarray(np.column_stack([np.ones(n), environment]))
        fixed, _ = np.linalg.qr(
            np.column_stack([np.ones(n), rng.normal(size=n)]), mode="reduced"
        )
        fixed = np.asfortranarray(fixed)
        annotations = rng.uniform(0.2, 1.1, size=(m, k))
        spec = GlobalVariantProbeSpec(77123, 3, b)
        plan = plan_generalized_gxe_variant_work(GeneralizedGxEPlanInputs(
            num_samples=n, num_variants=m, num_basis=q, num_annotations=k,
            num_probes=b,
            memory_limit_bytes=512 * 1024**2, genotype_format="bed",
            threads=threads, preferred_variant_block_width=8,
            preferred_rhs_tile_columns=5, rhs_policy="tiled",
        ))
        result = GeneralizedGxEPass1Executor(
            genotype_operator=ArraySequentialGenotypeOperator(genotype),
            basis=basis, fixed_effect_basis=fixed, annotations=annotations,
            annotation_names=("a", "b"),
            annotation_masses=np.sum(annotations, axis=0), probe_spec=spec,
            work_plan=plan,
            nn_operator=ProtectedNNOperator(threads=threads, native_module=gxeldcore),
            annotation_tile_width=2, probe_tile_width=5,
            retain_base_sources=True, native_probe_module=gxeldcore,
        ).execute()
        np.savez(
            output, base=result.base_sources,
            contextual=result.contextual_sources, same=result.same_person,
        )
        """
    )
    outputs = []
    for threads in (1, 2):
        output = tmp_path / f"threads-{threads}.npz"
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
    for name in ("base", "contextual", "same"):
        np.testing.assert_allclose(
            outputs[0][name], outputs[1][name], rtol=5.0e-14, atol=5.0e-14
        )
    assert dict(gxeldcore.build_info())["global_variant_probe_supported"] is True


def test_mature_integrity_diagnostic_repairs_corrupted_pass1_nn_output() -> None:
    from summit import gxeldcore

    info = dict(gxeldcore.build_info())
    diagnostic = getattr(
        gxeldcore, "_test_protected_matmul_nn_integrity_diagnostic", None
    )
    if (
        not info["gemm_integrity_enabled"]
        or not info["gemm_checksum_enabled"]
        or not callable(diagnostic)
    ):
        pytest.skip("native integrity diagnostic is unavailable in this build")
    threads = _configured_native_threads(gxeldcore)
    rng = np.random.default_rng(404404)
    n, m, b = 512, 2048, 512
    genotype = np.asfortranarray(rng.normal(size=(n, m)))
    basis = np.ones((n, 1), dtype=np.float64, order="F")
    fixed = orthonormalize(np.ones((n, 1), dtype=np.float64))
    annotations = np.ones((m, 1), dtype=np.float64)
    spec = GlobalVariantProbeSpec(404404, 0, b)
    plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=n,
            num_variants=m,
            num_basis=1,
            num_annotations=1,
            num_probes=b,
            memory_limit_bytes=1024**3,
            genotype_format="bed",
            threads=threads,
            preferred_variant_block_width=m,
            preferred_rhs_tile_columns=b,
            rhs_policy="tiled",
        )
    )

    class FaultDiagnosticNN(ProtectedNNOperator):
        def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            output, repaired, raw = diagnostic(
                left,
                right,
                self.threads,
                fault_injection_row=17,
                fault_injection_column=29,
                fault_injection_delta=1.0,
            )
            evidence = dict(raw)
            assert evidence["classification"] == (
                "vendor_result_outside_forward_error_bound"
            )
            assert int(repaired) == 1
            self.calls += 1
            self.repaired_columns += int(repaired)
            return np.asarray(output)

    result = GeneralizedGxEPass1Executor(
        genotype_operator=ArraySequentialGenotypeOperator(genotype),
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=("baseline",),
        annotation_masses=np.asarray([float(m)]),
        probe_spec=spec,
        work_plan=plan,
        nn_operator=FaultDiagnosticNN(threads=threads, native_module=gxeldcore),
        probe_tile_width=b,
        retain_base_sources=True,
        native_probe_module=gxeldcore,
    ).execute()
    probes = spec.generate(np.arange(m, dtype=np.int64))
    expected = genotype @ probes
    np.testing.assert_allclose(
        result.base_sources[0], expected, rtol=4.0e-14, atol=4.0e-12
    )
    assert result.ledger.repair_count == 1
    assert result.telemetry["source_nn"]["repaired_columns"] == 1
