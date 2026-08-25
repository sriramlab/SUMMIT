from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from bed_reader import open_bed, to_bed

from generalized_gxe_variant_ldscore_oracle import orthonormalize
from summit.ldscore.generalized_gxe_native import (
    GeneralizedGxENativeBEDExecutor,
    generalized_gxe_performance_ledger_from_native,
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
    build_generalized_gxe_variant_reference_from_native_v1,
    serialize_generalized_gxe_inference_axes,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


def _fixture(
    tmp_path: Path,
    *,
    q_count: int,
    annotation_count: int,
    seed: int,
) -> tuple[Path, np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n, m = 13, 12
    probabilities = rng.uniform(0.15, 0.45, size=m)
    raw = rng.binomial(2, probabilities, size=(n, m)).astype(np.float64)
    raw[0] = 0.0
    raw[1] = 1.0
    raw[2] = 2.0
    raw[4, 3] = np.nan
    raw[9, 8] = np.nan
    prefix = tmp_path / f"native_q{q_count}_k{annotation_count}"
    to_bed(str(prefix) + ".bed", raw)

    means = np.nanmean(raw, axis=0)
    genotype = np.where(np.isnan(raw), means, raw) - means
    sums_of_squares = np.sum(genotype * genotype, axis=0, dtype=np.float64)
    genotype *= np.sqrt((n - 1) / sums_of_squares)
    genotype = np.asfortranarray(genotype)

    environment = rng.normal(size=n)
    basis_columns = [np.ones(n), environment]
    if q_count == 3:
        basis_columns.append(0.35 * environment**2 + rng.normal(size=n))
    basis = np.asfortranarray(np.column_stack(basis_columns[:q_count]))
    fixed = orthonormalize(
        np.column_stack([np.ones(n), rng.normal(size=n)])
    )
    annotations = rng.uniform(0.1, 1.4, size=(m, annotation_count))
    # Unequal contiguous segments ensure genotype blocks cross jackknife
    # boundaries for the width-four and width-five plans below.
    block_ids = np.repeat(
        np.arange(3, dtype=np.int64), np.asarray([3, 4, 5])
    )
    probe_spec = GlobalVariantProbeSpec(
        root_seed=seed, probe_offset=7, probe_count=13
    )
    return (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    )


def _plan(
    genotype: np.ndarray,
    basis: np.ndarray,
    annotations: np.ndarray,
    *,
    variant_width: int,
    probe_width: int,
    threads: int,
):
    return plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=genotype.shape[0],
            num_variants=genotype.shape[1],
            num_basis=basis.shape[1],
            num_annotations=annotations.shape[1],
            num_probes=13,
            memory_limit_bytes=512 * 1024**2,
            genotype_format="bed",
            threads=threads,
            preferred_variant_block_width=variant_width,
            preferred_rhs_tile_columns=basis.shape[1] ** 2 * probe_width,
            rhs_policy="tiled",
        )
    )


def _reference(
    *,
    genotype: np.ndarray,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    probe_spec: GlobalVariantProbeSpec,
    plan,
    probe_width: int,
):
    operator = ArraySequentialGenotypeOperator(genotype)
    names = tuple(f"annotation_{index}" for index in range(annotations.shape[1]))
    pass1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator,
        basis=basis,
        fixed_effect_basis=fixed,
        annotations=annotations,
        annotation_names=names,
        annotation_masses=np.sum(annotations, axis=0, dtype=np.float64),
        probe_spec=probe_spec,
        work_plan=plan,
        nn_operator=NumpyNNOperator(),
        annotation_tile_width=annotations.shape[1],
        probe_tile_width=min(5, probe_spec.probe_count),
        same_person_sample_tile_width=4,
        retain_base_sources=True,
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
        probe_tile_width=probe_width,
    ).execute()
    return pass1, result


def _native(
    *,
    prefix: Path,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    probe_spec: GlobalVariantProbeSpec,
    plan,
    probe_width: int,
    backend: str,
    threads: int,
    source_probe_width: int | None = None,
    retain_base: bool = True,
    qualification_fault_injection=None,
):
    from summit import gxeldcore

    configured = int(gxeldcore.configured_blas_threads())
    execution_threads = configured if configured > 0 else threads
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptors = {
        suffix: os.open(str(prefix) + suffix, flags)
        for suffix in (".bed", ".bim", ".fam")
    }
    try:
        executor = GeneralizedGxENativeBEDExecutor(
            stable_descriptors=descriptors,
            row_selection=None,
            ddof=1,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=tuple(
                f"annotation_{index}" for index in range(annotations.shape[1])
            ),
            annotation_masses=np.sum(
                annotations, axis=0, dtype=np.float64
            ),
            probe_spec=probe_spec,
            work_plan=plan,
            probe_tile_width=probe_width,
            source_probe_tile_width=source_probe_width,
            same_person_sample_tile_width=4,
            threads=execution_threads,
            retain_base_sources=retain_base,
            backend=backend,
            qualification_fault_injection=qualification_fault_injection,
        )
        return executor, executor.execute()
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


@pytest.mark.parametrize(
    ("q_count", "annotation_count", "seed"),
    ((1, 1, 6101), (2, 1, 6201), (3, 1, 6301), (3, 2, 6302)),
)
def test_native_dense_matches_every_stage05_layer(
    tmp_path: Path,
    q_count: int,
    annotation_count: int,
    seed: int,
) -> None:
    (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    ) = _fixture(
        tmp_path,
        q_count=q_count,
        annotation_count=annotation_count,
        seed=seed,
    )
    plan = _plan(
        genotype,
        basis,
        annotations,
        variant_width=4,
        probe_width=3,
        threads=1,
    )
    pass1, expected = _reference(
        genotype=genotype,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        probe_spec=probe_spec,
        plan=plan,
        probe_width=3,
    )
    executor, observed = _native(
        prefix=prefix,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        probe_spec=probe_spec,
        plan=plan,
        probe_width=3,
        backend="dense",
        threads=1,
    )
    comparisons = (
        (observed.base_sources, pass1.base_sources),
        (observed.contextual_sources, pass1.contextual_sources),
        (observed.same_person, pass1.same_person),
        (observed.directional_ldscores, expected.directional_ldscores),
        (observed.directed_numerator, expected.directed_numerator),
        (observed.symmetric_numerator, expected.symmetric_numerator),
        (observed.genetic_gram, expected.genetic_gram),
    )
    for actual, target in comparisons:
        np.testing.assert_allclose(actual, target, rtol=3.0e-13, atol=3.0e-13)
    assert observed.pair_table == expected.pair_table
    assert observed.component_table == expected.component_table
    assert dict(observed.ledger)["observed_reference_genotype_passes"] == 2
    assert dict(observed.ledger)["observed_retained_variant_visits"] == 24
    assert dict(observed.ledger)["duplicate_variant_visits"] == 0
    assert dict(executor.info())["state"] == "finalized"


@pytest.mark.parametrize("threads", (1,))
def test_packed_mailman_matches_dense_across_threads_and_tiles(
    tmp_path: Path, threads: int
) -> None:
    (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    ) = _fixture(tmp_path, q_count=3, annotation_count=2, seed=6400 + threads)
    plan = _plan(
        genotype,
        basis,
        annotations,
        variant_width=5,
        probe_width=4,
        threads=threads,
    )
    _, dense = _native(
        prefix=prefix,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        probe_spec=probe_spec,
        plan=plan,
        probe_width=4,
        backend="dense",
        threads=threads,
        source_probe_width=13,
    )
    _, packed = _native(
        prefix=prefix,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        probe_spec=probe_spec,
        plan=plan,
        probe_width=4,
        backend="packed",
        threads=threads,
        source_probe_width=5,
    )
    for name in (
        "base_sources",
        "contextual_sources",
        "same_person",
        "directional_ldscores",
        "directed_numerator",
        "genetic_gram",
    ):
        np.testing.assert_allclose(
            getattr(packed, name),
            getattr(dense, name),
            rtol=3.0e-13,
            atol=3.0e-13,
        )
    assert dict(packed.telemetry)["packed_backend_used"] is True
    assert dict(packed.telemetry)["source_nn_calls"] != dict(dense.telemetry)[
        "source_nn_calls"
    ]
    assert dict(packed.ledger)["observed_retained_variant_visits"] == 24
    assert dict(packed.genotype_scale) == dict(dense.genotype_scale)


def test_packed_native_one_and_multiple_threads_match_in_fresh_processes(
    tmp_path: Path,
) -> None:
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("native multi-thread comparison requires two available CPUs")
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import numpy as np
        from test_generalized_gxe_native import _fixture, _native, _plan

        threads = int(sys.argv[1])
        case = Path(sys.argv[2])
        output = sys.argv[3]
        (
            prefix, genotype, basis, fixed, annotations, block_ids, probe_spec,
        ) = _fixture(case, q_count=3, annotation_count=2, seed=6711)
        plan = _plan(
            genotype, basis, annotations,
            variant_width=5, probe_width=4, threads=threads,
        )
        _, result = _native(
            prefix=prefix, basis=basis, fixed=fixed,
            annotations=annotations,
            probe_spec=probe_spec, plan=plan, probe_width=4,
            backend="packed", threads=threads,
        )
        np.savez(
            output,
            source=result.contextual_sources,
            directional=result.directional_ldscores,
            directed=result.directed_numerator,
        )
        """
    )
    outputs = []
    for threads in (1, 2):
        case = tmp_path / f"threads-{threads}"
        case.mkdir()
        output = tmp_path / f"native-threads-{threads}.npz"
        environment = dict(os.environ)
        environment["BLIS_NUM_THREADS"] = str(threads)
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["OMP_THREAD_LIMIT"] = str(threads)
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(Path(__file__).resolve().parent),
                environment.get("PYTHONPATH", ""),
            ]
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(threads),
                str(case),
                str(output),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(np.load(output))
    for name in ("source", "directional", "directed"):
        np.testing.assert_allclose(
            outputs[0][name], outputs[1][name], rtol=3.0e-13, atol=3.0e-13
        )


def test_context_is_single_use_and_omits_unrequested_base_sources(
    tmp_path: Path,
) -> None:
    (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    ) = _fixture(tmp_path, q_count=2, annotation_count=1, seed=6501)
    plan = _plan(
        genotype,
        basis,
        annotations,
        variant_width=5,
        probe_width=4,
        threads=1,
    )
    executor, result = _native(
        prefix=prefix,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        probe_spec=probe_spec,
        plan=plan,
        probe_width=4,
        backend="dense",
        threads=1,
        retain_base=False,
    )
    assert result.base_sources is None
    with pytest.raises(RuntimeError, match="single-use"):
        executor.execute()


def test_descriptor_mutation_before_run_fails_without_publication(
    tmp_path: Path,
) -> None:
    from summit import gxeldcore

    (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    ) = _fixture(tmp_path, q_count=1, annotation_count=1, seed=6601)
    plan = _plan(
        genotype,
        basis,
        annotations,
        variant_width=4,
        probe_width=3,
        threads=1,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptors = {
        suffix: os.open(str(prefix) + suffix, flags)
        for suffix in (".bed", ".bim", ".fam")
    }
    try:
        configured = int(gxeldcore.configured_blas_threads())
        executor = GeneralizedGxENativeBEDExecutor(
            stable_descriptors=descriptors,
            row_selection=None,
            ddof=1,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=("annotation_0",),
            annotation_masses=np.sum(annotations, axis=0),
            probe_spec=probe_spec,
            work_plan=plan,
            probe_tile_width=3,
            same_person_sample_tile_width=4,
            threads=configured if configured > 0 else 1,
        )
        with open(str(prefix) + ".bim", "ab") as handle:
            handle.write(b"mutated\n")
        with pytest.raises(RuntimeError, match="changed"):
            executor.execute()
        assert dict(executor.info())["state"] == "failed"
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


@pytest.mark.parametrize("phase", ("pass1_nn", "pass2_tn"))
def test_algebraic_checksum_fault_fails_before_publication(
    tmp_path: Path,
    phase: str,
) -> None:
    from summit import gxeldcore

    (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    ) = _fixture(tmp_path, q_count=3, annotation_count=1, seed=6701)
    plan = _plan(
        genotype,
        basis,
        annotations,
        variant_width=4,
        probe_width=3,
        threads=1,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptors = {
        suffix: os.open(str(prefix) + suffix, flags)
        for suffix in (".bed", ".bim", ".fam")
    }
    try:
        configured = int(gxeldcore.configured_blas_threads())
        executor = GeneralizedGxENativeBEDExecutor(
            stable_descriptors=descriptors,
            row_selection=None,
            ddof=1,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=("annotation_0",),
            annotation_masses=np.sum(annotations, axis=0),
            probe_spec=probe_spec,
            work_plan=plan,
            probe_tile_width=3,
            same_person_sample_tile_width=4,
            threads=configured if configured > 0 else 1,
            qualification_fault_injection={
                "phase": phase,
                "row": 0,
                "column": 0,
                "delta": 1000.0,
            },
        )
        with pytest.raises(RuntimeError, match="checksum detected corrupted"):
            executor.execute()
        info = dict(executor.info())
        assert info["state"] == "failed"
        assert info["qualification_fault_injected"] is True
        assert info["integrity_failures"] == 1
        assert info["independent_integrity_audit_count"] == (
            0 if phase == "pass1_nn" else 1
        )
        assert info["protected_integrity_audit_count"] == 0
        with pytest.raises(RuntimeError, match="single-use"):
            executor.execute()
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_native_result_adapts_directly_to_compact_closed_artifact(
    tmp_path: Path,
) -> None:
    from summit import gxeldcore

    (
        prefix,
        genotype,
        basis,
        fixed,
        annotations,
        block_ids,
        probe_spec,
    ) = _fixture(tmp_path, q_count=1, annotation_count=1, seed=6801)
    plan = _plan(
        genotype,
        basis,
        annotations,
        variant_width=4,
        probe_width=3,
        threads=1,
    )
    _, result = _native(
        prefix=prefix,
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        probe_spec=probe_spec,
        plan=plan,
        probe_width=3,
        backend="dense",
        threads=1,
    )
    scale = result.genotype_scale
    assert scale["centering_source"] == "provided_v1"
    assert scale["centering_formula"] == "provided_variant_affine_mean_v1"
    assert (
        scale["scaling_formula"]
        == "dosage_minus_mean_times_inverse_scale_v1"
    )
    assert scale["missing_imputation"] == "sealed_mean_v1"
    raw = open_bed(str(prefix) + ".bed").read(dtype=np.float64)
    means = np.nanmean(raw, axis=0)
    compact = open_bed(str(prefix) + ".bed", count_A1=False).read(
        dtype=np.float64
    )
    inverse = np.empty(compact.shape[1], dtype=np.float64)
    for variant in range(compact.shape[1]):
        observed = compact[:, variant][~np.isnan(compact[:, variant])]
        total = int(np.sum(observed, dtype=np.float64))
        total_squares = int(np.sum(observed * observed, dtype=np.float64))
        m2 = total_squares - total * total / observed.size
        inverse[variant] = np.sqrt((compact.shape[0] - 1) / m2)
    np.testing.assert_array_equal(result.affine_mean, means)
    np.testing.assert_array_equal(result.affine_inverse_scale, inverse)
    assert result.affine_mean.flags.writeable is False
    assert result.affine_inverse_scale.flags.writeable is False
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=genotype.shape[1],
        num_samples=genotype.shape[0],
        basis_names=("intercept",),
        fixed_effect_rank=fixed.shape[1],
        annotation_names=("annotation_0",),
        annotation_masses=np.sum(annotations, axis=0),
        variant_block_ids=block_ids,
        block_labels=("block_0", "block_1", "block_2"),
        residual_component_names=("identity",),
    )
    telemetry = dict(result.telemetry)
    performance_ledger = generalized_gxe_performance_ledger_from_native(result)
    assert set(performance_ledger["phase_wall_seconds"]) == {
        "pass1", "barrier", "pass2", "finalize"
    }
    assert all(
        value > 0.0
        for value in performance_ledger["phase_wall_seconds"].values()
    )
    assert performance_ledger["bytes_read"] > 0
    assert performance_ledger["gemm_dimensions"]
    assert performance_ledger["peak_rss_bytes"] > 0
    assert performance_ledger["output_bytes"] > 0
    assert int(telemetry["integrity_audit_count"]) >= 2
    assert int(telemetry["integrity_audit_count"]) == (
        int(telemetry["protected_integrity_audit_count"])
        + int(telemetry["independent_integrity_audit_count"])
    )
    assert int(telemetry["independent_integrity_audit_count"]) >= 2
    assert int(telemetry["full_serial_output_witness_calls"]) == 0
    assert int(telemetry["tile_induced_descriptor_rereads"]) == 0
    assert int(telemetry["integrity_failure_count"]) == 0
    assert int(telemetry["checksum_recomputed_columns"]) == 0
    assert int(telemetry["roundoff_only_columns"]) == 0
    assert dict(telemetry["numa_evidence"])["gemm_output_count"] == (
        int(telemetry["source_nn_calls"])
        + int(telemetry["target_tn_calls"])
    )
    diagnostics = {
        "maximum_source_projection_leakage": float(
            telemetry["maximum_projection_leakage"]
        ),
        "maximum_presymmetry_absolute_error": float(
            telemetry["presymmetry_absolute_error"]
        ),
        "maximum_presymmetry_relative_error": float(
            telemetry["presymmetry_relative_error"]
        ),
        "same_person_probe_count": probe_spec.probe_count,
        "same_person_cross_tile_finalized": True,
        "minimum_annotation_mass": float(np.min(result.annotation_masses)),
        "all_values_finite": True,
        "normal_matrix_rank": len(result.component_table),
        "normal_matrix_condition": 1.0,
        "dense_oracle_fixture_version": "stage07_native_adapter_v1",
        "backend_fixed_probe_maximum_error": 3.0e-13,
    }
    artifact = build_generalized_gxe_variant_reference_from_native_v1(
        result,
        axes=axes,
        annotations=annotations,
        probe_spec=probe_spec,
        genotype_scale_plan=scale,
        performance_ledger=performance_ledger,
        provenance={"native_module": str(Path(gxeldcore.__file__))},
        diagnostics=diagnostics,
        include_directional_panel=False,
    )
    assert artifact.directional_ldscores is None
    assert artifact.deleted_genetic_gram is None
    assert artifact.manifest["pass_ledger"]["pass1_decoded_blocks"] == 3
    assert artifact.manifest["pass_ledger"]["pass2_decoded_blocks"] == 3
    assert artifact.manifest["pass_ledger"][
        "observed_reference_genotype_passes"
    ] == 2
    np.testing.assert_array_equal(artifact.genetic_gram, result.genetic_gram)
    np.testing.assert_array_equal(artifact.same_person, result.same_person)
