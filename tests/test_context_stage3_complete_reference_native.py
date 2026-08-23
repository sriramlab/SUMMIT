from __future__ import annotations

from pathlib import Path
import gc
import os
from dataclasses import replace

import numpy as np
import pytest

from summit.context import (
    ContextualReferencePublicationIdentityV1,
    assemble_context_normal_equations,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    coefficients_to_omegas,
    common_scale_features,
    contextual_variant_order_allele_sha256_v1,
    context_kernel_actions,
    derive_context_outputs,
    fit_context_model,
    load_contextual_reference_v1,
    rank_revealing_projector,
    run_contextual_reference_v1,
    solve_context_normal_equations,
    write_contextual_reference_v1,
)
from summit.context.reference_v1 import _NATIVE_RESULT_KEYS

from test_context_stage2_streamed_reference import (
    _assert_close,
    _components,
    _executor,
    _make_case,
    _philox_probes,
    _run_checkpoint_failure,
    _scale_plan,
)


def _variant_probes(case: object, count: int = 3) -> np.ndarray:
    rng = np.random.default_rng(731_000 + int(case.phi.shape[1]))
    return np.asfortranarray(
        2.0
        * rng.integers(
            0, 2, size=(case.retained_variant_rows.size, count), dtype=np.int8
        )
        - 1.0,
        dtype=np.float64,
    )


def _stable_scale_plan(case: object) -> object:
    return _scale_plan(
        retained_variant_order_sha256=case.retained_variant_order_sha256,
        counted_allele_mode=case.counted_allele_mode,
        centering_source="provided_v1",
        affine_mean_sha256=case.affine_mean_sha256,
        affine_inverse_scale_sha256=case.affine_inverse_scale_sha256,
    )


def _stage3_executor(case: object, probes: np.ndarray, **changes: object) -> object:
    stable_scale = _stable_scale_plan(case)
    values: dict[str, object] = {
        "variant_probes": probes,
        "variant_probe_count": probes.shape[1],
        "variant_probe_tile": 0,
        "group_tile": 0,
        "grouped_algorithm": "group_restricted_action_v1",
        "direct_grouped_scaling": "action_scaled_v1",
        "enable_grouped_differential": False,
        "centering_source": "provided_v1",
        "scale_plan_sha256": stable_scale.digest,
    }
    values.update(changes)
    return _executor(case, **values)


def _oracle(case: object, probes: np.ndarray) -> object:
    groups = tuple(case.group_names[int(index)] for index in case.group_index)
    return build_context_reference(
        genotype=case.scaled_genotype,
        basis=case.phi,
        projector=rank_revealing_projector(case.fixed_basis),
        annotations=case.annotations,
        component_index=_components(case),
        basis_hash=array_sha256(case.phi),
        fixed_effect_hash=array_sha256(case.fixed_basis),
        variant_hash=canonical_sha256(
            {"logical_variants": list(case.expected_variant_ids)}
        ),
        loo_groups=groups,
        genotype_scaling=_stable_scale_plan(case).digest,
        gram_method="hutchinson",
        gram_probes=case.sample_probes,
        same_person_method="ustat",
        variant_probes=probes,
        probe_tile_size=2,
        contribution_storage="loo_grouped",
    )


def _publication_identity(case: object) -> ContextualReferencePublicationIdentityV1:
    flags = np.full(
        case.retained_variant_rows.size,
        case.counted_allele_mode == "bim_a1_counted_v1",
        dtype=np.uint8,
    )
    probes = _variant_probes(case)
    stable_scale = _stable_scale_plan(case)
    return ContextualReferencePublicationIdentityV1(
        sample_order_sha256=canonical_sha256(
            {"retained_sample_rows": case.retained_sample_rows.tolist()}
        ),
        variant_order_allele_sha256=contextual_variant_order_allele_sha256_v1(
            case.retained_variant_rows,
            case.expected_variant_ids,
            case.expected_counted_alleles,
            case.expected_other_alleles,
            flags,
        ),
        fixed_effect_spec_sha256=canonical_sha256(
            {"fixed_effect_basis": array_sha256(case.fixed_basis)}
        ),
        basis_specification_sha256=canonical_sha256(
            {"context_basis": "explicit_test_matrix_v1"}
        ),
        basis_calibration_sha256=canonical_sha256(
            {"evaluated_phi": array_sha256(case.phi)}
        ),
        retained_sample_map_sha256=array_sha256(case.retained_sample_rows),
        retained_variant_order_sha256=case.retained_variant_order_sha256,
        fixed_basis_sha256=array_sha256(case.fixed_basis),
        evaluated_phi_sha256=array_sha256(case.phi),
        genotype_scale_plan_sha256=stable_scale.digest,
        missingness_sha256=case.missingness_sha256,
        annotation_map_sha256=array_sha256(case.annotations),
        annotation_names=case.annotation_names,
        group_map_sha256=array_sha256(case.group_index),
        group_ids=case.group_names,
        sample_probe_policy="explicit_rademacher_v1",
        sample_probe_identity_sha256=array_sha256(case.sample_probes),
        variant_probe_policy="explicit_variant_rademacher_v1",
        variant_probe_identity_sha256=array_sha256(probes),
    )


@pytest.mark.parametrize("q_count", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "annotation_mode",
    ["generic_nonnegative_weights_v1", "strict_disjoint_binary_v1"],
)
def test_native_stage3_q1_to_q4_matches_python_and_publishes_roundtrip(
    tmp_path: Path,
    q_count: int,
    annotation_mode: str,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=q_count,
        annotation_mode=annotation_mode,
        name=f"stage3-q{q_count}-{annotation_mode}",
    )
    probes = _variant_probes(case)
    expected = _oracle(case, probes)
    artifact = run_contextual_reference_v1(
        _stage3_executor(case, probes), _publication_identity(case)
    )

    _assert_close(artifact.gram, expected.gram, "full Gram")
    _assert_close(artifact.same_person, expected.same_person, "same-person D_R")
    _assert_close(
        artifact.group_gram_unnormalized_num,
        expected.gram_numerator_contributions,
        "grouped unnormalized numerators",
    )
    _assert_close(
        artifact.group_annotation_masses,
        expected.group_annotation_masses,
        "group annotation masses",
    )
    np.testing.assert_array_equal(
        artifact.group_variant_counts, expected.group_variant_counts
    )
    component_annotation = np.asarray(
        [entry.annotation_index for entry in artifact.component_index.entries],
        dtype=np.int64,
    )
    component_masses = artifact.annotation_masses[component_annotation]
    _assert_close(
        np.sum(artifact.group_gram_unnormalized_num, axis=0, dtype=np.float64),
        artifact.gram * np.outer(component_masses, component_masses),
        "exact raw-Gram reconstruction",
    )
    ledger = artifact.manifest["execution"]["admission"]["semantic_call_ledger"]
    assert ledger["direct_grouped_tn"] == 0
    assert all(
        ledger[name] == 0
        for name in (
            "trait_score_tn",
            "trait_feature_projection_tn",
            "trait_feature_projection_nn",
        )
    )
    diagnostics = artifact.manifest["execution"]["diagnostics"]
    if annotation_mode == "strict_disjoint_binary_v1":
        assert diagnostics["strict_disjoint_optimized_path"] is True
        assert diagnostics["strict_zero_weight_columns_eliminated"] > 0
        assert (
            diagnostics["target_useful_columns"] < diagnostics["target_dense_columns"]
        )
    else:
        assert diagnostics["strict_disjoint_optimized_path"] is False
        assert diagnostics["strict_zero_weight_columns_eliminated"] == 0
    assert artifact.manifest["identity"]["variant_order_allele_sha256"] == (
        _publication_identity(case).variant_order_allele_sha256
    )
    assert artifact.manifest["identity"]["sealed_plan_sha256"]
    assert artifact.manifest["identity"]["source_tree_sha256"]

    if q_count == 4 and annotation_mode == "strict_disjoint_binary_v1":
        path = write_contextual_reference_v1(
            artifact, tmp_path / "native.contextual-reference-v1.npz"
        )
        loaded = load_contextual_reference_v1(path)
        _assert_close(loaded.gram, expected.gram, "roundtrip Gram")
        _assert_close(loaded.same_person, expected.same_person, "roundtrip D_R")
        _assert_close(
            loaded.group_gram_unnormalized_num,
            expected.gram_numerator_contributions,
            "roundtrip grouped numerators",
        )


def _science_arrays(result: dict[str, object]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(result[name])
        for name in (
            "gram",
            "same_person",
            "group_gram_unnormalized_num",
            "annotation_masses",
            "group_annotation_masses",
        )
    )


@pytest.mark.parametrize(
    "tile_name",
    [
        "variant_block",
        "sample_probe_resident",
        "sample_probe_tile",
        "action_tile",
        "annotation_tile",
        "context_tile",
        "variant_probe_tile",
        "group_tile",
    ],
)
@pytest.mark.parametrize(
    "annotation_mode",
    ["generic_nonnegative_weights_v1", "strict_disjoint_binary_v1"],
)
def test_native_stage3_each_tile_is_independently_invariant(
    tmp_path: Path, tile_name: str, annotation_mode: str
) -> None:
    case = _make_case(
        tmp_path,
        q_count=3,
        annotation_mode=annotation_mode,
        name=f"stage3-tile-{annotation_mode}-{tile_name}",
    )
    probes = _variant_probes(case, 4)
    baseline = dict(_stage3_executor(case, probes).run())
    extents = {
        "variant_block": case.retained_variant_rows.size,
        "sample_probe_resident": case.sample_probes.shape[1],
        "sample_probe_tile": case.sample_probes.shape[1],
        "action_tile": len(_components(case)),
        "annotation_tile": len(case.annotation_names),
        "context_tile": case.phi.shape[1],
        "variant_probe_tile": probes.shape[1],
        "group_tile": len(case.group_names),
    }
    assert baseline["admission"]["selected_tiles"][tile_name] == extents[tile_name]
    for width in range(1, extents[tile_name] + 1):
        tiled = dict(_stage3_executor(case, probes, **{tile_name: width}).run())
        assert tiled["admission"]["selected_tiles"][tile_name] == width
        for observed, expected in zip(
            _science_arrays(tiled), _science_arrays(baseline)
        ):
            _assert_close(observed, expected, f"{annotation_mode} {tile_name}={width}")
        np.testing.assert_array_equal(
            tiled["group_variant_counts"], baseline["group_variant_counts"]
        )


def test_native_stage3_bd2_tile1_includes_cross_tile_probe_pairs(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=4, name="stage3-bd2-cross-tile")
    probes = _variant_probes(case, 2)
    expected = _oracle(case, probes)
    result = dict(
        _stage3_executor(case, probes, variant_probe_tile=1, group_tile=1).run()
    )
    _assert_close(result["gram"], expected.gram, "BD=2 Gram")
    _assert_close(result["same_person"], expected.same_person, "BD=2 cross-tile D_R")
    _assert_close(
        result["group_gram_unnormalized_num"],
        expected.gram_numerator_contributions,
        "BD=2 grouped numerators",
    )
    assert result["diagnostics"]["variant_probe_coverage"] == 2
    assert result["diagnostics"]["same_person_cross_tile_pairs_included"] is True


def test_native_variant_philox_xi_matches_identical_explicit_xi_across_schedules(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=3, name="stage3-variant-philox")
    keys, probes = _philox_probes(case.retained_variant_rows.size, 4, root_seed=741_921)
    thread_counts = [1]
    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2:
        thread_counts.append(2)

    baseline: dict[str, object] | None = None
    for variant_probe_tile in (0, 1):
        for decode_threads in thread_counts:
            explicit = dict(
                _stage3_executor(
                    case,
                    probes,
                    variant_probe_tile=variant_probe_tile,
                    decode_threads=decode_threads,
                ).run()
            )
            counter = dict(
                _executor(
                    case,
                    variant_probes=None,
                    variant_philox_keys=keys,
                    variant_probe_count=probes.shape[1],
                    variant_probe_tile=variant_probe_tile,
                    group_tile=0,
                    grouped_algorithm="group_restricted_action_v1",
                    direct_grouped_scaling="action_scaled_v1",
                    enable_grouped_differential=False,
                    decode_threads=decode_threads,
                ).run()
            )
            assert explicit["variant_probe_policy"] == (
                "explicit_variant_rademacher_v1"
            )
            assert counter["variant_probe_policy"] == (
                "numpy_philox_variant_per_probe_key_v1"
            )
            assert explicit["variant_probe_identity_sha256"] == array_sha256(probes)
            assert counter["variant_probe_identity_sha256"] == array_sha256(keys)
            assert (
                explicit["variant_probe_identity_sha256"]
                != counter["variant_probe_identity_sha256"]
            )
            assert (
                explicit["variant_probe_count"]
                == counter["variant_probe_count"]
                == probes.shape[1]
            )
            assert explicit["diagnostics"]["variant_probe_coverage"] == probes.shape[1]
            assert counter["diagnostics"]["variant_probe_coverage"] == probes.shape[1]
            for observed, expected in zip(
                _science_arrays(counter), _science_arrays(explicit)
            ):
                _assert_close(
                    observed,
                    expected,
                    f"variant Philox tile={variant_probe_tile} threads={decode_threads}",
                )
            np.testing.assert_array_equal(
                counter["group_variant_counts"], explicit["group_variant_counts"]
            )
            if baseline is None:
                baseline = explicit
            else:
                for observed, expected in zip(
                    _science_arrays(explicit), _science_arrays(baseline)
                ):
                    _assert_close(
                        observed,
                        expected,
                        "explicit variant-probe schedule invariance",
                    )
                for observed, expected in zip(
                    _science_arrays(counter), _science_arrays(baseline)
                ):
                    _assert_close(
                        observed,
                        expected,
                        "counter variant-probe schedule invariance",
                    )


def test_native_stage3_named_off_diagonal_factors_1_2_4(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=2,
        annotation_count=1,
        name="stage3-factors-1-2-4",
    )
    probes = _variant_probes(case, 3)
    components = _components(case)
    eta = np.asarray(
        [entry.kernel_factor for entry in components.entries], dtype=np.float64
    )
    diag = next(entry.index for entry in components.entries if entry.q == entry.r)
    off = next(entry.index for entry in components.entries if entry.q != entry.r)
    assert eta[diag] == 1.0
    assert eta[off] == 2.0
    projector = np.eye(case.fixed_basis.shape[0]) - (
        case.fixed_basis @ case.fixed_basis.T
    )
    features = common_scale_features(case.scaled_genotype, case.phi, projector)
    mass = float(np.sum(case.annotations[:, 0], dtype=np.float64))

    actions = context_kernel_actions(
        case.scaled_genotype,
        case.phi,
        projector,
        case.annotations,
        components,
        case.sample_probes,
    )
    base_raw_actions = actions * mass / eta[:, None, None]
    feature_probe = np.einsum(
        "qnm,nb->qmb", features, case.sample_probes, optimize=True
    )
    feature_action = np.einsum(
        "qnm,anb->aqmb", features, base_raw_actions, optimize=True
    )
    base_directional = np.zeros(
        (len(case.group_names), len(components), len(components)), dtype=np.float64
    )
    weights = case.annotations[:, 0]
    for left in components.entries:
        for right in components.entries:
            contribution = np.sum(
                feature_action[right.index, left.q] * feature_probe[left.r],
                axis=1,
                dtype=np.float64,
            )
            if left.q != left.r:
                contribution += np.sum(
                    feature_action[right.index, left.r] * feature_probe[left.q],
                    axis=1,
                    dtype=np.float64,
                )
                contribution *= 0.5
            np.add.at(
                base_directional[:, left.index, right.index],
                case.group_index,
                weights * contribution,
            )
    base_grouped = (
        0.5
        * (base_directional + np.swapaxes(base_directional, 1, 2))
        / case.sample_probes.shape[1]
    )

    sqrt_weight = np.sqrt(weights)
    source = np.empty(
        (case.phi.shape[1], case.scaled_genotype.shape[0], probes.shape[1]),
        dtype=np.float64,
    )
    for context in range(case.phi.shape[1]):
        source[context] = features[context] @ (sqrt_weight[:, None] * probes)
    base_g = np.empty(
        (len(components), case.scaled_genotype.shape[0], probes.shape[1]),
        dtype=np.float64,
    )
    for component in components.entries:
        base_g[component.index] = source[component.q] * source[component.r] / mass
    probe_sums = np.sum(base_g, axis=2, dtype=np.float64)
    same_probe = np.einsum("aib,cib->ac", base_g, base_g, optimize=True)
    base_same = (probe_sums @ probe_sums.T - same_probe) / (
        probes.shape[1] * (probes.shape[1] - 1)
    )
    base_same = 0.5 * (base_same + base_same.T)

    result = dict(_stage3_executor(case, probes).run())
    grouped = np.asarray(result["group_gram_unnormalized_num"])
    same = np.asarray(result["same_person"])
    factor = np.outer(eta, eta)
    _assert_close(grouped, base_grouped * factor[None, :, :], "group factors")
    _assert_close(same, base_same * factor, "same-person factors")
    _assert_close(grouped[:, diag, diag], base_grouped[:, diag, diag], "factor 1")
    _assert_close(grouped[:, diag, off], 2.0 * base_grouped[:, diag, off], "factor 2")
    _assert_close(grouped[:, off, off], 4.0 * base_grouped[:, off, off], "factor 4")
    _assert_close(same[diag, diag], base_same[diag, diag], "D factor 1")
    _assert_close(same[diag, off], 2.0 * base_same[diag, off], "D factor 2")
    _assert_close(same[off, off], 4.0 * base_same[off, off], "D factor 4")


@pytest.mark.parametrize(
    "annotation_mode",
    ["generic_nonnegative_weights_v1", "strict_disjoint_binary_v1"],
)
def test_native_stage3_restricted_and_both_direct_placements_agree(
    tmp_path: Path, annotation_mode: str
) -> None:
    case = _make_case(
        tmp_path,
        q_count=3,
        annotation_mode=annotation_mode,
        name=f"stage3-direct-placements-{annotation_mode}",
    )
    probes = _variant_probes(case, 3)
    expected = _oracle(case, probes)
    differential_executor = _stage3_executor(
        case,
        probes,
        direct_grouped_scaling="both_differential_v1",
        enable_grouped_differential=True,
    )
    restricted = dict(differential_executor.run())
    snapshot = dict(differential_executor.differential_snapshot())
    for name in (
        "group_gram_numerator_restricted",
        "group_gram_numerator_direct_action_scaled",
        "group_gram_numerator_direct_genotype_scaled",
    ):
        _assert_close(snapshot[name], expected.gram_numerator_contributions, name)
    _assert_close(
        restricted["group_gram_unnormalized_num"],
        expected.gram_numerator_contributions,
        "restricted selected output",
    )

    for scaling in ("action_scaled_v1", "genotype_scaled_v1"):
        result = dict(
            _stage3_executor(
                case,
                probes,
                grouped_algorithm="direct_grouped_tn_v1",
                direct_grouped_scaling=scaling,
                enable_grouped_differential=True,
            ).run()
        )
        _assert_close(
            result["group_gram_unnormalized_num"],
            expected.gram_numerator_contributions,
            f"direct selected {scaling}",
        )
        assert result["diagnostics"]["group_reconstruction_verified"] is True


@pytest.mark.parametrize(
    "annotation_mode",
    ["generic_nonnegative_weights_v1", "strict_disjoint_binary_v1"],
)
@pytest.mark.parametrize("scaling", ["action_scaled_v1", "genotype_scaled_v1"])
def test_native_stage3_selected_direct_only_uses_one_placement_pass(
    tmp_path: Path, annotation_mode: str, scaling: str
) -> None:
    case = _make_case(
        tmp_path,
        q_count=3,
        annotation_mode=annotation_mode,
        name=f"stage3-direct-only-{annotation_mode}-{scaling}",
    )
    probes = _variant_probes(case, 3)
    expected = _oracle(case, probes)
    result = dict(
        _stage3_executor(
            case,
            probes,
            grouped_algorithm="direct_grouped_tn_v1",
            direct_grouped_scaling=scaling,
            enable_grouped_differential=False,
        ).run()
    )
    _assert_close(
        result["gram"], expected.gram, f"direct-only Gram {annotation_mode} {scaling}"
    )
    _assert_close(
        result["same_person"],
        expected.same_person,
        f"direct-only D_R {annotation_mode} {scaling}",
    )
    _assert_close(
        result["group_gram_unnormalized_num"],
        expected.gram_numerator_contributions,
        f"direct-only grouped numerator {annotation_mode} {scaling}",
    )
    assert result["selected_grouped_algorithm"] == "direct_grouped_tn_v1"
    assert result["direct_grouped_scaling"] == scaling
    assert result["grouped_differential_enabled"] is False
    admission = result["admission"]
    assert admission["group_descriptor_passes"] == 1
    assert admission["phase_ledger"]["grouped"]["descriptor_passes"] == 1
    assert admission["phase_ledger"]["grouped"]["selected_algorithm"] == (
        "direct_grouped_tn_v1"
    )
    assert admission["semantic_call_ledger"]["direct_grouped_tn"] > 0
    diagnostics = result["diagnostics"]
    assert diagnostics["direct_action_restricted_max_abs"] == 0.0
    assert diagnostics["direct_genotype_restricted_max_abs"] == 0.0
    assert diagnostics["direct_placement_max_abs"] == 0.0
    assert diagnostics["group_reconstruction_verified"] is True


def test_native_stage3_compact_schema_ownership_and_one_shot_lifecycle(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage3-compact")
    probes = _variant_probes(case)
    executor = _stage3_executor(case, probes)
    result = dict(executor.run())
    assert set(result) == _NATIVE_RESULT_KEYS
    assert result["complete_reference_statistics"] is True
    assert result["complete_reference_artifact"] is False
    assert result["state"] == "reference_statistics_complete_ready"
    assert result["output_ownership"] == {"owns_data": True, "read_only": True}
    for name in (
        "gram",
        "same_person",
        "group_gram_unnormalized_num",
        "annotation_masses",
        "group_annotation_masses",
        "group_variant_counts",
    ):
        value = result[name]
        assert isinstance(value, np.ndarray)
        assert value.flags.owndata
        assert not value.flags.writeable
    gram_copy = np.array(result["gram"], copy=True)
    del executor
    gc.collect()
    _assert_close(result["gram"], gram_copy, "owned result lifetime")
    with pytest.raises(ValueError):
        result["gram"][0, 0] = 0.0


def test_native_stage3_executor_is_one_shot(tmp_path: Path) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage3-one-shot")
    probes = _variant_probes(case)
    executor = _stage3_executor(case, probes)
    executor.run()
    with pytest.raises(RuntimeError, match="one-shot"):
        executor.run()


def test_native_stage3_exact_capacity_and_semantic_reconstruction_failure(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage3-capacity")
    probes = _variant_probes(case)
    estimate = dict(_stage3_executor(case, probes).preflight())
    exact = _stage3_executor(
        case,
        probes,
        workspace_cap_bytes=estimate["required_workspace_bytes"],
        telemetry_capacity=estimate["required_telemetry_capacity"],
    )
    result = dict(exact.run())
    assert result["telemetry"]["capacity"] == estimate["required_telemetry_capacity"]
    assert result["telemetry"]["complete_without_drop"] is True
    with pytest.raises(RuntimeError, match="workspace cap"):
        _stage3_executor(
            case,
            probes,
            workspace_cap_bytes=estimate["required_workspace_bytes"] - 1,
        )
    with pytest.raises(RuntimeError, match="telemetry capacity"):
        _stage3_executor(
            case,
            probes,
            telemetry_capacity=estimate["required_telemetry_capacity"] - 1,
        )
    with pytest.raises(RuntimeError, match="semantic group reconstruction"):
        _stage3_executor(case, probes, fault_mode="semantic_group_reconstruction").run()


def test_native_artifact_all_single_and_selected_multi_deletions_flow_through_fit(
    tmp_path: Path,
) -> None:
    base = _make_case(tmp_path, q_count=2, name="stage3-native-fit")
    group_names = tuple(f"g{index}" for index in range(base.group_index.size))
    case = replace(
        base,
        group_index=np.arange(base.group_index.size, dtype=np.int64),
        group_names=group_names,
    )
    probes = _variant_probes(case, 3)
    components = _components(case)
    projector = rank_revealing_projector(case.fixed_basis)
    groups = tuple(case.group_names[int(index)] for index in case.group_index)
    hashes = {
        "basis_hash": array_sha256(case.phi),
        "fixed_effect_hash": array_sha256(case.fixed_basis),
        "variant_hash": canonical_sha256(
            {"logical_variants": list(case.expected_variant_ids)}
        ),
    }
    oracle = build_context_reference(
        genotype=case.scaled_genotype,
        basis=case.phi,
        projector=projector,
        annotations=case.annotations,
        component_index=components,
        loo_groups=groups,
        genotype_scaling=_stable_scale_plan(case).digest,
        gram_method="hutchinson",
        gram_probes=case.sample_probes,
        same_person_method="ustat",
        variant_probes=probes,
        probe_tile_size=2,
        contribution_storage="loo_grouped",
        **hashes,
    )
    rng = np.random.default_rng(884_311)
    summary = build_context_trait_summary(
        genotype=case.scaled_genotype,
        basis=case.phi,
        phenotype=rng.normal(size=case.fixed_basis.shape[0]),
        projector=projector,
        annotations=case.annotations,
        component_index=components,
        residual_basis=np.asfortranarray(
            rng.normal(size=(case.fixed_basis.shape[0], 1))
        ),
        residual_names=("residual",),
        loo_groups=groups,
        genotype_scaling=_stable_scale_plan(case).digest,
        block_size=3,
        contribution_storage="loo_grouped",
        **hashes,
    )
    artifact = run_contextual_reference_v1(
        _stage3_executor(case, probes), _publication_identity(case)
    )
    bridged = artifact.to_development_grouped_reference(
        annotation_hash=oracle.manifest["annotation_hash"],
        loo_grouping_hash=oracle.manifest["loo_grouping_hash"],
        **hashes,
    )
    grid = np.asarray([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    metric = case.phi.T @ case.phi / case.phi.shape[0]
    observed_fit = fit_context_model(
        bridged,
        summary,
        context_grid=grid,
        basis_metric=metric,
        project_psd=False,
    )
    expected_fit = fit_context_model(
        oracle,
        summary,
        context_grid=grid,
        basis_metric=metric,
        project_psd=False,
    )
    _assert_close(
        observed_fit.raw_coefficients,
        expected_fit.raw_coefficients,
        "native artifact full fit",
    )
    assert observed_fit.loo_coefficients.shape[0] == len(group_names)
    _assert_close(
        observed_fit.loo_coefficients,
        expected_fit.loo_coefficients,
        "all single-group deletion fits",
    )
    for observed, expected in zip(observed_fit.raw_omegas, expected_fit.raw_omegas):
        _assert_close(observed, expected, "full Omega")
    for observed_row, expected_row in zip(
        observed_fit.loo_coefficients, expected_fit.loo_coefficients
    ):
        for observed, expected in zip(
            coefficients_to_omegas(observed_row[: len(components)], components),
            coefficients_to_omegas(expected_row[: len(components)], components),
        ):
            _assert_close(observed, expected, "single-deletion Omega")
    for observed, expected in zip(
        observed_fit.context_outputs["annotations"],
        expected_fit.context_outputs["annotations"],
    ):
        _assert_close(
            observed["covariance_surface"],
            expected["covariance_surface"],
            "full covariance surface",
        )
    assert len(observed_fit.jackknife_context_outputs) == len(group_names)
    for observed_replicate, expected_replicate in zip(
        observed_fit.jackknife_context_outputs,
        expected_fit.jackknife_context_outputs,
    ):
        for observed, expected in zip(
            observed_replicate["annotations"],
            expected_replicate["annotations"],
        ):
            _assert_close(
                observed["covariance_surface"],
                expected["covariance_surface"],
                "single-deletion covariance surface",
            )

    for deleted in (("g0", "g3"), ("g1", "g4", "g7")):
        observed_equations = assemble_context_normal_equations(
            bridged, summary, deleted
        )
        expected_equations = assemble_context_normal_equations(oracle, summary, deleted)
        observed_solve = solve_context_normal_equations(observed_equations)
        expected_solve = solve_context_normal_equations(expected_equations)
        _assert_close(
            observed_solve.coefficients,
            expected_solve.coefficients,
            f"multi-delete fit {deleted}",
        )
        observed_outputs = derive_context_outputs(
            observed_solve.coefficients[: len(components)], components, grid, metric
        )
        expected_outputs = derive_context_outputs(
            expected_solve.coefficients[: len(components)], components, grid, metric
        )
        for observed, expected in zip(
            observed_outputs["annotations"], expected_outputs["annotations"]
        ):
            _assert_close(
                observed["covariance_surface"],
                expected["covariance_surface"],
                f"multi-delete covariance surface {deleted}",
            )


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_names", "missing_group", "unknown_group", "empty_annotation"],
)
def test_native_stage3_group_contract_failures(tmp_path: Path, mutation: str) -> None:
    mode = (
        "strict_disjoint_binary_v1"
        if mutation == "empty_annotation"
        else "generic_nonnegative_weights_v1"
    )
    case = _make_case(tmp_path, q_count=2, annotation_mode=mode, name=mutation)
    probes = _variant_probes(case)
    changes: dict[str, object] = {}
    if mutation == "duplicate_names":
        changes["group_names"] = ["g0", "g0", "g2"]
    elif mutation == "missing_group":
        changes["group_index"] = np.zeros_like(case.group_index)
    elif mutation == "unknown_group":
        group_index = case.group_index.copy()
        group_index[0] = len(case.group_names)
        changes["group_index"] = group_index
    else:
        changes["group_index"] = np.arange(case.group_index.size, dtype=np.int64) % 2
        changes["group_names"] = ["a0-only", "a1-only"]
    with pytest.raises(RuntimeError):
        _stage3_executor(case, probes, **changes)


def test_native_stage3_sealed_group_permutation_mutation_is_detected(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage3-group-order-mutation")
    probes = _variant_probes(case)
    executor = _stage3_executor(
        case,
        probes,
        test_checkpoint="post_seal",
        test_mutation_target="group_execution_order",
    )
    error = _run_checkpoint_failure(executor, "post_seal")
    assert "mutation" in str(error).lower() or "identity" in str(error).lower()


STAGE3_OPERATIONS = (
    "group_target_nn",
    "group_cross_gram_tn",
    "direct_grouped_tn",
    "same_person_target_nn",
    "same_person_projection_tn",
    "same_person_projection_nn",
    "same_person_gram_tn",
)


def _fault_executor(
    case: object, probes: np.ndarray, operation: str, mode: str
) -> object:
    differential = operation == "direct_grouped_tn"
    anchor_executor = _stage3_executor(
        case,
        probes,
        direct_grouped_scaling=(
            "both_differential_v1" if differential else "action_scaled_v1"
        ),
        enable_grouped_differential=differential,
    )
    anchor = next(
        item["semantic_anchor"]
        for item in anchor_executor.semantic_anchors()
        if item["operation"] == operation
    )
    return _stage3_executor(
        case,
        probes,
        direct_grouped_scaling=(
            "both_differential_v1" if differential else "action_scaled_v1"
        ),
        enable_grouped_differential=differential,
        fault_operation=operation,
        fault_mode=mode,
        fault_semantic_anchor=anchor,
    )


@pytest.mark.parametrize("operation", STAGE3_OPERATIONS)
@pytest.mark.parametrize(
    "mode",
    [
        "one_shot",
        "repeated",
        "repair_corruption",
        "force_fallback",
        "nan",
        "inf",
        "canary",
    ],
)
def test_native_stage3_semantic_operations_recover_without_science_change(
    tmp_path: Path, operation: str, mode: str
) -> None:
    case = _make_case(tmp_path, q_count=2, name=f"stage3-{operation}-{mode}")
    probes = _variant_probes(case)
    differential = operation == "direct_grouped_tn"
    baseline = dict(
        _stage3_executor(
            case,
            probes,
            direct_grouped_scaling=(
                "both_differential_v1" if differential else "action_scaled_v1"
            ),
            enable_grouped_differential=differential,
        ).run()
    )
    result = dict(_fault_executor(case, probes, operation, mode).run())
    for observed, expected in zip(_science_arrays(result), _science_arrays(baseline)):
        _assert_close(observed, expected, f"{operation} {mode}")
    telemetry = result["telemetry"]
    assert telemetry["injection_count"] == 1
    if mode in {"one_shot", "nan", "inf", "canary"}:
        assert telemetry["repair_count"] == 1
    elif mode in {"repeated", "repair_corruption"}:
        assert telemetry["fallback_count"] == 1
    else:
        assert telemetry["fallback_count"] == 1


@pytest.mark.parametrize("operation", STAGE3_OPERATIONS)
@pytest.mark.parametrize(
    "mode",
    [
        "fallback_failure",
        "fallback_corruption",
        "operand_mutation",
        "runtime_mutation",
    ],
)
def test_native_stage3_semantic_operations_terminal_fallback_failure(
    tmp_path: Path, operation: str, mode: str
) -> None:
    case = _make_case(tmp_path, q_count=2, name=f"stage3-terminal-{operation}-{mode}")
    probes = _variant_probes(case)
    executor = _fault_executor(case, probes, operation, mode)
    with pytest.raises(RuntimeError):
        executor.run()
    with pytest.raises(RuntimeError, match="one-shot"):
        executor.run()
