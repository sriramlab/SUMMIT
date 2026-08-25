from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from summit.context import (
    APPROXIMATE_DELETION_CONTRACT,
    NATIVE_OWNED_RESPONSIBILITIES,
    OPTIONAL_INTERPRETATION_FIELDS,
    PYTHON_OWNED_RESPONSIBILITIES,
    RAW_FIT_FIELDS,
    AnnotationMode,
    ArtifactFamily,
    BasisColumnSpec,
    ContextAdmissionLedgerV1,
    ContextBasisSpec,
    ContextComponentIndex,
    ContextPairIndex,
    ContextSchemaIdentityV1,
    DeletionSemantics,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
    GroupedEncodingVersion,
    LogicalSchemaVersion,
    SemanticOperation,
    annotation_output_contract,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    build_disjoint_annotation_partition,
    canonical_sha256,
    checked_add_u64,
    checked_mul_u64,
    checked_u64,
    fit_context_model,
    load_context_reference,
    rank_revealing_projector,
    require_identical_scale_plans,
    validate_context_manifest,
    write_context_reference,
)
from summit.context.schema import UINT64_MAX


PAIR_INDEX_DIGESTS = {
    1: "7ad52c53d2e60a48ee481335655da773de7c368f19058749ce22b428f31ee651",
    2: "ed5af052fb162b216aec1377053029d6781f1b2b8d5371ac5acff97086c7f2e1",
    3: "a9895e7060834060a736dd4e8dd7aa1abb89811baf57796ec0365254e16e8723",
    4: "6b43a6f4f98014960db86bd354288b6dbb2262eb0e5c45246e7290eeb3464c24",
}

SEMANTIC_OPERATION_VALUES = (
    "sample_probe_projection_tn",
    "sample_probe_projection_nn",
    "source_tn",
    "full_target_nn",
    "action_projection_tn",
    "action_projection_nn",
    "action_gram_tn",
    "group_target_nn",
    "group_cross_gram_tn",
    "direct_grouped_tn",
    "same_person_target_nn",
    "same_person_projection_tn",
    "same_person_projection_nn",
    "same_person_gram_tn",
    "trait_score_tn",
    "trait_feature_projection_tn",
    "trait_feature_projection_nn",
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _scale_plan(**changes: object) -> GenotypeScalePlanV1:
    values: dict[str, object] = {
        "policy": GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1,
        "retained_variant_order_sha256": _sha("variant-order"),
        "allele_orientation": "counted_allele_is_a1",
        "allele_coding": "a1_dosage_in_0_2",
        "centering_source": "sealed_reference_affine_vector",
        "centering_formula": "dosage_minus_mean",
        "scaling_formula": "multiply_inverse_standard_deviation",
        "missing_imputation": "sealed_variant_mean_before_affine_transform",
        "ploidy_policy": "diploid_autosome_only",
        "affine_mean_sha256": _sha("means"),
        "affine_inverse_scale_sha256": _sha("inverse-scales"),
    }
    values.update(changes)
    return GenotypeScalePlanV1(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def stage0_artifacts() -> dict[str, object]:
    rng = np.random.default_rng(91231)
    n_samples, n_variants = 24, 12
    genotype = rng.normal(size=(n_samples, n_variants))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    basis = np.ones((n_samples, 1), dtype=np.float64)
    fixed = np.column_stack([np.ones(n_samples), rng.normal(size=n_samples)])
    projector = rank_revealing_projector(fixed)
    phenotype = rng.normal(size=n_samples)
    annotations = np.zeros((n_variants, 2), dtype=np.float64)
    annotations[np.arange(n_variants), np.arange(n_variants) % 2] = 1.0
    annotation_names = ("even", "odd")
    components = ContextComponentIndex(annotation_names, ContextPairIndex(1))
    loo_groups = tuple(f"group:{index // 2}" for index in range(n_variants))
    variant_hash = canonical_sha256({"variants": list(range(n_variants))})
    common = {
        "genotype": genotype,
        "basis": basis,
        "projector": projector,
        "annotations": annotations,
        "component_index": components,
        "loo_groups": loo_groups,
        "basis_hash": array_sha256(basis),
        "fixed_effect_hash": array_sha256(fixed),
        "variant_hash": variant_hash,
        "genotype_scaling": "pre_scaled_input",
    }
    reference = build_context_reference(
        **common,
        gram_method="exact",
        same_person_method="exact",
    )
    trait = build_context_trait_summary(
        **common,
        phenotype=phenotype,
        residual_basis=np.ones((n_samples, 1), dtype=np.float64),
        residual_names=("residual:constant",),
        block_size=4,
    )
    fit = fit_context_model(reference, trait)
    partition = build_disjoint_annotation_partition(
        annotations,
        annotation_names,
        definitions=(
            {"kind": "parity", "nested": {"labels": ["even"]}},
            {"kind": "parity", "nested": {"labels": ["odd"]}},
        ),
        variant_hash=variant_hash,
        loo_groups=loo_groups,
        source="stage0_schema_test",
    )
    return {
        "partition": partition,
        "reference": reference,
        "trait": trait,
        "fit": fit,
    }


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_q1_q4_pair_index_digests_are_frozen(q: int) -> None:
    assert ContextPairIndex(q).digest == PAIR_INDEX_DIGESTS[q]


def test_array_hash_is_endian_and_layout_canonical_and_rejects_objects() -> None:
    logical = np.arange(12, dtype=np.float64).reshape(3, 4)
    fortran = np.asfortranarray(logical)
    big_endian = logical.astype(">f8")
    backing = np.full((3, 8), np.nan, dtype=np.float64)
    noncontiguous = backing[:, ::2]
    noncontiguous[...] = logical

    expected = array_sha256(logical)
    assert array_sha256(fortran) == expected
    assert array_sha256(big_endian) == expected
    assert array_sha256(noncontiguous) == expected
    assert array_sha256(logical.astype(np.float32)) != expected
    with pytest.raises(ValueError, match="object dtype"):
        array_sha256(np.asarray([{"not": "stable"}], dtype=object))


def test_spec_sequences_and_nested_parameters_are_caller_independent() -> None:
    category = ["baseline", {"code": 0}]
    one_hot = BasisColumnSpec(
        "group",
        "one_hot",
        "group",
        parameters=(("category", category),),
    )
    columns = [BasisColumnSpec("constant", "constant"), one_hot]
    basis_spec = ContextBasisSpec("stage0_basis", columns)  # type: ignore[arg-type]
    annotation_names = ["left", "right"]
    components = ContextComponentIndex(  # type: ignore[arg-type]
        annotation_names, ContextPairIndex(2)
    )
    spec_digest = basis_spec.digest
    component_digest = components.digest

    category.append("caller-mutation")
    columns.append(BasisColumnSpec("external", "linear", "external"))
    annotation_names.append("caller-mutation")

    assert basis_spec.digest == spec_digest
    assert basis_spec.names == ("constant", "group")
    assert components.digest == component_digest
    assert components.annotation_names == ("left", "right")
    stored_category = one_hot.parameters[0][1]
    assert stored_category == ["baseline", {"code": 0}]
    with pytest.raises(TypeError, match="immutable"):
        stored_category.append("artifact-mutation")
    with pytest.raises(TypeError, match="immutable"):
        stored_category[1]["code"] = 1


def _view_alias(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_shape = (*value.shape[:-1], value.shape[-1] * 2)
    base = np.empty(base_shape, dtype=value.dtype)
    view = base[..., ::2]
    view[...] = value
    return base, view, np.array(value, copy=True)


@pytest.mark.parametrize(
    ("artifact_name", "field"),
    [
        ("partition", "weights"),
        ("reference", "annotation_weights"),
        ("trait", "genetic_rhs"),
        ("fit", "raw_coefficients"),
    ],
)
def test_artifact_arrays_own_view_inputs_and_are_read_only(
    stage0_artifacts: dict[str, object], artifact_name: str, field: str
) -> None:
    artifact = stage0_artifacts[artifact_name]
    base, view, expected = _view_alias(np.asarray(getattr(artifact, field)))
    frozen = replace(artifact, **{field: view})
    stored = np.asarray(getattr(frozen, field))

    assert stored.flags.owndata
    assert stored.flags.c_contiguous
    assert not stored.flags.writeable
    assert not np.shares_memory(stored, base)
    base[...] = -9876
    np.testing.assert_array_equal(stored, expected)
    with pytest.raises(ValueError, match="read-only"):
        stored.flat[0] = 0


def test_artifact_manifests_are_recursively_owned_and_immutable(
    stage0_artifacts: dict[str, object],
) -> None:
    for artifact_name in ("partition", "reference", "trait", "fit"):
        artifact = stage0_artifacts[artifact_name]
        caller_manifest = copy.deepcopy(artifact.manifest)
        caller_manifest["stage0_nested"] = {"labels": ["sealed"]}
        frozen = replace(artifact, manifest=caller_manifest)

        caller_manifest["stage0_nested"]["labels"].append("caller-mutation")
        assert frozen.manifest["stage0_nested"]["labels"] == ["sealed"]
        with pytest.raises(TypeError, match="immutable"):
            frozen.manifest["stage0_nested"]["labels"].append("artifact-mutation")
        with pytest.raises(TypeError, match="immutable"):
            frozen.manifest["stage0_nested"]["new_field"] = True


def test_partition_definitions_and_fit_outputs_are_recursively_frozen(
    stage0_artifacts: dict[str, object],
) -> None:
    partition = stage0_artifacts["partition"]
    definitions = [
        {"nested": {"labels": [name]}} for name in partition.annotation_names
    ]
    frozen_partition = replace(partition, definitions=definitions)
    definitions[0]["nested"]["labels"].append("caller-mutation")
    assert frozen_partition.definitions[0]["nested"]["labels"] == ["even"]
    with pytest.raises(TypeError, match="immutable"):
        frozen_partition.definitions[0]["nested"]["labels"].append(
            "artifact-mutation"
        )

    fit = stage0_artifacts["fit"]
    backing = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = backing[:, ::2]
    outputs = {"surface": {"labels": ["a", "b"], "values": view}}
    expected = view.copy()
    frozen_fit = replace(fit, context_outputs=outputs)
    outputs["surface"]["labels"].append("caller-mutation")
    backing[...] = -1.0
    assert frozen_fit.context_outputs["surface"]["labels"] == ["a", "b"]
    np.testing.assert_array_equal(
        frozen_fit.context_outputs["surface"]["values"], expected
    )
    with pytest.raises(TypeError, match="immutable"):
        frozen_fit.context_outputs["surface"]["labels"].append(
            "artifact-mutation"
        )
    with pytest.raises(ValueError, match="read-only"):
        frozen_fit.context_outputs["surface"]["values"].flat[0] = 0.0


def test_schema_identity_keeps_each_version_axis_explicit() -> None:
    identity = ContextSchemaIdentityV1(
        artifact_family=ArtifactFamily.REFERENCE,
        logical_schema_version=LogicalSchemaVersion.REFERENCE_V1,
        grouped_encoding_version=GroupedEncodingVersion.DENSE_UNNORMALIZED_V1,
        native_backend_version="gxeldcore-1.9",
        build_id="build-a",
    )
    assert identity.to_dict() == {
        "artifact_family": "contextual_reference",
        "logical_schema_version": "contextual_reference_v1",
        "grouped_encoding_version": "grouped_unnormalized_dense_v1",
        "native_api_version": 1,
        "native_backend_version": "gxeldcore-1.9",
        "build_id": "build-a",
    }
    assert replace(identity, native_backend_version="gxeldcore-2.0").digest != (
        identity.digest
    )
    assert replace(identity, build_id="build-b").digest != identity.digest
    trait_identity = replace(
        identity,
        artifact_family=ArtifactFamily.TRAIT,
        logical_schema_version=LogicalSchemaVersion.TRAIT_V1,
    )
    assert trait_identity.digest != identity.digest
    with pytest.raises(ValueError, match="disagree"):
        replace(identity, logical_schema_version=LogicalSchemaVersion.FIT_V1)
    with pytest.raises(ValueError):
        ArtifactFamily("unversioned_context_blob")
    with pytest.raises(ValueError):
        LogicalSchemaVersion("contextual_reference")
    with pytest.raises(ValueError):
        GroupedEncodingVersion("grouped_unknown")
    with pytest.raises(ValueError, match="schema version"):
        ContextBasisSpec(
            "bool_version",
            (BasisColumnSpec("constant", "constant"),),
            schema_version=True,
        )
    with pytest.raises(ValueError, match="closed V1 enum"):
        replace(identity, grouped_encoding_version="grouped_unknown")
    for invalid_api_version in (True, 1.0, 2):
        with pytest.raises(ValueError, match="native API version"):
            replace(identity, native_api_version=invalid_api_version)


def test_development_manifest_does_not_accept_boolean_schema_version(
    stage0_artifacts: dict[str, object],
) -> None:
    payload = copy.deepcopy(stage0_artifacts["reference"].manifest)
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        validate_context_manifest(payload)


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("policy", GenotypeScalePolicy.PRE_SCALED_DENSE_V1),
        ("retained_variant_order_sha256", _sha("other-order")),
        ("allele_orientation", "counted_allele_is_a2"),
        ("allele_coding", "a2_dosage_in_0_2"),
        ("centering_source", "trait_sample_means"),
        ("centering_formula", "mean_minus_dosage"),
        ("scaling_formula", "divide_standard_deviation"),
        ("missing_imputation", "zero_after_affine_transform"),
        ("ploidy_policy", "sex_aware_x_chromosome"),
        ("affine_mean_sha256", _sha("other-means")),
        ("affine_inverse_scale_sha256", _sha("other-inverse-scales")),
    ],
)
def test_scale_plan_compatibility_requires_exact_identity(
    field: str, different_value: object
) -> None:
    reference = _scale_plan()
    independent_copy = _scale_plan()
    assert (
        require_identical_scale_plans(reference, independent_copy)
        == reference.digest
    )
    with pytest.raises(ValueError, match="differ"):
        require_identical_scale_plans(
            reference,
            _scale_plan(**{field: different_value}),
        )


def test_scale_policy_is_closed() -> None:
    assert tuple(policy.value for policy in GenotypeScalePolicy) == (
        "pre_scaled_dense_v1",
        "sealed_variant_affine_v1",
    )
    with pytest.raises(ValueError):
        GenotypeScalePolicy("free_form_scaling_description")


def test_annotation_modes_freeze_strict_and_overlap_interpretation() -> None:
    strict = annotation_output_contract(AnnotationMode.STRICT_DISJOINT_BINARY_V1)
    overlap = annotation_output_contract(
        AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1
    )
    assert strict == {
        "per_annotation_role": "standalone_covariance",
        "standalone_annotation_covariance_allowed": True,
        "combined_total_surface_required": False,
    }
    assert overlap == {
        "per_annotation_role": "conditional_contribution",
        "standalone_annotation_covariance_allowed": False,
        "combined_total_surface_required": True,
    }
    with pytest.raises(TypeError, match="immutable"):
        overlap["per_annotation_role"] = "standalone_covariance"
    with pytest.raises(ValueError, match="closed V1 enum"):
        annotation_output_contract("overlap")  # type: ignore[arg-type]


def test_deletion_raw_interpretation_and_responsibility_contracts_are_frozen() -> None:
    assert tuple(DeletionSemantics) == (
        DeletionSemantics.APPROXIMATE_SUMMARY_ONLY_V1,
    )
    assert APPROXIMATE_DELETION_CONTRACT == {
        "semantics": "approximate_summary_only_v1",
        "grouped_storage": "unnormalized_numerators",
        "same_person": "reuse_full_unchanged",
        "residual_only_moments": "reuse_full_unchanged",
        "empty_annotation": "reject",
        "claim": "approximate_summary_only",
    }
    with pytest.raises(TypeError, match="immutable"):
        APPROXIMATE_DELETION_CONTRACT["claim"] = "exact"

    assert RAW_FIT_FIELDS == (
        "raw_coefficients",
        "raw_omegas",
        "raw_rank",
        "raw_condition_diagnostics",
        "raw_loo_coefficients",
        "raw_jackknife_covariance",
    )
    assert OPTIONAL_INTERPRETATION_FIELDS == (
        "psd_coefficients",
        "psd_omegas",
        "regularized_coefficients",
    )
    assert set(NATIVE_OWNED_RESPONSIBILITIES).isdisjoint(
        PYTHON_OWNED_RESPONSIBILITIES
    )
    assert "descriptor_and_file_identity" in NATIVE_OWNED_RESPONSIBILITIES
    assert (
        "small_normal_system_assembly_and_raw_rank_checked_solve"
        in PYTHON_OWNED_RESPONSIBILITIES
    )
    with pytest.raises(ValueError, match="closed genotype-scale enum"):
        _scale_plan(policy="free_form_scaling_description")


def _admission_ledger(**changes: object) -> ContextAdmissionLedgerV1:
    values: dict[str, object] = {
        "permanent_bytes": 100,
        "phase_bytes": {
            "source": 10,
            "action": 20,
            "grouped": 30,
            "same_person": 40,
            "trait": 50,
        },
        "retry_bytes": 7,
        "trusted_fallback_bytes": 11,
        "telemetry_bytes": 13,
        "compact_output_bytes": 17,
        "headroom_bytes": 19,
        "semantic_call_counts": {
            operation: index + 1
            for index, operation in enumerate(SemanticOperation)
        },
        "maximum_events_per_call": 3,
        "phase_checkpoint_events": 5,
        "repair_fallback_events": 7,
        "semantic_verification_events": 11,
        "publication_events": 13,
    }
    values.update(changes)
    return ContextAdmissionLedgerV1(**values)  # type: ignore[arg-type]


def test_semantic_operation_and_admission_formulas_are_exact() -> None:
    assert tuple(operation.value for operation in SemanticOperation) == (
        SEMANTIC_OPERATION_VALUES
    )
    ledger = _admission_ledger()
    assert ledger.protected_calls == sum(range(1, 18)) == 153
    assert ledger.telemetry_events == 153 * 3 + 5 + 7 + 11 + 13 == 495
    assert ledger.peak_bytes == 100 + 50 + 7 + 11 + 13 + 17 + 19 == 217
    assert tuple(sorted(ledger.semantic_call_counts)) == tuple(
        sorted(SEMANTIC_OPERATION_VALUES)
    )
    with pytest.raises(TypeError, match="immutable"):
        ledger.phase_bytes["source"] = 0
    with pytest.raises(TypeError, match="immutable"):
        ledger.semantic_call_counts[SEMANTIC_OPERATION_VALUES[0]] = 0

    incomplete = {operation: 0 for operation in tuple(SemanticOperation)[:-1]}
    with pytest.raises(ValueError, match="enumerate every operation"):
        _admission_ledger(semantic_call_counts=incomplete)


def test_admission_arithmetic_rejects_uint64_overflow() -> None:
    assert checked_u64("limit", UINT64_MAX) == UINT64_MAX
    assert checked_add_u64("sum", UINT64_MAX - 1, 1) == UINT64_MAX
    assert checked_mul_u64("product", UINT64_MAX, 1) == UINT64_MAX
    for invalid in (-1, UINT64_MAX + 1, True):
        with pytest.raises(OverflowError):
            checked_u64("invalid", invalid)
    with pytest.raises(OverflowError, match="exceeds"):
        checked_add_u64("sum", UINT64_MAX, 1)
    with pytest.raises(OverflowError, match="exceeds"):
        checked_mul_u64("product", UINT64_MAX, 2)

    overflow_calls = {operation: 0 for operation in SemanticOperation}
    overflow_calls[SemanticOperation.SOURCE_TN] = UINT64_MAX
    overflow_calls[SemanticOperation.FULL_TARGET_NN] = 1
    with pytest.raises(OverflowError, match="protected_calls"):
        _admission_ledger(semantic_call_counts=overflow_calls).protected_calls

    overflow_events = {operation: 0 for operation in SemanticOperation}
    overflow_events[SemanticOperation.SOURCE_TN] = UINT64_MAX
    with pytest.raises(OverflowError, match="protected_call_events"):
        _admission_ledger(
            semantic_call_counts=overflow_events,
            maximum_events_per_call=2,
        ).telemetry_events

    zero_phases = {
        phase: 0
        for phase in ("source", "action", "grouped", "same_person", "trait")
    }
    zero_calls = {operation: 0 for operation in SemanticOperation}
    with pytest.raises(OverflowError, match="peak_bytes"):
        _admission_ledger(
            permanent_bytes=UINT64_MAX,
            phase_bytes=zero_phases,
            retry_bytes=1,
            trusted_fallback_bytes=0,
            telemetry_bytes=0,
            compact_output_bytes=0,
            headroom_bytes=0,
            semantic_call_counts=zero_calls,
            maximum_events_per_call=0,
            phase_checkpoint_events=0,
            repair_fallback_events=0,
            semantic_verification_events=0,
            publication_events=0,
        ).peak_bytes


def test_reference_loader_rejects_component_index_manifest_tamper(
    stage0_artifacts: dict[str, object], tmp_path
) -> None:
    manifest_path, _ = write_context_reference(
        stage0_artifacts["reference"], tmp_path / "reference"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["component_index_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="component-index digest mismatch"):
        load_context_reference(manifest_path)


def test_context_reference_loader_rejects_legacy_kind_before_artifact_io(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "legacy-reference.json"
    manifest_path.write_text(
        json.dumps({"kind": "summit.gxe.reference", "schema_version": 4}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported context manifest kind"):
        load_context_reference(manifest_path)


def test_gxe_reference_loader_rejects_context_kind_before_artifact_io(
    tmp_path,
) -> None:
    from summit.ldscore import gxe_score

    manifest_path = tmp_path / "context-reference.json"
    manifest_path.write_text(
        json.dumps({"kind": "summit.context.reference", "schema_version": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema-v4.*GxE reference"):
        gxe_score._validate_reference_manifest(manifest_path)
