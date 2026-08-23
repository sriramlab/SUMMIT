"""Closed V1 identity, scaling, ownership, and admission contracts.

Stable descriptor-backed reference and trait executors and the summary-side
fit use these contracts. This module does not itself execute kernels or write
artifacts, and the older ``CONTEXT_SCHEMA_VERSION`` development formats remain
separate from the stable V1 families.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .spec import canonical_sha256, freeze_context_mapping


CONTEXT_NATIVE_API_VERSION = 1
UINT64_MAX = (1 << 64) - 1


class ArtifactFamily(str, Enum):
    REFERENCE = "contextual_reference"
    TRAIT = "contextual_trait"
    FIT = "contextual_fit"


class LogicalSchemaVersion(str, Enum):
    REFERENCE_V1 = "contextual_reference_v1"
    TRAIT_V1 = "contextual_trait_v1"
    FIT_V1 = "contextual_fit_v1"


class GroupedEncodingVersion(str, Enum):
    DENSE_UNNORMALIZED_V1 = "grouped_unnormalized_dense_v1"


class GenotypeScalePolicy(str, Enum):
    PRE_SCALED_DENSE_V1 = "pre_scaled_dense_v1"
    SEALED_VARIANT_AFFINE_V1 = "sealed_variant_affine_v1"


class AnnotationMode(str, Enum):
    STRICT_DISJOINT_BINARY_V1 = "strict_disjoint_binary_v1"
    GENERIC_NONNEGATIVE_WEIGHTS_V1 = "generic_nonnegative_weights_v1"


class DeletionSemantics(str, Enum):
    APPROXIMATE_SUMMARY_ONLY_V1 = "approximate_summary_only_v1"


class GroupedAttributionAlgorithm(str, Enum):
    DIRECT_GROUPED_TN_V1 = "direct_grouped_tn_v1"
    GROUP_RESTRICTED_ACTION_V1 = "group_restricted_action_v1"
    AUTO_QUALIFIED_V1 = "auto_qualified_v1"


class SemanticOperation(str, Enum):
    SAMPLE_PROBE_PROJECTION_TN = "sample_probe_projection_tn"
    SAMPLE_PROBE_PROJECTION_NN = "sample_probe_projection_nn"
    SOURCE_TN = "source_tn"
    FULL_TARGET_NN = "full_target_nn"
    ACTION_PROJECTION_TN = "action_projection_tn"
    ACTION_PROJECTION_NN = "action_projection_nn"
    ACTION_GRAM_TN = "action_gram_tn"
    GROUP_TARGET_NN = "group_target_nn"
    GROUP_CROSS_GRAM_TN = "group_cross_gram_tn"
    DIRECT_GROUPED_TN = "direct_grouped_tn"
    SAME_PERSON_TARGET_NN = "same_person_target_nn"
    SAME_PERSON_PROJECTION_TN = "same_person_projection_tn"
    SAME_PERSON_PROJECTION_NN = "same_person_projection_nn"
    SAME_PERSON_GRAM_TN = "same_person_gram_tn"
    TRAIT_SCORE_TN = "trait_score_tn"
    TRAIT_FEATURE_PROJECTION_TN = "trait_feature_projection_tn"
    TRAIT_FEATURE_PROJECTION_NN = "trait_feature_projection_nn"


NATIVE_OWNED_RESPONSIBILITIES = (
    "descriptor_and_file_identity",
    "decode_impute_and_scale",
    "wide_projection_and_protected_operations",
    "fixed_probe_generation",
    "full_and_grouped_sufficient_statistics",
    "scratch_thread_numa_integrity_and_telemetry",
    "phase_transitions_mutation_checks_and_compact_output",
)

PYTHON_OWNED_RESPONSIBILITIES = (
    "basis_annotation_and_group_specification",
    "native_output_and_schema_validation",
    "population_transfer_and_deletion_renormalization",
    "small_normal_system_assembly_and_raw_rank_checked_solve",
    "optional_psd_or_regularized_interpretation",
    "jackknife_covariance_surfaces_and_artifact_io",
)

RAW_FIT_FIELDS = (
    "raw_coefficients",
    "raw_omegas",
    "raw_rank",
    "raw_condition_diagnostics",
    "raw_loo_coefficients",
    "raw_jackknife_covariance",
)

OPTIONAL_INTERPRETATION_FIELDS = (
    "psd_coefficients",
    "psd_omegas",
    "regularized_coefficients",
)

APPROXIMATE_DELETION_CONTRACT = freeze_context_mapping(
    {
        "semantics": DeletionSemantics.APPROXIMATE_SUMMARY_ONLY_V1.value,
        "grouped_storage": "unnormalized_numerators",
        "same_person": "reuse_full_unchanged",
        "residual_only_moments": "reuse_full_unchanged",
        "empty_annotation": "reject",
        "claim": "approximate_summary_only",
    }
)

_ANNOTATION_OUTPUT_CONTRACTS = {
    AnnotationMode.STRICT_DISJOINT_BINARY_V1: freeze_context_mapping(
        {
            "per_annotation_role": "standalone_covariance",
            "standalone_annotation_covariance_allowed": True,
            "combined_total_surface_required": False,
        }
    ),
    AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1: freeze_context_mapping(
        {
            "per_annotation_role": "conditional_contribution",
            "standalone_annotation_covariance_allowed": False,
            "combined_total_surface_required": True,
        }
    ),
}


def annotation_output_contract(mode: AnnotationMode) -> Mapping[str, Any]:
    if not isinstance(mode, AnnotationMode):
        raise ValueError("annotation mode must use a closed V1 enum value.")
    return _ANNOTATION_OUTPUT_CONTRACTS[mode]


def _sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest.") from exc
    return value.lower()


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def checked_u64(name: str, value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= UINT64_MAX
    ):
        raise OverflowError(f"{name} is not an unsigned 64-bit value: {value!r}.")
    return value


def checked_add_u64(name: str, *values: int) -> int:
    result = 0
    for value in values:
        result += checked_u64(name, value)
        if result > UINT64_MAX:
            raise OverflowError(f"{name} exceeds uint64 capacity.")
    return result


def checked_mul_u64(name: str, *values: int) -> int:
    result = 1
    for value in values:
        item = checked_u64(name, value)
        if item and result > UINT64_MAX // item:
            raise OverflowError(f"{name} exceeds uint64 capacity.")
        result *= item
    return result


@dataclass(frozen=True)
class ContextSchemaIdentityV1:
    artifact_family: ArtifactFamily
    logical_schema_version: LogicalSchemaVersion
    grouped_encoding_version: GroupedEncodingVersion
    native_backend_version: str
    build_id: str
    native_api_version: int = CONTEXT_NATIVE_API_VERSION

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("artifact_family", ArtifactFamily),
            ("logical_schema_version", LogicalSchemaVersion),
            ("grouped_encoding_version", GroupedEncodingVersion),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise ValueError(f"{name} must use a closed V1 enum value.")
        if (
            isinstance(self.native_api_version, bool)
            or not isinstance(self.native_api_version, int)
            or self.native_api_version != CONTEXT_NATIVE_API_VERSION
        ):
            raise ValueError("Unsupported contextual native API version.")
        expected = {
            ArtifactFamily.REFERENCE: LogicalSchemaVersion.REFERENCE_V1,
            ArtifactFamily.TRAIT: LogicalSchemaVersion.TRAIT_V1,
            ArtifactFamily.FIT: LogicalSchemaVersion.FIT_V1,
        }[self.artifact_family]
        if self.logical_schema_version is not expected:
            raise ValueError("Artifact family and logical schema version disagree.")
        _nonempty("native_backend_version", self.native_backend_version)
        _nonempty("build_id", self.build_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_family": self.artifact_family.value,
            "logical_schema_version": self.logical_schema_version.value,
            "grouped_encoding_version": self.grouped_encoding_version.value,
            "native_api_version": self.native_api_version,
            "native_backend_version": self.native_backend_version,
            "build_id": self.build_id,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class GenotypeScalePlanV1:
    policy: GenotypeScalePolicy
    retained_variant_order_sha256: str
    allele_orientation: str
    allele_coding: str
    centering_source: str
    centering_formula: str
    scaling_formula: str
    missing_imputation: str
    ploidy_policy: str
    affine_mean_sha256: str
    affine_inverse_scale_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, GenotypeScalePolicy):
            raise ValueError("policy must use a closed genotype-scale enum value.")
        for name in (
            "retained_variant_order_sha256",
            "affine_mean_sha256",
            "affine_inverse_scale_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        for name in (
            "allele_orientation",
            "allele_coding",
            "centering_source",
            "centering_formula",
            "scaling_formula",
            "missing_imputation",
            "ploidy_policy",
        ):
            _nonempty(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "retained_variant_order_sha256": self.retained_variant_order_sha256,
            "allele_orientation": self.allele_orientation,
            "allele_coding": self.allele_coding,
            "centering_source": self.centering_source,
            "centering_formula": self.centering_formula,
            "scaling_formula": self.scaling_formula,
            "missing_imputation": self.missing_imputation,
            "ploidy_policy": self.ploidy_policy,
            "affine_mean_sha256": self.affine_mean_sha256,
            "affine_inverse_scale_sha256": self.affine_inverse_scale_sha256,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


def require_identical_scale_plans(
    reference: GenotypeScalePlanV1, trait: GenotypeScalePlanV1
) -> str:
    if reference.digest != trait.digest:
        raise ValueError("Reference and trait genotype scale plans differ.")
    return reference.digest


_PHASES = ("source", "action", "grouped", "same_person", "trait")


@dataclass(frozen=True)
class ContextAdmissionLedgerV1:
    """Checked aggregate admission over explicit dimension-derived ledgers."""

    permanent_bytes: int
    phase_bytes: Mapping[str, int]
    retry_bytes: int
    trusted_fallback_bytes: int
    telemetry_bytes: int
    compact_output_bytes: int
    headroom_bytes: int
    semantic_call_counts: Mapping[str, int]
    maximum_events_per_call: int
    phase_checkpoint_events: int
    repair_fallback_events: int
    semantic_verification_events: int
    publication_events: int

    def __post_init__(self) -> None:
        phase = {
            str(key): checked_u64(f"phase_bytes.{key}", value)
            for key, value in self.phase_bytes.items()
        }
        if set(phase) != set(_PHASES):
            raise ValueError(f"phase_bytes must contain exactly {list(_PHASES)}.")
        calls: dict[str, int] = {}
        for key, value in self.semantic_call_counts.items():
            operation = key.value if isinstance(key, SemanticOperation) else str(key)
            if operation in calls:
                raise ValueError(
                    f"semantic_call_counts repeats operation {operation!r}."
                )
            calls[operation] = checked_u64(f"semantic_call_counts.{operation}", value)
        expected_operations = {operation.value for operation in SemanticOperation}
        if set(calls) != expected_operations:
            raise ValueError("semantic_call_counts must enumerate every operation.")
        object.__setattr__(self, "phase_bytes", freeze_context_mapping(phase))
        object.__setattr__(self, "semantic_call_counts", freeze_context_mapping(calls))
        for name in (
            "permanent_bytes",
            "retry_bytes",
            "trusted_fallback_bytes",
            "telemetry_bytes",
            "compact_output_bytes",
            "headroom_bytes",
            "maximum_events_per_call",
            "phase_checkpoint_events",
            "repair_fallback_events",
            "semantic_verification_events",
            "publication_events",
        ):
            checked_u64(name, getattr(self, name))

    @property
    def protected_calls(self) -> int:
        return checked_add_u64("protected_calls", *self.semantic_call_counts.values())

    @property
    def telemetry_events(self) -> int:
        call_events = checked_mul_u64(
            "protected_call_events", self.protected_calls, self.maximum_events_per_call
        )
        return checked_add_u64(
            "telemetry_events",
            call_events,
            self.phase_checkpoint_events,
            self.repair_fallback_events,
            self.semantic_verification_events,
            self.publication_events,
        )

    @property
    def peak_bytes(self) -> int:
        return checked_add_u64(
            "peak_bytes",
            self.permanent_bytes,
            max(self.phase_bytes.values()),
            self.retry_bytes,
            self.trusted_fallback_bytes,
            self.telemetry_bytes,
            self.compact_output_bytes,
            self.headroom_bytes,
        )
