#pragma once

// Stage 0 frozen design contract only.
//
// This header is deliberately not included by the active build and contains no
// executor, binding, or function implementation.  It fixes the types and
// invariants that a later contextual-native implementation must satisfy.  In
// particular, a sealed plan owns every array, string, map, and duplicated
// descriptor; no plan retains a borrowed writable view.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace summit::context_v1 {

inline constexpr std::uint32_t kContextNativeApiVersion = 1;

enum class ArtifactFamily : std::uint32_t {
    contextual_reference = 1,
    contextual_trait = 2,
    contextual_fit = 3,
};

enum class LogicalSchemaVersion : std::uint32_t {
    contextual_reference_v1 = 1,
    contextual_trait_v1 = 2,
    contextual_fit_v1 = 3,
};

enum class GroupedEncodingVersion : std::uint32_t {
    grouped_unnormalized_dense_v1 = 1,
};

enum class NativeBackendKind : std::uint32_t {
    pre_scaled_dense_differential_v1 = 1,
    plink_bed_descriptor_stream_v1 = 2,
};

enum class GenotypeScalePolicy : std::uint32_t {
    pre_scaled_dense_v1 = 1,
    sealed_variant_affine_v1 = 2,
};

enum class GenotypeAffineFormula : std::uint32_t {
    pre_scaled_no_native_transform_v1 = 1,
    dosage_minus_mean_times_inverse_scale_v1 = 2,
};

enum class AnnotationMode : std::uint32_t {
    strict_disjoint_binary_v1 = 1,
    generic_nonnegative_weights_v1 = 2,
};

enum class AnnotationRepresentation : std::uint32_t {
    strict_annotation_id_per_variant_v1 = 1,
    generic_dense_variant_by_annotation_v1 = 2,
};

enum class GroupedAttributionAlgorithm : std::uint32_t {
    direct_grouped_tn_v1 = 1,
    group_restricted_action_v1 = 2,
    auto_qualified_v1 = 3,
};

enum class GroupedTNPolicy : std::uint32_t {
    action_scaled_v1 = 1,
    genotype_scaled_v1 = 2,
    auto_qualified_v1 = 3,
};

enum class NumericPolicy : std::uint32_t {
    fp64_v1 = 1,
};

enum class ProbePolicy : std::uint32_t {
    explicit_rademacher_v1 = 1,
    counter_philox_rademacher_v1 = 2,
};

enum class ProbeStream : std::uint32_t {
    sample_reference_v1 = 1,
    variant_same_person_v1 = 2,
};

enum class DeletionPolicy : std::uint32_t {
    approximate_summary_only_v1 = 1,
};

enum class ReferenceGroupNumeratorUnit : std::uint32_t {
    raw_source_target_annotation_mass_product_fixed_probe_v1 = 1,
};

enum class TraitGroupNumeratorUnit : std::uint32_t {
    raw_source_annotation_mass_v1 = 1,
};

enum class PhenotypeNormalizationPolicy : std::uint32_t {
    project_then_unit_residual_variance_v1 = 1,
};

enum class RetryFallbackPolicy : std::uint32_t {
    checked_retry_then_preallocated_trusted_fp64_v1 = 1,
};

enum class NumaPolicy : std::uint32_t {
    one_process_one_socket_v1 = 1,
    one_process_two_sockets_v1 = 2,
    two_processes_split_global_probe_ranges_v1 = 3,
};

enum class OutputNodePolicy : std::uint32_t {
    local_first_touch_verified_v1 = 1,
};

enum class SemanticOperation : std::uint32_t {
    sample_probe_projection_tn = 1,
    sample_probe_projection_nn = 2,
    source_tn = 3,
    full_target_nn = 4,
    action_projection_tn = 5,
    action_projection_nn = 6,
    action_gram_tn = 7,
    group_target_nn = 8,
    group_cross_gram_tn = 9,
    direct_grouped_tn = 10,
    same_person_target_nn = 11,
    same_person_projection_tn = 12,
    same_person_projection_nn = 13,
    same_person_gram_tn = 14,
    trait_score_tn = 15,
    trait_feature_projection_tn = 16,
    trait_feature_projection_nn = 17,
};

enum class ExecutionPhase : std::uint32_t {
    validation_and_sealing = 1,
    admission = 2,
    source = 3,
    action = 4,
    group = 5,
    same_person = 6,
    residual_moments = 7,
    trait = 8,
    finalization = 9,
    publication = 10,
};

enum class AllocationCategory : std::uint32_t {
    decode = 1,
    sample_probe_panel = 2,
    source_scores = 3,
    targets = 4,
    actions = 5,
    group_targets = 6,
    group_action_tile = 7,
    group_cross = 8,
    direct_group_scaled = 9,
    direct_group_output = 10,
    group_accumulator = 11,
    same_person_sketch = 12,
    same_person_action = 13,
    same_person_persistent = 14,
    trait_features = 15,
    trait_scores = 16,
    compact_outputs = 17,
    projection = 18,
    integrity_retry = 19,
    trusted_fallback = 20,
    telemetry = 21,
    explicit_headroom = 22,
};

enum class TelemetryEventClass : std::uint32_t {
    phase_transition = 1,
    mutation_checkpoint = 2,
    protected_call = 3,
    repair = 4,
    retry = 5,
    trusted_fallback = 6,
    semantic_verification = 7,
    scratch_release = 8,
    publication = 9,
};

enum class ArrayLayout : std::uint32_t {
    column_major_v1 = 1,
    row_major_v1 = 2,
};

// All SHA-256 strings below are lowercase hexadecimal digests over a typed,
// little-endian, shape-aware canonical envelope.  Empty digest fields reject.
struct OwnedF64ArrayV1 {
    std::vector<std::uint64_t> shape;
    ArrayLayout layout;
    std::vector<double> values;
    std::string canonical_sha256;
};

struct OwnedI64ArrayV1 {
    std::vector<std::uint64_t> shape;
    ArrayLayout layout;
    std::vector<std::int64_t> values;
    std::string canonical_sha256;
};

struct OwnedU64ArrayV1 {
    std::vector<std::uint64_t> shape;
    ArrayLayout layout;
    std::vector<std::uint64_t> values;
    std::string canonical_sha256;
};

struct PairEntryV1 {
    std::uint32_t q;
    std::uint32_t r;
    std::uint32_t eta;
};

struct ComponentEntryV1 {
    std::uint32_t annotation_index;
    std::uint32_t pair_index;
};

struct ScientificIndexMapsV1 {
    // Canonical pair order: all (q,q), then lexicographic q<r.
    // Canonical component order: annotation-major, pair-minor.
    std::vector<PairEntryV1> pair_map;
    std::vector<ComponentEntryV1> component_map;
    std::string pair_map_sha256;
    std::string component_map_sha256;
};

struct SchemaIdentityV1 {
    ArtifactFamily artifact_family;
    LogicalSchemaVersion logical_schema_version;
    GroupedEncodingVersion grouped_encoding_version;
    std::uint32_t native_api_version;
    std::string native_backend_version;
    std::string build_id;
};

struct BuildIdentityV1 {
    std::string build_id;
    std::string source_commit;
    bool source_dirty;
    std::string source_tree_sha256;
    std::string native_binary_sha256;
    std::string compiler_id;
    std::string compiler_version;
    std::string compiler_flags;
    std::string isa;
    std::string openmp_runtime;
    std::string blas_identity;
    std::string blas_version;
    std::string blas_linkage;
    std::uint32_t blas_threads;
};

struct FileIdentityV1 {
    std::uint64_t device;
    std::uint64_t inode;
    std::uint64_t byte_count;
    std::uint64_t link_count;
    std::int64_t modification_time_ns;
    std::int64_t change_time_ns;
    std::string canonical_identity_sha256;
};

struct DescriptorInputV1 {
    // The implementation must duplicate fd with close-on-exec and retain the
    // duplicate.  Closing or reusing the caller's fd cannot affect the plan.
    int fd;
    FileIdentityV1 expected_identity;
};

struct PlinkDescriptorSetV1 {
    DescriptorInputV1 bed;
    DescriptorInputV1 bim;
    DescriptorInputV1 fam;
};

struct RetainedSampleMapV1 {
    std::vector<std::uint64_t> fam_row_indices;
    std::string canonical_sha256;
};

struct VariantEntryV1 {
    std::uint64_t bim_row_index;
    std::string variant_id;
    std::string counted_allele;
    std::string other_allele;
};

struct AlleleAwareVariantMapV1 {
    std::vector<VariantEntryV1> variants;
    std::string canonical_sha256;
};

struct GenotypeScalePlanV1 {
    GenotypeScalePolicy policy;
    GenotypeAffineFormula affine_formula;
    // Native-owned vectors are empty only for pre_scaled_dense_v1; otherwise
    // both have length M.  All digest/provenance fields remain mandatory in
    // either mode and bind the upstream transform for pre-scaled input.
    std::vector<double> mean;
    std::vector<double> inverse_scale;
    std::string mean_sha256;
    std::string inverse_scale_sha256;
    std::string allele_orientation_id;
    std::string allele_coding_id;
    std::string centering_cohort_id;
    std::string centering_formula_id;
    std::string scaling_formula_id;
    std::string imputation_policy_id;
    std::string ploidy_policy_id;
    std::string retained_variant_map_sha256;
    std::string scale_plan_sha256;
};

struct GenotypeInputV1 {
    NativeBackendKind backend;
    // Exactly one representation is active.  Descriptor streaming requires
    // sealed_variant_affine_v1; dense differential input requires
    // pre_scaled_dense_v1 and is never a production reader.
    PlinkDescriptorSetV1 descriptors;
    OwnedF64ArrayV1 pre_scaled_dense_n_by_m;
    RetainedSampleMapV1 retained_samples;
    AlleleAwareVariantMapV1 retained_variants;
    GenotypeScalePlanV1 scale;
};

struct FixedEffectProjectionV1 {
    OwnedF64ArrayV1 orthonormal_u_n_by_rank;
    std::uint64_t residual_rank;
    std::string fixed_effect_spec_sha256;
};

struct ContextBasisInputV1 {
    OwnedF64ArrayV1 phi_n_by_q;
    std::string basis_spec_sha256;
    std::string basis_calibration_sha256;
};

struct AnnotationInputV1 {
    AnnotationMode mode;
    AnnotationRepresentation representation;
    std::vector<std::string> names;
    // strict_annotation_ids has length M and values [0,K).  generic_weights is
    // [M,K] column-major, finite, and nonnegative.  The inactive field is empty.
    std::vector<std::uint32_t> strict_annotation_ids;
    OwnedF64ArrayV1 generic_weights_m_by_k;
    std::vector<double> annotation_masses;
    std::string annotation_map_sha256;
};

struct DeletionGroupInputV1 {
    std::vector<std::string> labels;
    std::vector<std::uint32_t> group_index_per_variant;
    std::vector<std::uint64_t> group_variant_counts;
    OwnedF64ArrayV1 group_annotation_masses_j_by_k;
    std::string group_map_sha256;
};

struct AnnotationOutputIdentityV1 {
    AnnotationMode mode;
    std::vector<std::string> names;
    std::string annotation_map_sha256;
};

struct DeletionGroupOutputIdentityV1 {
    std::vector<std::string> labels;
    std::string group_map_sha256;
    std::string group_assignment_sha256;
};

struct ProbeInputV1 {
    ProbePolicy policy;
    ProbeStream stream;
    std::uint64_t total_probe_count;
    // Explicit sample probes are [N,B_T]; explicit variant probes are [M,B_D].
    // Empty for the counter policy.
    OwnedF64ArrayV1 explicit_rademacher;
    std::uint64_t counter_key_hi;
    std::uint64_t counter_key_lo;
    std::uint32_t counter_mapping_version;
    std::string probe_identity_sha256;
};

struct TilePlanV1 {
    std::uint64_t variant_block;
    std::uint64_t sample_probe_resident;
    std::uint64_t sample_probe_tile;
    std::uint64_t variant_probe_tile;
    std::uint64_t action_tile;
    std::uint64_t annotation_tile;
    std::uint64_t context_tile;
    std::uint64_t group_tile;
    std::uint64_t trait_feature_tile;
};

struct RuntimePolicyV1 {
    std::uint32_t decode_threads;
    std::uint32_t blas_threads;
    NumaPolicy numa_policy;
    OutputNodePolicy output_node_policy;
    std::vector<std::uint32_t> expected_cpu_ids;
    std::vector<std::uint32_t> expected_numa_nodes;
    std::uint64_t workspace_cap_bytes;
    std::uint64_t safety_margin_bytes;
};

struct IntegrityPolicyV1 {
    RetryFallbackPolicy retry_fallback_policy;
    std::uint32_t maximum_attempts_per_call;
    std::uint64_t telemetry_capacity;
    std::string trusted_backend_id;
    std::uint32_t trusted_backend_threads;
};

struct ContextDimensionsV1 {
    std::uint64_t n;
    std::uint64_t m;
    std::uint32_t q;
    std::uint32_t k;
    std::uint32_t pair_count;
    std::uint32_t component_count;
    std::uint32_t group_count;
};

struct CommonPlanInputsV1 {
    ContextDimensionsV1 dimensions;
    SchemaIdentityV1 schema;
    GenotypeInputV1 genotype;
    FixedEffectProjectionV1 projection;
    ContextBasisInputV1 context_basis;
    AnnotationInputV1 annotations;
    DeletionGroupInputV1 deletion_groups;
    ScientificIndexMapsV1 maps;
    NumericPolicy numeric_policy;
    TilePlanV1 tiles;
    RuntimePolicyV1 runtime;
    IntegrityPolicyV1 integrity;
};

struct ReferencePlanInputsV1 {
    CommonPlanInputsV1 common;
    ProbeInputV1 sample_probes;
    ProbeInputV1 variant_probes;
    GroupedAttributionAlgorithm grouped_algorithm;
    GroupedTNPolicy direct_grouped_tn_policy;
};

struct TraitPlanInputsV1 {
    CommonPlanInputsV1 common;
    OwnedF64ArrayV1 phenotype_n_by_l;
    std::vector<std::string> phenotype_names;
    OwnedF64ArrayV1 residual_basis_n_by_h;
    std::vector<std::string> residual_component_names;
    PhenotypeNormalizationPolicy phenotype_normalization;
    std::string compatible_reference_identity_sha256;
};

struct MemoryLedgerEntryV1 {
    AllocationCategory category;
    ExecutionPhase first_live_phase;
    ExecutionPhase last_live_phase;
    std::uint64_t element_count;
    std::uint64_t byte_count;
    std::uint64_t concurrent_copies;
};

struct SemanticCallLedgerEntryV1 {
    ExecutionPhase phase;
    SemanticOperation operation;
    std::uint64_t protected_call_count;
    std::uint64_t maximum_attempt_count;
};

struct EventLedgerEntryV1 {
    TelemetryEventClass event_class;
    std::uint64_t base_event_count;
    std::uint64_t worst_case_event_count;
};

struct PhaseLedgerEntryV1 {
    ExecutionPhase phase;
    std::uint64_t logical_descriptor_passes;
    std::uint64_t decoded_blocks;
    std::uint64_t variant_record_visits;
    std::uint64_t expected_bytes_read;
};

struct PreflightEstimateV1 {
    // Every product/sum is evaluated with unsigned 128-bit intermediates and
    // must fit uint64_t before it appears here.  Overflow is a validation error.
    bool checked_arithmetic_complete;
    bool admitted;
    std::uint64_t peak_bytes;
    std::uint64_t retry_reserve_bytes;
    std::uint64_t trusted_fallback_reserve_bytes;
    std::uint64_t telemetry_bytes;
    std::uint64_t explicit_headroom_bytes;
    std::uint64_t total_descriptor_passes;
    std::uint64_t total_decoded_blocks;
    std::uint64_t total_protected_calls;
    std::uint64_t worst_case_protected_attempts;
    std::uint64_t base_telemetry_events;
    std::uint64_t worst_case_telemetry_events;
    std::uint64_t admitted_telemetry_capacity;
    std::vector<MemoryLedgerEntryV1> memory_ledger;
    std::vector<SemanticCallLedgerEntryV1> semantic_call_ledger;
    std::vector<EventLedgerEntryV1> event_ledger;
    std::vector<PhaseLedgerEntryV1> phase_ledger;
    GroupedAttributionAlgorithm selected_grouped_algorithm;
    GroupedTNPolicy selected_direct_grouped_tn_policy;
    TilePlanV1 selected_tiles;
    std::string selected_plan_json;
    std::string selected_plan_sha256;
    std::string admission_ledger_sha256;
};

struct ApproximateDeletionCommonV1 {
    DeletionPolicy policy;
    bool grouped_values_are_unnormalized_numerators;
    bool multiple_group_subtraction_supported;
    std::string retained_mass_formula_id;
};

struct ReferenceDeletionMetadataV1 {
    ApproximateDeletionCommonV1 common;
    ReferenceGroupNumeratorUnit group_numerator_unit;
    bool reuse_full_reference_same_person_for_every_deletion;
    bool exact_deleted_reference_kernels;
};

struct TraitDeletionMetadataV1 {
    ApproximateDeletionCommonV1 common;
    TraitGroupNumeratorUnit group_numerator_unit;
    bool keep_residual_only_moments_fixed;
};

struct ExecutionEvidenceV1 {
    std::uint64_t observed_descriptor_passes;
    std::uint64_t observed_decoded_blocks;
    std::uint64_t observed_protected_calls;
    std::uint64_t observed_telemetry_events;
    std::uint64_t repair_count;
    std::uint64_t retry_count;
    std::uint64_t trusted_fallback_count;
    std::uint64_t measured_peak_rss_bytes;
    bool semantic_call_ledger_exact;
    bool phase_ledger_exact;
    bool telemetry_complete_without_drop;
    bool scratch_released_before_publication;
    bool inputs_unchanged_at_all_checkpoints;
    std::string canonical_telemetry_json;
    std::string telemetry_sha256;
};

struct ResultIdentityV1 {
    SchemaIdentityV1 schema;
    BuildIdentityV1 build;
    ScientificIndexMapsV1 maps;
    AnnotationOutputIdentityV1 annotations;
    DeletionGroupOutputIdentityV1 deletion_groups;
    std::string genotype_scale_plan_sha256;
    std::string retained_sample_map_sha256;
    std::string retained_variant_map_sha256;
    std::string fixed_effect_spec_sha256;
    std::string basis_spec_sha256;
    std::string basis_calibration_sha256;
    std::string sealed_plan_sha256;
    std::string output_arrays_sha256;
    std::string manifest_sha256;
};

struct ContextualReferenceResultV1 {
    // Compact arrays only.  No returned array may have a retained-sample (N)
    // or retained-variant (M) axis.
    std::uint64_t reference_n;
    OwnedF64ArrayV1 t_r_c_by_c;
    OwnedF64ArrayV1 d_r_c_by_c;
    OwnedF64ArrayV1 group_gram_unnormalized_num_j_by_c_by_c;
    OwnedF64ArrayV1 annotation_masses_k;
    OwnedF64ArrayV1 group_annotation_masses_j_by_k;
    OwnedU64ArrayV1 group_variant_counts_j;
    double gram_pre_symmetry_max_abs;
    double same_person_pre_symmetry_max_abs;
    double group_reconstruction_max_abs;
    ReferenceDeletionMetadataV1 deletion;
    ResultIdentityV1 identity;
    ExecutionEvidenceV1 execution;
    std::string canonical_manifest_json;
};

struct ContextualTraitResultV1 {
    // Layouts include declared trait/residual maps in the manifest; they are
    // never inferred from shape.
    OwnedF64ArrayV1 genetic_rhs;
    OwnedF64ArrayV1 genetic_traces_c;
    OwnedF64ArrayV1 genetic_residual_c_by_h;
    OwnedF64ArrayV1 residual_rhs;
    OwnedF64ArrayV1 residual_traces_h;
    OwnedF64ArrayV1 residual_gram_h_by_h;
    OwnedF64ArrayV1 group_rhs_unnormalized_num;
    OwnedF64ArrayV1 group_trace_unnormalized_num_j_by_c;
    OwnedF64ArrayV1 group_genetic_residual_num_j_by_c_by_h;
    OwnedF64ArrayV1 annotation_masses_k;
    OwnedF64ArrayV1 group_annotation_masses_j_by_k;
    OwnedU64ArrayV1 group_variant_counts_j;
    TraitDeletionMetadataV1 deletion;
    ResultIdentityV1 identity;
    ExecutionEvidenceV1 execution;
    std::string canonical_manifest_json;
};

class ContextualReferencePlanV1 final {
public:
    ContextualReferencePlanV1(ContextualReferencePlanV1&&) noexcept;
    ContextualReferencePlanV1& operator=(ContextualReferencePlanV1&&) noexcept;
    ContextualReferencePlanV1(const ContextualReferencePlanV1&) = delete;
    ContextualReferencePlanV1& operator=(const ContextualReferencePlanV1&) = delete;
    ~ContextualReferencePlanV1();

private:
    struct Impl;
    explicit ContextualReferencePlanV1(std::unique_ptr<const Impl>);
    std::unique_ptr<const Impl> impl_;

    friend ContextualReferencePlanV1 seal_reference_plan(ReferencePlanInputsV1);
    friend PreflightEstimateV1 preflight_reference(const ContextualReferencePlanV1&);
    friend ContextualReferenceResultV1 run_reference(ContextualReferencePlanV1&&);
};

class ContextualTraitPlanV1 final {
public:
    ContextualTraitPlanV1(ContextualTraitPlanV1&&) noexcept;
    ContextualTraitPlanV1& operator=(ContextualTraitPlanV1&&) noexcept;
    ContextualTraitPlanV1(const ContextualTraitPlanV1&) = delete;
    ContextualTraitPlanV1& operator=(const ContextualTraitPlanV1&) = delete;
    ~ContextualTraitPlanV1();

private:
    struct Impl;
    explicit ContextualTraitPlanV1(std::unique_ptr<const Impl>);
    std::unique_ptr<const Impl> impl_;

    friend ContextualTraitPlanV1 seal_trait_plan(TraitPlanInputsV1);
    friend PreflightEstimateV1 preflight_trait(const ContextualTraitPlanV1&);
    friend ContextualTraitResultV1 run_trait(ContextualTraitPlanV1&&);
};

// seal_* defensively owns/canonicalizes inputs, duplicates descriptors, verifies
// maps/digests, performs checked admission, and freezes the selected plan.
// run_* consumes that noncopyable plan and is one-shot.  It may publish only if
// observed phase/call/event counts equal preflight and all evidence is complete.
ContextualReferencePlanV1 seal_reference_plan(ReferencePlanInputsV1 inputs);
PreflightEstimateV1 preflight_reference(const ContextualReferencePlanV1& plan);
ContextualReferenceResultV1 run_reference(ContextualReferencePlanV1&& plan);

ContextualTraitPlanV1 seal_trait_plan(TraitPlanInputsV1 inputs);
PreflightEstimateV1 preflight_trait(const ContextualTraitPlanV1& plan);
ContextualTraitResultV1 run_trait(ContextualTraitPlanV1&& plan);

}  // namespace summit::context_v1
