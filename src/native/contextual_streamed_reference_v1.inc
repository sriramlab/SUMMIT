// Contextual native API V1: descriptor-owned Stage 2/3 reference and Stage 4
// trait-summary core.
//
// ESTIMATOR IDENTITY: the reference path in this file uses sample-axis probes
// and kernel actions to estimate an aggregate contextual Gram. It is separate
// from the generalized variant-axis, two-pass, per-variant GxE LD-score
// estimator. Do not reuse grouped-action numerators as that estimator's
// fixed-full-genome-LD-score SNP-block jackknife.
//
// This file is included after gxeldcore.cpp's anonymous namespace.  It may use
// the established TU-private descriptor, SHA-256, Philox, thread-admission,
// and deterministic GEMM substrate, but it has no dependency on legacy X/W
// science or on DirectContext/MultiEnvironment*.

#include <condition_variable>
#include <numeric>
#include <tuple>
#include <unordered_set>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace summit::context_v1 {

namespace stage2_detail {

constexpr const char* kBackendV1 =
    "plink_bed_descriptor_stream_stage2_v1";
constexpr const char* kTraitBackendV1 =
    "plink_bed_descriptor_stream_trait_v1";
constexpr const char* kExecutionBackendV1 =
    "deterministic_tiled_fp64_with_scalar_witness_v1";
constexpr std::uint64_t kContextualBackendVersionV1 = 2;
constexpr std::size_t kProtectedPrefixCanariesV1 = 8;
constexpr std::size_t kProtectedSuffixCanariesV1 = 8;
constexpr const char* kStage3ResultKindV1 =
    "stage3_complete_reference_statistics_v1";
constexpr const char* kGroupedEncodingV1 =
    "grouped_unnormalized_dense_v1";
constexpr const char* kDeletionPolicyV1 =
    "approximate_summary_only_v1";
constexpr const char* kScalePolicyV1 = "sealed_variant_affine_v1";
constexpr const char* kAlleleCodingV1 =
    "plink_bed_snp_major_diploid_hardcall_v1";
constexpr const char* kCenteringFormulaV1 =
    "provided_variant_affine_mean_v1";
constexpr const char* kScalingFormulaV1 =
    "dosage_minus_mean_times_inverse_scale_v1";
constexpr const char* kMissingImputationV1 = "sealed_mean_v1";
constexpr const char* kPloidyPolicyV1 = "diploid_v1";
constexpr double kCanaryV1 = -9.876543210123456e287;

enum class EventClassV1 : std::uint8_t {
    phase_transition = 0,
    mutation_checkpoint,
    protected_call,
    fault_injection,
    fault_detection,
    repair,
    retry,
    trusted_fallback,
    semantic_verification,
    scratch_release,
    publication,
    terminal_failure,
};

enum class SemanticRoleV1 : std::uint8_t {
    ordinary = 0,
    tile,
    global_merge,
};

enum class SemanticPlacementV1 : std::uint8_t {
    none = 0,
    action_scaled,
    genotype_scaled,
};

enum class VariantMembershipV1 : std::uint8_t {
    logical_half_open_range = 0,
    strict_logical_range,
    group_execution_slice,
    strict_group_execution_slice,
};

enum class SemanticPhaseV1 : std::uint8_t {
    unspecified = 0,
    source,
    action,
    gram,
    group,
    same_person,
    residual_moments,
    trait,
    finalization,
    publication,
};

struct SemanticCoordinateV1 {
    ContextualDenseSemanticV1 operation = ContextualDenseSemanticV1::count;
    std::uint64_t resident_begin = 0;
    std::uint64_t resident_end = 0;
    std::uint64_t probe_begin = 0;
    std::uint64_t probe_end = 0;
    std::uint64_t variant_begin = 0;
    std::uint64_t variant_end = 0;
    std::uint64_t annotation_begin = 0;
    std::uint64_t annotation_end = 0;
    std::uint64_t context_begin = 0;
    std::uint64_t context_end = 0;
    std::uint64_t action_begin = 0;
    std::uint64_t action_end = 0;
    std::uint64_t group_begin = 0;
    std::uint64_t group_end = 0;
    std::uint64_t canonical_begin = 0;
    std::uint64_t canonical_end = 0;
    VariantMembershipV1 variant_membership =
        VariantMembershipV1::logical_half_open_range;
    std::uint64_t variant_membership_source_begin = 0;
    std::uint64_t variant_membership_source_end = 0;
    std::uint64_t variant_membership_count = 0;
    std::array<char, 65> variant_membership_sha256{};
    bool variant_range_is_exact = true;
    SemanticPhaseV1 phase = SemanticPhaseV1::unspecified;
    SemanticRoleV1 role = SemanticRoleV1::ordinary;
    SemanticPlacementV1 placement = SemanticPlacementV1::none;
};

struct EventEvidenceV1 {
    std::uint64_t elapsed_ns = 0;
    std::uint64_t operand_fingerprint = 0;
    std::uint64_t witness_fingerprint = 0;
    std::uint64_t accepted_output_fingerprint = 0;
    std::uint64_t runtime_before_fingerprint = 0;
    std::uint64_t runtime_after_fingerprint = 0;
    bool prefix_canary_verified = false;
    bool suffix_canary_verified = false;
    bool finiteness_verified = false;
    bool serialized_entry_verified = false;
    bool deterministic_non_vendor_backend = false;
};

struct EventV1 {
    EventClassV1 event_class = EventClassV1::protected_call;
    const char* phase = "";
    SemanticCoordinateV1 coordinate{};
    const char* resolution = "";
    std::uint32_t attempt = 0;
    int rows = 0;
    int columns = 0;
    int reduction = 0;
    int left_stride = 0;
    int right_stride = 0;
    int output_stride = 0;
    bool transpose_left = false;
    std::uint64_t sequence = 0;
    std::int64_t process_id = 0;
    std::int64_t thread_id = 0;
    std::uint64_t elapsed_ns = 0;
    std::uint64_t operand_fingerprint = 0;
    std::uint64_t witness_fingerprint = 0;
    std::uint64_t accepted_output_fingerprint = 0;
    std::uint64_t runtime_before_fingerprint = 0;
    std::uint64_t runtime_after_fingerprint = 0;
    bool prefix_canary_verified = false;
    bool suffix_canary_verified = false;
    bool finiteness_verified = false;
    bool serialized_entry_verified = false;
    bool deterministic_non_vendor_backend = false;
};

struct BimEntryV1 {
    std::string variant_id;
    std::string allele_1;
    std::string allele_2;
};

struct PreflightV1 {
    std::uint64_t permanent_bytes = 0;
    std::uint64_t source_phase_bytes = 0;
    std::uint64_t action_phase_bytes = 0;
    std::uint64_t group_phase_bytes = 0;
    std::uint64_t same_person_phase_bytes = 0;
    std::uint64_t compact_output_bytes = 0;
    std::uint64_t integrity_reserve_bytes = 0;
    std::uint64_t telemetry_bytes = 0;
    std::uint64_t required_workspace_bytes = 0;
    std::uint64_t required_telemetry_capacity = 0;
    std::uint64_t maximum_protected_output_elements = 0;
    std::uint64_t resident_batches = 0;
    std::uint64_t source_descriptor_passes = 0;
    std::uint64_t action_descriptor_passes = 0;
    std::uint64_t source_decoded_blocks = 0;
    std::uint64_t action_decoded_blocks = 0;
    std::uint64_t group_descriptor_passes = 0;
    std::uint64_t group_decoded_blocks = 0;
    std::uint64_t same_person_descriptor_passes = 0;
    std::uint64_t same_person_decoded_blocks = 0;
    std::uint64_t residual_phase_bytes = 0;
    std::uint64_t trait_phase_bytes = 0;
    std::uint64_t trait_descriptor_passes = 0;
    std::uint64_t trait_decoded_blocks = 0;
    std::uint64_t total_protected_calls = 0;
    std::array<std::uint64_t, kContextualDenseSemanticCountV1> calls{};
};

enum class PerformancePhaseV1 : std::uint8_t {
    source = 0,
    action,
    gram,
    group,
    same_person,
    residual_moments,
    trait,
    finalization,
    publication,
    run_total,
    count,
};

constexpr std::array<const char*, 10> kPerformancePhaseNamesV1{{
    "source", "action", "gram", "group", "same_person",
    "residual_moments", "trait", "finalization", "publication",
    "run_total",
}};

enum class PerformanceCategoryV1 : std::uint8_t {
    decode = 0,
    projection,
    row_scale,
    packing,
    reduction,
    count,
};

constexpr std::array<const char*, 5> kPerformanceCategoryNamesV1{{
    "decode", "projection", "row_scale", "packing", "reduction",
}};

std::uint64_t saturating_add_u64(
    std::uint64_t left, std::uint64_t right
) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
        ? std::numeric_limits<std::uint64_t>::max() : left + right;
}

std::uint64_t nonnegative_delta_u64(
    std::uint64_t end, std::uint64_t begin
) noexcept {
    return end >= begin ? end - begin : 0;
}

std::uint64_t elapsed_ns_since(
    const std::chrono::steady_clock::time_point& begin
) noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin
    ).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

struct ProcessResourceSnapshotV1 {
    bool process_cpu_available = false;
    bool proc_status_available = false;
    bool rusage_available = false;
    bool proc_io_available = false;
    std::uint64_t process_cpu_ns = 0;
    std::uint64_t rss_bytes = 0;
    std::uint64_t rss_high_water_bytes = 0;
    std::uint64_t minor_page_faults = 0;
    std::uint64_t major_page_faults = 0;
    std::uint64_t input_block_operations = 0;
    std::uint64_t output_block_operations = 0;
    std::uint64_t io_read_characters = 0;
    std::uint64_t io_write_characters = 0;
    std::uint64_t io_read_syscalls = 0;
    std::uint64_t io_write_syscalls = 0;
    std::uint64_t io_read_bytes = 0;
    std::uint64_t io_write_bytes = 0;
};

ProcessResourceSnapshotV1 capture_process_resources() noexcept {
    ProcessResourceSnapshotV1 result;
#if defined(CLOCK_PROCESS_CPUTIME_ID)
    timespec process_time{};
    if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_time) == 0 &&
        process_time.tv_sec >= 0 && process_time.tv_nsec >= 0) {
        result.process_cpu_available = true;
        result.process_cpu_ns = saturating_add_u64(
            static_cast<std::uint64_t>(process_time.tv_sec) *
                UINT64_C(1000000000),
            static_cast<std::uint64_t>(process_time.tv_nsec)
        );
    }
#endif
#if defined(__linux__)
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) == 0) {
        result.rusage_available = true;
        result.minor_page_faults = usage.ru_minflt > 0
            ? static_cast<std::uint64_t>(usage.ru_minflt) : 0;
        result.major_page_faults = usage.ru_majflt > 0
            ? static_cast<std::uint64_t>(usage.ru_majflt) : 0;
        result.input_block_operations = usage.ru_inblock > 0
            ? static_cast<std::uint64_t>(usage.ru_inblock) : 0;
        result.output_block_operations = usage.ru_oublock > 0
            ? static_cast<std::uint64_t>(usage.ru_oublock) : 0;
    }
    try {
        std::ifstream status("/proc/self/status");
        std::string line;
        bool saw_rss = false, saw_hwm = false;
        while (std::getline(status, line)) {
            const bool is_rss = line.compare(0, 6, "VmRSS:") == 0;
            const bool is_hwm = line.compare(0, 6, "VmHWM:") == 0;
            if (!is_rss && !is_hwm) continue;
            std::istringstream value_stream(line.substr(6));
            std::uint64_t value = 0;
            std::string unit;
            if (!(value_stream >> value >> unit) || unit != "kB") continue;
            if (is_rss) {
                result.rss_bytes = saturating_add_u64(
                    0, value > std::numeric_limits<std::uint64_t>::max() /
                            UINT64_C(1024)
                        ? std::numeric_limits<std::uint64_t>::max()
                        : value * UINT64_C(1024)
                );
                saw_rss = true;
            } else {
                result.rss_high_water_bytes = value >
                        std::numeric_limits<std::uint64_t>::max() / UINT64_C(1024)
                    ? std::numeric_limits<std::uint64_t>::max()
                    : value * UINT64_C(1024);
                saw_hwm = true;
            }
        }
        result.proc_status_available = saw_rss && saw_hwm;
    } catch (...) {
    }
    try {
        std::ifstream io("/proc/self/io");
        std::string key;
        std::uint64_t value = 0;
        std::uint32_t fields = 0;
        while (io >> key >> value) {
            if (key == "rchar:") {
                result.io_read_characters = value;
                fields |= 1U << 0U;
            } else if (key == "wchar:") {
                result.io_write_characters = value;
                fields |= 1U << 1U;
            } else if (key == "syscr:") {
                result.io_read_syscalls = value;
                fields |= 1U << 2U;
            } else if (key == "syscw:") {
                result.io_write_syscalls = value;
                fields |= 1U << 3U;
            } else if (key == "read_bytes:") {
                result.io_read_bytes = value;
                fields |= 1U << 4U;
            } else if (key == "write_bytes:") {
                result.io_write_bytes = value;
                fields |= 1U << 5U;
            }
        }
        result.proc_io_available = fields == ((1U << 6U) - 1U);
    } catch (...) {
    }
#endif
    return result;
}

std::uint64_t process_cpu_now_ns(bool& available) noexcept {
    available = false;
#if defined(CLOCK_PROCESS_CPUTIME_ID)
    timespec process_time{};
    if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_time) == 0 &&
        process_time.tv_sec >= 0 && process_time.tv_nsec >= 0) {
        available = true;
        return saturating_add_u64(
            static_cast<std::uint64_t>(process_time.tv_sec) *
                UINT64_C(1000000000),
            static_cast<std::uint64_t>(process_time.tv_nsec)
        );
    }
#endif
    return 0;
}

struct PerformancePhaseRecordV1 {
    std::uint64_t calls = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t process_cpu_ns = 0;
    std::uint64_t rss_begin_bytes = 0;
    std::uint64_t rss_end_bytes = 0;
    std::uint64_t rss_high_water_begin_bytes = 0;
    std::uint64_t rss_high_water_end_bytes = 0;
    std::uint64_t minor_page_faults = 0;
    std::uint64_t major_page_faults = 0;
    std::uint64_t input_block_operations = 0;
    std::uint64_t output_block_operations = 0;
    std::uint64_t io_read_characters = 0;
    std::uint64_t io_write_characters = 0;
    std::uint64_t io_read_syscalls = 0;
    std::uint64_t io_write_syscalls = 0;
    std::uint64_t io_read_bytes = 0;
    std::uint64_t io_write_bytes = 0;
    bool process_cpu_available = false;
    bool proc_status_available = false;
    bool rusage_available = false;
    bool proc_io_available = false;
};

struct PerformanceCategoryRecordV1 {
    std::uint64_t calls = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t process_cpu_ns = 0;
    std::uint64_t work_items = 0;
    std::uint64_t logical_bytes = 0;
};

struct PerformanceOperationRecordV1 {
    std::uint64_t calls = 0;
    double logical_flop_count = 0.0;
    std::uint64_t output_elements = 0;
    std::uint64_t primary_wall_ns = 0;
    std::uint64_t witness_wall_ns = 0;
    std::uint64_t retry_calls = 0;
    std::uint64_t fallback_calls = 0;
    std::uint64_t recovery_wall_ns = 0;
    std::uint64_t protected_total_wall_ns = 0;
    std::uint64_t minimum_rows = 0;
    std::uint64_t maximum_rows = 0;
    std::uint64_t minimum_columns = 0;
    std::uint64_t maximum_columns = 0;
    std::uint64_t minimum_reduction = 0;
    std::uint64_t maximum_reduction = 0;
    bool transpose_left = false;
};

class PerformancePhaseScopeV1 final {
public:
    explicit PerformancePhaseScopeV1(PerformancePhaseRecordV1& record) noexcept
        : record_(&record), wall_begin_(std::chrono::steady_clock::now()),
          begin_(capture_process_resources()) {}
    PerformancePhaseScopeV1(const PerformancePhaseScopeV1&) = delete;
    PerformancePhaseScopeV1& operator=(const PerformancePhaseScopeV1&) = delete;
    ~PerformancePhaseScopeV1() noexcept { finish(); }

    void finish() noexcept {
        if (record_ == nullptr) return;
        const ProcessResourceSnapshotV1 end = capture_process_resources();
        PerformancePhaseRecordV1& record = *record_;
        record.calls = saturating_add_u64(record.calls, 1);
        record.wall_ns = saturating_add_u64(
            record.wall_ns, elapsed_ns_since(wall_begin_)
        );
        record.process_cpu_ns = saturating_add_u64(
            record.process_cpu_ns,
            begin_.process_cpu_available && end.process_cpu_available
                ? nonnegative_delta_u64(
                      end.process_cpu_ns, begin_.process_cpu_ns
                  ) : 0
        );
        if (record.calls == 1) {
            record.rss_begin_bytes = begin_.rss_bytes;
            record.rss_high_water_begin_bytes =
                begin_.rss_high_water_bytes;
        }
        record.rss_end_bytes = end.rss_bytes;
        record.rss_high_water_end_bytes = end.rss_high_water_bytes;
        record.minor_page_faults = saturating_add_u64(
            record.minor_page_faults,
            begin_.rusage_available && end.rusage_available
                ? nonnegative_delta_u64(
                      end.minor_page_faults, begin_.minor_page_faults
                  ) : 0
        );
        record.major_page_faults = saturating_add_u64(
            record.major_page_faults,
            begin_.rusage_available && end.rusage_available
                ? nonnegative_delta_u64(
                      end.major_page_faults, begin_.major_page_faults
                  ) : 0
        );
        record.input_block_operations = saturating_add_u64(
            record.input_block_operations,
            begin_.rusage_available && end.rusage_available
                ? nonnegative_delta_u64(
                      end.input_block_operations,
                      begin_.input_block_operations
                  ) : 0
        );
        record.output_block_operations = saturating_add_u64(
            record.output_block_operations,
            begin_.rusage_available && end.rusage_available
                ? nonnegative_delta_u64(
                      end.output_block_operations,
                      begin_.output_block_operations
                  ) : 0
        );
        record.io_read_characters = saturating_add_u64(
            record.io_read_characters,
            begin_.proc_io_available && end.proc_io_available
                ? nonnegative_delta_u64(
                      end.io_read_characters, begin_.io_read_characters
                  ) : 0
        );
        record.io_write_characters = saturating_add_u64(
            record.io_write_characters,
            begin_.proc_io_available && end.proc_io_available
                ? nonnegative_delta_u64(
                      end.io_write_characters, begin_.io_write_characters
                  ) : 0
        );
        record.io_read_syscalls = saturating_add_u64(
            record.io_read_syscalls,
            begin_.proc_io_available && end.proc_io_available
                ? nonnegative_delta_u64(
                      end.io_read_syscalls, begin_.io_read_syscalls
                  ) : 0
        );
        record.io_write_syscalls = saturating_add_u64(
            record.io_write_syscalls,
            begin_.proc_io_available && end.proc_io_available
                ? nonnegative_delta_u64(
                      end.io_write_syscalls, begin_.io_write_syscalls
                  ) : 0
        );
        record.io_read_bytes = saturating_add_u64(
            record.io_read_bytes,
            begin_.proc_io_available && end.proc_io_available
                ? nonnegative_delta_u64(
                      end.io_read_bytes, begin_.io_read_bytes
                  ) : 0
        );
        record.io_write_bytes = saturating_add_u64(
            record.io_write_bytes,
            begin_.proc_io_available && end.proc_io_available
                ? nonnegative_delta_u64(
                      end.io_write_bytes, begin_.io_write_bytes
                  ) : 0
        );
        record.process_cpu_available = begin_.process_cpu_available &&
            end.process_cpu_available;
        record.proc_status_available = begin_.proc_status_available &&
            end.proc_status_available;
        record.rusage_available = begin_.rusage_available &&
            end.rusage_available;
        record.proc_io_available = begin_.proc_io_available &&
            end.proc_io_available;
        record_ = nullptr;
    }

private:
    PerformancePhaseRecordV1* record_ = nullptr;
    std::chrono::steady_clock::time_point wall_begin_{};
    ProcessResourceSnapshotV1 begin_{};
};

class PerformanceCategoryScopeV1 final {
public:
    PerformanceCategoryScopeV1(
        PerformanceCategoryRecordV1& record,
        std::uint64_t work_items = 0,
        std::uint64_t logical_bytes = 0
    ) noexcept : record_(&record),
                 wall_begin_(std::chrono::steady_clock::now()),
                 cpu_begin_ns_(process_cpu_now_ns(cpu_available_)) {
        record.work_items = saturating_add_u64(record.work_items, work_items);
        record.logical_bytes = saturating_add_u64(
            record.logical_bytes, logical_bytes
        );
    }
    PerformanceCategoryScopeV1(const PerformanceCategoryScopeV1&) = delete;
    PerformanceCategoryScopeV1& operator=(
        const PerformanceCategoryScopeV1&
    ) = delete;
    ~PerformanceCategoryScopeV1() noexcept { finish(); }

    void finish() noexcept {
        if (record_ == nullptr) return;
        bool end_cpu_available = false;
        const std::uint64_t end_cpu_ns = process_cpu_now_ns(end_cpu_available);
        record_->calls = saturating_add_u64(record_->calls, 1);
        record_->wall_ns = saturating_add_u64(
            record_->wall_ns, elapsed_ns_since(wall_begin_)
        );
        record_->process_cpu_ns = saturating_add_u64(
            record_->process_cpu_ns,
            cpu_available_ && end_cpu_available
                ? nonnegative_delta_u64(end_cpu_ns, cpu_begin_ns_) : 0
        );
        record_ = nullptr;
    }

private:
    PerformanceCategoryRecordV1* record_ = nullptr;
    std::chrono::steady_clock::time_point wall_begin_{};
    bool cpu_available_ = false;
    std::uint64_t cpu_begin_ns_ = 0;
};

std::uint64_t checked_add(
    std::uint64_t left, std::uint64_t right, const char* label
) {
    return contextual_add_u64_v1(left, right, label);
}

std::uint64_t checked_mul(
    std::uint64_t left, std::uint64_t right, const char* label
) {
    return contextual_mul_u64_v1(left, right, label);
}

std::uint64_t bytes_for(
    std::initializer_list<std::uint64_t> dimensions,
    std::uint64_t item_bytes,
    const char* label
) {
    std::uint64_t elements = 1;
    for (std::uint64_t dimension : dimensions)
        elements = checked_mul(elements, dimension, label);
    return checked_mul(elements, item_bytes, label);
}

std::size_t checked_size(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error(
            std::string("ContextualReferenceExecutorV1 overflow: ") + label
        );
    return static_cast<std::size_t>(value);
}

std::size_t elements_for(
    std::initializer_list<std::uint64_t> dimensions, const char* label
) {
    return checked_size(bytes_for(dimensions, 1, label), label);
}

std::size_t resolve_tile(
    std::uint64_t requested, std::size_t extent, const char* label
) {
    if (extent == 0)
        throw std::runtime_error(
            std::string("ContextualReferenceExecutorV1 empty extent: ") + label
        );
    const std::uint64_t resolved = requested == 0
        ? static_cast<std::uint64_t>(extent)
        : std::min(requested, static_cast<std::uint64_t>(extent));
    return checked_size(resolved, label);
}

bool is_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f')))
            return false;
    }
    return true;
}

void sha_update(Sha256& digest, const void* pointer, std::size_t bytes) {
    digest.update(static_cast<const unsigned char*>(pointer), bytes);
}

void sha_update_string(Sha256& digest, const char* value) {
    sha_update(digest, value, std::strlen(value));
}

void sha_update_i64_le(Sha256& digest, std::int64_t value) {
    std::array<unsigned char, 8> encoded{};
    const std::uint64_t bits = static_cast<std::uint64_t>(value);
    for (std::size_t index = 0; index < encoded.size(); ++index)
        encoded[index] = static_cast<unsigned char>(bits >> (8U * index));
    sha_update(digest, encoded.data(), encoded.size());
}

void sha_update_u64_le(Sha256& digest, std::uint64_t value) {
    std::array<unsigned char, 8> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
        encoded[index] = static_cast<unsigned char>(value >> (8U * index));
    sha_update(digest, encoded.data(), encoded.size());
}

std::string array_sha256_i64(const std::vector<std::int64_t>& values) {
    Sha256 digest;
    sha_update_string(digest, "<i8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(values.size()));
    for (std::int64_t value : values) sha_update_i64_le(digest, value);
    return digest.finish();
}

std::string array_sha256_i64_c_order(
    const std::vector<std::int64_t>& values,
    std::initializer_list<std::size_t> dimensions
) {
    std::uint64_t expected = 1;
    for (std::size_t dimension : dimensions)
        expected = checked_mul(expected, dimension,
                               "integer C-order digest elements");
    if (values.size() != checked_size(
            expected, "integer C-order digest elements"))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 integer digest shape mismatch"
        );
    Sha256 digest;
    sha_update_string(digest, "<i8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    for (std::size_t dimension : dimensions)
        sha_update_i64_le(digest, static_cast<std::int64_t>(dimension));
    for (std::int64_t value : values) sha_update_i64_le(digest, value);
    return digest.finish();
}

std::string array_sha256_f64(const std::vector<double>& values) {
    Sha256 digest;
    sha_update_string(digest, "<f8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(values.size()));
    for (double value : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        sha_update_u64_le(digest, bits);
    }
    return digest.finish();
}

std::string array_sha256_f64_c_order(
    const std::vector<double>& values,
    std::initializer_list<std::size_t> dimensions
) {
    std::uint64_t expected = 1;
    for (std::size_t dimension : dimensions)
        expected = checked_mul(expected, dimension,
                               "C-order digest elements");
    if (values.size() != checked_size(expected, "C-order digest elements"))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 C-order digest shape mismatch"
        );
    Sha256 digest;
    sha_update_string(digest, "<f8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    for (std::size_t dimension : dimensions)
        sha_update_i64_le(digest, static_cast<std::int64_t>(dimension));
    for (double value : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        sha_update_u64_le(digest, bits);
    }
    return digest.finish();
}

std::string array_sha256_f64_matrix_c(
    const std::vector<double>& column_major,
    std::size_t rows, std::size_t columns
) {
    if (column_major.size() != rows * columns)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 probe hash shape mismatch"
        );
    Sha256 digest;
    sha_update_string(digest, "<f8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(rows));
    sha_update_i64_le(digest, static_cast<std::int64_t>(columns));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const double value = column_major[row + rows * column];
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            sha_update_u64_le(digest, bits);
        }
    }
    return digest.finish();
}

std::string array_sha256_u64_matrix_c(
    const std::vector<std::uint64_t>& values,
    std::size_t rows, std::size_t columns
) {
    if (values.size() != stage2_detail::elements_for(
            {rows, columns}, "publication matrix elements"))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 probe-key hash shape mismatch"
        );
    Sha256 digest;
    sha_update_string(digest, "<u8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(rows));
    sha_update_i64_le(digest, static_cast<std::int64_t>(columns));
    for (std::uint64_t value : values) sha_update_u64_le(digest, value);
    return digest.finish();
}

std::string array_sha256_f64_span(
    const double* values, std::size_t count
) {
    Sha256 digest;
    sha_update_string(digest, "<f8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(count));
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        sha_update_u64_le(digest, bits);
    }
    return digest.finish();
}

std::string array_sha256_u64_span(
    const std::uint64_t* values, std::size_t count
) {
    Sha256 digest;
    sha_update_string(digest, "<u8");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(count));
    for (std::size_t index = 0; index < count; ++index)
        sha_update_u64_le(digest, values[index]);
    return digest.finish();
}

std::string variant_probe_plan_sha256(
    const std::vector<std::string>& leaves
) {
    std::ostringstream canonical;
    canonical << "{\"global_probe_count\":" << leaves.size()
              << ",\"magic\":\"SUMMIT_CONTEXTUAL_VARIANT_PROBE_PLAN_V1\""
              << ",\"ordered_probe_sha256s\":[";
    for (std::size_t index = 0; index < leaves.size(); ++index) {
        if (index != 0) canonical << ',';
        canonical << '\"' << leaves[index] << '\"';
    }
    canonical << "]}";
    const std::string encoded = canonical.str();
    Sha256 digest;
    sha_update(digest, encoded.data(), encoded.size());
    return digest.finish();
}

Sha256 begin_missingness_sha256(std::size_t variants, std::size_t samples) {
    Sha256 digest;
    sha_update_string(digest, "|u1");
    const unsigned char zero = 0;
    sha_update(digest, &zero, 1);
    sha_update_i64_le(digest, static_cast<std::int64_t>(variants));
    sha_update_i64_le(digest, static_cast<std::int64_t>(samples));
    return digest;
}

std::string json_escape_ascii(const std::string& value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') {
            output << '\\' << static_cast<char>(character);
        } else if (character == '\b') output << "\\b";
        else if (character == '\f') output << "\\f";
        else if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character < 0x20U || character >= 0x80U) {
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 centering_source must be ASCII"
            );
        } else output << static_cast<char>(character);
    }
    return output.str();
}

std::string scale_plan_sha256(
    const std::string& retained_order,
    const std::string& orientation,
    const std::string& mean_digest,
    const std::string& inverse_digest,
    const std::string& centering_source
) {
    // This is exactly schema.GenotypeScalePlanV1.to_dict() serialized through
    // spec.canonical_json: sorted keys, compact separators, ensure_ascii=True.
    std::ostringstream json;
    json << "{\"affine_inverse_scale_sha256\":\"" << inverse_digest
         << "\",\"affine_mean_sha256\":\"" << mean_digest
         << "\",\"allele_coding\":\"" << kAlleleCodingV1
         << "\",\"allele_orientation\":\"" << orientation
         << "\",\"centering_formula\":\"" << kCenteringFormulaV1
         << "\",\"centering_source\":\""
         << json_escape_ascii(centering_source)
         << "\",\"missing_imputation\":\"" << kMissingImputationV1
         << "\",\"ploidy_policy\":\"" << kPloidyPolicyV1
         << "\",\"policy\":\"" << kScalePolicyV1
         << "\",\"retained_variant_order_sha256\":\"" << retained_order
         << "\",\"scaling_formula\":\"" << kScalingFormulaV1 << "\"}";
    Sha256 digest;
    const std::string payload = json.str();
    sha_update(digest, payload.data(), payload.size());
    return digest.finish();
}

std::string read_regular_fd_text(
    int descriptor, const FileState& state, const char* label
) {
    const std::size_t size = checked_size(
        static_cast<std::uint64_t>(state.size), label
    );
    std::string result(size, '\0');
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t observed = ::pread(
            descriptor, result.data() + offset, size - offset,
            static_cast<off_t>(offset)
        );
        if (observed < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("Failed reading contextual ") + label + ": " +
                std::strerror(errno)
            );
        }
        if (observed == 0)
            throw std::runtime_error(
                std::string("Unexpected EOF in contextual ") + label
            );
        offset += static_cast<std::size_t>(observed);
    }
    return result;
}

std::string full_file_sha256(
    int descriptor, const FileState& state, const char* label
) {
    Sha256 digest;
    std::array<unsigned char, 65536> buffer{};
    std::uint64_t offset = 0;
    const std::uint64_t size = static_cast<std::uint64_t>(state.size);
    while (offset < size) {
        const std::size_t requested = static_cast<std::size_t>(
            std::min<std::uint64_t>(buffer.size(), size - offset)
        );
        const ssize_t observed = ::pread(
            descriptor, buffer.data(), requested, static_cast<off_t>(offset)
        );
        if (observed < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("Failed hashing contextual ") + label + ": " +
                std::strerror(errno)
            );
        }
        if (observed == 0)
            throw std::runtime_error(
                std::string("Unexpected EOF hashing contextual ") + label
            );
        digest.update(buffer.data(), static_cast<std::size_t>(observed));
        offset += static_cast<std::uint64_t>(observed);
    }
    return digest.finish();
}

std::string bed_record_sha256(
    const unsigned char* record, std::size_t bytes
) {
    Sha256 digest;
    digest.update(record, bytes);
    return digest.finish();
}

std::vector<BimEntryV1> parse_bim(
    int descriptor, const FileState& state
) {
    const std::string text = read_regular_fd_text(descriptor, state, "BIM");
    std::istringstream input(text);
    std::vector<BimEntryV1> entries;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty())
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 BIM contains a blank row"
            );
        std::istringstream fields(line);
        std::string chromosome, variant, centimorgan, position, a1, a2, extra;
        if (!(fields >> chromosome >> variant >> centimorgan >> position >> a1 >> a2) ||
            (fields >> extra))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 BIM rows must contain exactly six fields"
            );
        if (variant.empty() || a1.empty() || a2.empty())
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 BIM identity field is empty"
            );
        entries.push_back(BimEntryV1{variant, a1, a2});
    }
    if (entries.empty())
        throw std::runtime_error("ContextualReferenceExecutorV1 BIM is empty");
    return entries;
}

const char* event_class_name(EventClassV1 value) {
    switch (value) {
        case EventClassV1::phase_transition: return "phase_transition";
        case EventClassV1::mutation_checkpoint: return "mutation_checkpoint";
        case EventClassV1::protected_call: return "protected_call";
        case EventClassV1::fault_injection: return "fault_injection";
        case EventClassV1::fault_detection: return "fault_detection";
        case EventClassV1::repair: return "repair";
        case EventClassV1::retry: return "retry";
        case EventClassV1::trusted_fallback: return "trusted_fallback";
        case EventClassV1::semantic_verification: return "semantic_verification";
        case EventClassV1::scratch_release: return "scratch_release";
        case EventClassV1::publication: return "publication";
        case EventClassV1::terminal_failure: return "terminal_failure";
    }
    return "unknown";
}

const char* semantic_phase_name(SemanticPhaseV1 value) {
    switch (value) {
        case SemanticPhaseV1::unspecified: return "unspecified";
        case SemanticPhaseV1::source: return "source";
        case SemanticPhaseV1::action: return "action";
        case SemanticPhaseV1::gram: return "gram";
        case SemanticPhaseV1::group: return "group";
        case SemanticPhaseV1::same_person: return "same_person";
        case SemanticPhaseV1::residual_moments: return "residual_moments";
        case SemanticPhaseV1::trait: return "trait";
        case SemanticPhaseV1::finalization: return "finalization";
        case SemanticPhaseV1::publication: return "publication";
    }
    return "unknown";
}

const char* semantic_role_name(SemanticRoleV1 value) {
    switch (value) {
        case SemanticRoleV1::ordinary: return "ordinary";
        case SemanticRoleV1::tile: return "tile";
        case SemanticRoleV1::global_merge: return "global_merge";
    }
    return "unknown";
}

const char* semantic_placement_name(SemanticPlacementV1 value) {
    switch (value) {
        case SemanticPlacementV1::none: return "none";
        case SemanticPlacementV1::action_scaled: return "action_scaled";
        case SemanticPlacementV1::genotype_scaled: return "genotype_scaled";
    }
    return "unknown";
}

const char* variant_coordinate_mode_name(VariantMembershipV1 value) {
    switch (value) {
        case VariantMembershipV1::logical_half_open_range:
            return "logical_half_open_range_v1";
        case VariantMembershipV1::strict_logical_range:
        case VariantMembershipV1::group_execution_slice:
        case VariantMembershipV1::strict_group_execution_slice:
            return "exact_membership_sha256_v1";
    }
    return "unknown";
}

std::string variant_membership_sha256(
    const SemanticCoordinateV1& coordinate
) {
    return coordinate.variant_membership_count == 0
        ? std::string{} : std::string(
            coordinate.variant_membership_sha256.data(), 64
        );
}

SemanticPhaseV1 semantic_phase_for_operation(
    ContextualDenseSemanticV1 operation
) noexcept {
    const std::size_t index = static_cast<std::size_t>(operation);
    if (index <= static_cast<std::size_t>(
            ContextualDenseSemanticV1::full_target_nn))
        return SemanticPhaseV1::source;
    if (index <= static_cast<std::size_t>(
            ContextualDenseSemanticV1::action_projection_nn))
        return SemanticPhaseV1::action;
    if (operation == ContextualDenseSemanticV1::action_gram_tn)
        return SemanticPhaseV1::gram;
    if (index <= static_cast<std::size_t>(
            ContextualDenseSemanticV1::direct_grouped_tn))
        return SemanticPhaseV1::group;
    if (index <= static_cast<std::size_t>(
            ContextualDenseSemanticV1::same_person_gram_tn))
        return SemanticPhaseV1::same_person;
    return SemanticPhaseV1::trait;
}

std::mutex& contextual_dispatcher_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct ContextualDispatcherObservationV1 {
    std::atomic<std::uint64_t> active{0};
    std::atomic<std::uint64_t> process_maximum{0};
};

ContextualDispatcherObservationV1& contextual_dispatcher_observation() {
    static ContextualDispatcherObservationV1 observation;
    return observation;
}

class ContextualDispatcherActivityV1 final {
public:
    ContextualDispatcherActivityV1()
        : observation_(contextual_dispatcher_observation()),
          active_on_entry_(
              observation_.active.fetch_add(1, std::memory_order_acq_rel) + 1
          ) {
        std::uint64_t maximum = observation_.process_maximum.load(
            std::memory_order_acquire
        );
        while (maximum < active_on_entry_ &&
               !observation_.process_maximum.compare_exchange_weak(
                   maximum, active_on_entry_, std::memory_order_acq_rel,
                   std::memory_order_acquire
               )) {}
    }

    ContextualDispatcherActivityV1(
        const ContextualDispatcherActivityV1&
    ) = delete;
    ContextualDispatcherActivityV1& operator=(
        const ContextualDispatcherActivityV1&
    ) = delete;

    ~ContextualDispatcherActivityV1() {
        observation_.active.fetch_sub(1, std::memory_order_acq_rel);
    }

    bool singleton_process_entry_observed() const noexcept {
        return active_on_entry_ == 1 &&
            observation_.process_maximum.load(std::memory_order_acquire) == 1;
    }

private:
    ContextualDispatcherObservationV1& observation_;
    std::uint64_t active_on_entry_ = 0;
};

std::int64_t current_process_id() noexcept {
    return static_cast<std::int64_t>(::getpid());
}

std::int64_t current_thread_id() noexcept {
#ifdef SYS_gettid
    return static_cast<std::int64_t>(::syscall(SYS_gettid));
#else
    return current_process_id();
#endif
}

std::uint64_t fingerprint_doubles(
    const double* values, std::size_t count
) noexcept {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::size_t index = 0; index < count; ++index) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(
            values + index
        );
        for (std::size_t byte = 0; byte < sizeof(double); ++byte) {
            hash ^= bytes[byte];
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

std::string semantic_anchor(const SemanticCoordinateV1& coordinate) {
    const std::size_t operation = static_cast<std::size_t>(coordinate.operation);
    const char* name = operation < kContextualDenseSemanticCountV1
        ? kContextualDenseSemanticNamesV1[operation] : "none";
    std::ostringstream value;
    value << name
          << "|resident=" << coordinate.resident_begin << ':' << coordinate.resident_end
          << "|probe=" << coordinate.probe_begin << ':' << coordinate.probe_end
          << "|variant=" << coordinate.variant_begin << ':' << coordinate.variant_end
          << "|annotation=" << coordinate.annotation_begin << ':' << coordinate.annotation_end
          << "|context=" << coordinate.context_begin << ':' << coordinate.context_end
          << "|action=" << coordinate.action_begin << ':' << coordinate.action_end;
    if (operation >= static_cast<std::size_t>(
            ContextualDenseSemanticV1::group_target_nn) &&
        operation <= static_cast<std::size_t>(
            ContextualDenseSemanticV1::direct_grouped_tn))
        value << "|group=" << coordinate.group_begin << ':'
              << coordinate.group_end;
    value << "|phase=" << semantic_phase_name(coordinate.phase)
          << "|role=" << semantic_role_name(coordinate.role)
          << "|placement=" << semantic_placement_name(coordinate.placement)
          << "|canonical=" << coordinate.canonical_begin << ':'
          << coordinate.canonical_end;
    if (coordinate.variant_membership !=
            VariantMembershipV1::logical_half_open_range)
        value << "|variant_mode="
              << variant_coordinate_mode_name(coordinate.variant_membership)
              << "|variant_count=" << coordinate.variant_membership_count
              << "|variant_sha256="
              << variant_membership_sha256(coordinate);
    return value.str();
}

template <typename T>
std::uint64_t fnv_vector(std::uint64_t hash, const std::vector<T>& values) {
    const auto update = [](const unsigned char* bytes, std::size_t count,
                           std::uint64_t seed) noexcept {
        for (std::size_t index = 0; index < count; ++index) {
            seed ^= bytes[index];
            seed *= UINT64_C(1099511628211);
        }
        return seed;
    };
    const std::uint64_t count = values.size();
    hash = update(reinterpret_cast<const unsigned char*>(&count), sizeof(count), hash);
    if (!values.empty())
        hash = update(reinterpret_cast<const unsigned char*>(values.data()),
                      values.size() * sizeof(T), hash);
    return hash;
}

std::uint64_t fnv_strings(
    std::uint64_t hash, const std::vector<std::string>& values
) {
    const std::uint64_t count = values.size();
    const auto update = [](const unsigned char* bytes, std::size_t size,
                           std::uint64_t seed) noexcept {
        for (std::size_t index = 0; index < size; ++index) {
            seed ^= bytes[index];
            seed *= UINT64_C(1099511628211);
        }
        return seed;
    };
    hash = update(reinterpret_cast<const unsigned char*>(&count), sizeof(count), hash);
    for (const std::string& value : values) {
        const std::uint64_t size = value.size();
        hash = update(reinterpret_cast<const unsigned char*>(&size), sizeof(size), hash);
        hash = update(reinterpret_cast<const unsigned char*>(value.data()),
                      value.size(), hash);
    }
    return hash;
}

std::uint64_t fnv_string(std::uint64_t hash, const std::string& value) {
    const auto update = [](const unsigned char* bytes, std::size_t size,
                           std::uint64_t seed) noexcept {
        for (std::size_t index = 0; index < size; ++index) {
            seed ^= bytes[index];
            seed *= UINT64_C(1099511628211);
        }
        return seed;
    };
    const std::uint64_t size = value.size();
    hash = update(reinterpret_cast<const unsigned char*>(&size), sizeof(size), hash);
    return update(reinterpret_cast<const unsigned char*>(value.data()),
                  value.size(), hash);
}

template <typename T, std::size_t N>
std::uint64_t fnv_array(std::uint64_t hash, const std::array<T, N>& values) {
    for (const T& value : values) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            hash ^= bytes[index];
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

std::size_t operation_index(const std::string& operation) {
    for (std::size_t index = 0; index < kContextualDenseSemanticCountV1; ++index) {
        if (operation == kContextualDenseSemanticNamesV1[index]) return index;
    }
    return kContextualDenseSemanticCountV1;
}

std::string semantic_point_anchor(
    ContextualDenseSemanticV1 operation,
    std::uint64_t resident, std::uint64_t probe, std::uint64_t variant,
    std::uint64_t annotation, std::uint64_t context, std::uint64_t action,
    std::uint64_t group = 0,
    SemanticPhaseV1 phase = SemanticPhaseV1::unspecified,
    SemanticRoleV1 role = SemanticRoleV1::ordinary,
    SemanticPlacementV1 placement = SemanticPlacementV1::none,
    std::uint64_t canonical = 0,
    bool include_identity = false
) {
    const std::size_t index = static_cast<std::size_t>(operation);
    std::ostringstream value;
    value << kContextualDenseSemanticNamesV1[index]
          << "|resident=" << resident << "|probe=" << probe
          << "|variant=" << variant << "|annotation=" << annotation
          << "|context=" << context << "|action=" << action;
    if (index >= static_cast<std::size_t>(
            ContextualDenseSemanticV1::group_target_nn) &&
        index <= static_cast<std::size_t>(
            ContextualDenseSemanticV1::direct_grouped_tn))
        value << "|group=" << group;
    if (include_identity)
        value << "|phase=" << semantic_phase_name(phase)
              << "|role=" << semantic_role_name(role)
              << "|placement=" << semantic_placement_name(placement)
              << "|canonical=" << canonical;
    return value.str();
}

nb::dict event_dict(const EventV1& event) {
    nb::dict item;
    item["event_class"] = event_class_name(event.event_class);
    item["phase"] = event.phase;
    const std::size_t operation = static_cast<std::size_t>(
        event.coordinate.operation
    );
    item["operation"] = operation < kContextualDenseSemanticCountV1
        ? kContextualDenseSemanticNamesV1[operation] : "none";
    item["semantic_anchor"] = semantic_anchor(event.coordinate);
    item["semantic_phase"] = semantic_phase_name(event.coordinate.phase);
    item["semantic_role"] = semantic_role_name(event.coordinate.role);
    item["semantic_placement"] = semantic_placement_name(
        event.coordinate.placement
    );
    item["canonical_begin"] = event.coordinate.canonical_begin;
    item["canonical_end"] = event.coordinate.canonical_end;
    item["variant_coordinate_mode"] = variant_coordinate_mode_name(
        event.coordinate.variant_membership
    );
    item["variant_range_is_exact"] =
        event.coordinate.variant_range_is_exact;
    item["variant_membership_count"] =
        event.coordinate.variant_membership_count;
    item["variant_membership_sha256"] =
        variant_membership_sha256(event.coordinate);
    item["resolution"] = event.resolution;
    item["attempt"] = event.attempt;
    item["rows"] = event.rows;
    item["columns"] = event.columns;
    item["reduction"] = event.reduction;
    item["left_stride"] = event.left_stride;
    item["right_stride"] = event.right_stride;
    item["output_stride"] = event.output_stride;
    item["transpose_left"] = event.transpose_left;
    item["sequence"] = event.sequence;
    item["process_id"] = event.process_id;
    item["thread_id"] = event.thread_id;
    item["elapsed_ns"] = event.elapsed_ns;
    item["operand_fingerprint_fnv64"] =
        std::to_string(event.operand_fingerprint);
    item["witness_fingerprint_fnv64"] =
        std::to_string(event.witness_fingerprint);
    item["accepted_output_fingerprint_fnv64"] =
        std::to_string(event.accepted_output_fingerprint);
    item["runtime_before_fingerprint_fnv64"] =
        std::to_string(event.runtime_before_fingerprint);
    item["runtime_after_fingerprint_fnv64"] =
        std::to_string(event.runtime_after_fingerprint);
    item["prefix_canary_verified"] = event.prefix_canary_verified;
    item["suffix_canary_verified"] = event.suffix_canary_verified;
    item["finiteness_verified"] = event.finiteness_verified;
    item["serialized_entry_verified"] = event.serialized_entry_verified;
    item["deterministic_non_vendor_backend"] =
        event.deterministic_non_vendor_backend;
    item["resident_begin"] = event.coordinate.resident_begin;
    item["resident_end"] = event.coordinate.resident_end;
    item["probe_begin"] = event.coordinate.probe_begin;
    item["probe_end"] = event.coordinate.probe_end;
    item["variant_begin"] = event.coordinate.variant_begin;
    item["variant_end"] = event.coordinate.variant_end;
    item["annotation_begin"] = event.coordinate.annotation_begin;
    item["annotation_end"] = event.coordinate.annotation_end;
    item["context_begin"] = event.coordinate.context_begin;
    item["context_end"] = event.coordinate.context_end;
    item["action_begin"] = event.coordinate.action_begin;
    item["action_end"] = event.coordinate.action_end;
    item["group_begin"] = event.coordinate.group_begin;
    item["group_end"] = event.coordinate.group_end;
    return item;
}

nb::dict build_provenance_dict() {
    nb::dict result;
    result["schema"] = "contextual_native_build_provenance_v1";
    result["source_commit"] = GWLDCORE_SOURCE_COMMIT;
    result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
    result["compiler_id"] = GWLDCORE_COMPILER_ID;
    result["compiler_version"] = GWLDCORE_COMPILER_VERSION;
    result["cxx_standard"] = 17;
    result["build_type"] = GWLDCORE_BUILD_TYPE;
    result["sanitizer_mode"] = GWLDCORE_SANITIZER_MODE;
    result["asan_enabled"] = bool(GWLDCORE_ASAN_ENABLED);
    result["ubsan_enabled"] = bool(GWLDCORE_UBSAN_ENABLED);
    result["effective_optimization"] = GWLDCORE_EFFECTIVE_OPTIMIZATION;
    result["architecture_tuning"] = GWLDCORE_ARCHITECTURE_TUNING;
    result["configured_compiler_flags"] =
        GWLDCORE_CONFIGURED_COMPILER_FLAGS;
    result["blas_vendor"] = GWLDCORE_BLAS_VENDOR;
    result["gemm_integrity_enabled"] = bool(GWLDCORE_GEMM_INTEGRITY_ENABLED);
    result["gemm_checksum_enabled"] = bool(GWLDCORE_GEMM_CHECKSUM_ENABLED);
    result["private_blas_enabled"] = bool(GWLDCORE_PRIVATE_BLAS_ENABLED);
    result["private_blas_backend"] = GWLDCORE_PRIVATE_BLAS_BACKEND;
    result["private_blas_sha256"] = GWLDCORE_PRIVATE_BLAS_SHA256;
    result["private_blas_source_commit"] =
        GWLDCORE_PRIVATE_BLAS_SOURCE_COMMIT;
    result["private_blas_source_tree_sha256"] =
        GWLDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256;
    result["private_blas_config_family"] =
        GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY;
    result["private_blas_header_sha256"] =
        GWLDCORE_PRIVATE_BLAS_HEADER_SHA256;
    result["private_blas_cblas_header_sha256"] =
        GWLDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256;
    result["private_openblas_enabled"] =
        bool(GWLDCORE_PRIVATE_OPENBLAS_ENABLED);
    result["private_openblas_sha256"] =
        GWLDCORE_PRIVATE_OPENBLAS_SHA256;
    result["native_arch_optimization_enabled"] = bool(GWLDCORE_NATIVE_OPT);
    result["openmp_enabled"] = bool(GWLDCORE_OPENMP_ENABLED);
    result["contextual_dispatch_backend"] = kExecutionBackendV1;
    result["contextual_dispatch_vendor_calls"] = false;
    return result;
}

}  // namespace stage2_detail

class ContextualReferenceExecutorV1 final {
public:
    ContextualReferenceExecutorV1(
        int bed_descriptor,
        int bim_descriptor,
        int fam_descriptor,
        nb_vec1_ro<std::int64_t> retained_sample_indices,
        nb_vec1_ro<std::int64_t> retained_variant_indices,
        std::vector<std::string> expected_variant_ids,
        std::vector<std::string> counted_alleles,
        std::vector<std::string> other_alleles,
        nb_vec1_ro<std::uint8_t> counted_allele_is_a1,
        nb_vec1_ro<double> affine_mean,
        nb_vec1_ro<double> affine_inverse_scale,
        nb_vec1_ro<std::int64_t> expected_missing_counts,
        nb_mat2f_ro<double> fixed_basis,
        nb_mat2f_ro<double> context_basis,
        nb_mat2f_ro<double> annotation_weights,
        nb_vec1_ro<std::int64_t> group_index,
        nb::object sample_probes,
        nb::object philox_keys,
        std::vector<std::string> annotation_names,
        std::vector<std::string> group_names,
        std::string annotation_mode,
        std::string retained_variant_order_sha256,
        std::string affine_mean_sha256,
        std::string affine_inverse_scale_sha256,
        std::string missingness_sha256,
        std::string scale_plan_sha256,
        std::string centering_source,
        std::uint64_t sample_probe_count,
        std::uint64_t variant_block,
        std::uint64_t sample_probe_resident,
        std::uint64_t sample_probe_tile,
        std::uint64_t action_tile,
        std::uint64_t annotation_tile,
        std::uint64_t context_tile,
        int decode_threads,
        int blas_threads,
        std::uint64_t workspace_cap_bytes,
        std::uint64_t telemetry_capacity,
        bool enable_differential_snapshot,
        std::string fault_operation,
        std::string fault_mode,
        std::string fault_semantic_anchor,
        std::string test_checkpoint,
        std::string test_mutation_target,
        std::string numa_policy,
        int output_numa_node,
        nb::object variant_probes,
        nb::object variant_philox_keys,
        std::uint64_t variant_probe_count,
        std::uint64_t variant_probe_tile,
        std::uint64_t group_tile,
        std::string grouped_algorithm,
        std::string direct_grouped_scaling,
        bool enable_grouped_differential,
        nb::object phenotype_batch = nb::none(),
        nb::object residual_basis = nb::none(),
        std::vector<std::string> trait_names = {},
        std::vector<std::string> residual_names = {},
        std::string execution_mode = "reference_v1",
        std::string phenotype_normalization =
            "project_then_unit_residual_variance_v1",
        std::uint64_t trait_feature_tile = 0
    ) :
        expected_variant_ids_(std::move(expected_variant_ids)),
        counted_alleles_(std::move(counted_alleles)),
        other_alleles_(std::move(other_alleles)),
        annotation_names_(std::move(annotation_names)),
        group_names_(std::move(group_names)),
        annotation_mode_(std::move(annotation_mode)),
        expected_retained_order_sha256_(std::move(retained_variant_order_sha256)),
        expected_mean_sha256_(std::move(affine_mean_sha256)),
        expected_inverse_sha256_(std::move(affine_inverse_scale_sha256)),
        expected_missingness_sha256_(std::move(missingness_sha256)),
        expected_scale_plan_sha256_(std::move(scale_plan_sha256)),
        centering_source_(std::move(centering_source)),
        decode_threads_(decode_threads), blas_threads_(blas_threads),
        workspace_cap_bytes_(workspace_cap_bytes),
        telemetry_capacity_(telemetry_capacity),
        enable_differential_snapshot_(enable_differential_snapshot),
        fault_operation_(std::move(fault_operation)),
        fault_mode_(std::move(fault_mode)),
        fault_semantic_anchor_(std::move(fault_semantic_anchor)),
        test_checkpoint_(std::move(test_checkpoint)),
        test_mutation_target_(std::move(test_mutation_target)),
        numa_policy_(std::move(numa_policy)),
        output_numa_node_(output_numa_node),
        grouped_algorithm_(std::move(grouped_algorithm)),
        direct_grouped_scaling_(std::move(direct_grouped_scaling)),
        enable_grouped_differential_(enable_grouped_differential),
        trait_names_(std::move(trait_names)),
        residual_names_(std::move(residual_names)),
        execution_mode_(std::move(execution_mode)),
        phenotype_normalization_(std::move(phenotype_normalization))
    {
        using namespace stage2_detail;
        try {
            validate_protected_gemm_threads(blas_threads_);
            if (decode_threads_ <= 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 decode_threads must be positive"
                );
#ifdef _OPENMP
            validate_configured_openmp_placement_for_entry();
            if (decode_threads_ > effective_openmp_capacity())
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 decode_threads exceed affinity capacity"
                );
#else
            if (decode_threads_ != 1)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 decode_threads must be one without OpenMP"
                );
#endif
            if (numa_policy_ != "unbound_first_touch_v1" || output_numa_node_ != -1)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 supports only unbound_first_touch_v1 with output_numa_node=-1"
                );
            bed_fd_ = duplicate_cloexec(bed_descriptor, "contextual BED");
            bim_fd_ = duplicate_cloexec(bim_descriptor, "contextual BIM");
            fam_fd_ = duplicate_cloexec(fam_descriptor, "contextual FAM");
            bed_state_ = validate_regular_fd(bed_fd_, "contextual BED");
            bim_state_ = validate_regular_fd(bim_fd_, "contextual BIM");
            fam_state_ = validate_regular_fd(fam_fd_, "contextual FAM");
            bim_file_sha256_ = full_file_sha256(
                bim_fd_, bim_state_, "BIM"
            );
            fam_file_sha256_ = full_file_sha256(
                fam_fd_, fam_state_, "FAM"
            );
            n_total_ = count_validated_rows_fd(fam_fd_, fam_state_, "contextual FAM");
            const std::vector<BimEntryV1> bim_entries = parse_bim(bim_fd_, bim_state_);
            m_total_ = checked_blas_dim(bim_entries.size(), "contextual BIM rows");
            const std::size_t expected_bed_bytes = checked_size(
                checked_add(
                    3,
                    checked_mul(
                        (static_cast<std::uint64_t>(n_total_) + 3U) / 4U,
                        static_cast<std::uint64_t>(m_total_), "BED bytes"
                    ), "BED bytes"
                ), "BED bytes"
            );
            if (static_cast<std::uint64_t>(bed_state_.size) != expected_bed_bytes)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 BED size disagrees with BIM/FAM"
                );
            bed_size_ = expected_bed_bytes;
            bytes_per_snp_ = (static_cast<std::size_t>(n_total_) + 3U) / 4U;

            copy_input(retained_sample_indices, retained_samples_);
            copy_input(retained_variant_indices, retained_variants_);
            copy_input(counted_allele_is_a1, counted_a1_);
            copy_input(affine_mean, affine_mean_);
            copy_input(affine_inverse_scale, affine_inverse_scale_);
            copy_input(expected_missing_counts, expected_missing_counts_);
            if (execution_mode_ != "reference_v1" &&
                execution_mode_ != "trait_v1")
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 unsupported execution mode"
                );
            trait_mode_ = execution_mode_ == "trait_v1";
            n_ = retained_samples_.size();
            m_ = retained_variants_.size();
            rank_ = fixed_basis.shape(1);
            q_ = context_basis.shape(1);
            k_ = annotation_weights.shape(1);
            groups_ = group_names_.size();
            if (n_ < 3 || m_ == 0 || rank_ == 0 || rank_ >= n_ ||
                q_ == 0 || q_ > 4 || k_ == 0 || groups_ == 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 dimensions are outside the Stage 2 envelope"
                );
            if (fixed_basis.shape(0) != n_ || context_basis.shape(0) != n_ ||
                annotation_weights.shape(0) != m_ || group_index.shape(0) != m_)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 scientific input shape mismatch"
                );
            if (expected_variant_ids_.size() != m_ || counted_alleles_.size() != m_ ||
                other_alleles_.size() != m_ || counted_a1_.size() != m_ ||
                affine_mean_.size() != m_ || affine_inverse_scale_.size() != m_ ||
                expected_missing_counts_.size() != m_ ||
                annotation_names_.size() != k_)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 retained variant metadata is mis-sized"
                );

            copy_matrix(fixed_basis,
                        elements_for({n_, rank_}, "fixed basis elements"),
                        fixed_basis_);
            copy_matrix(context_basis,
                        elements_for({n_, q_}, "context basis elements"),
                        context_basis_);
            copy_matrix(annotation_weights,
                        elements_for({m_, k_}, "annotation elements"),
                        annotation_weights_);
            copy_input(group_index, group_index_);
            if (trait_mode_) {
                if (phenotype_batch.is_none() || residual_basis.is_none())
                    throw std::runtime_error(
                        "ContextualTraitExecutorV1 requires phenotype and residual batches"
                    );
                const auto phenotypes = nb::cast<nb_mat2f_ro<double>>(
                    phenotype_batch
                );
                const auto residuals = nb::cast<nb_mat2f_ro<double>>(
                    residual_basis
                );
                l_ = phenotypes.shape(1);
                h_ = residuals.shape(1);
                if (phenotypes.shape(0) != n_ || residuals.shape(0) != n_ ||
                    l_ == 0 || h_ == 0 || trait_names_.size() != l_ ||
                    residual_names_.size() != h_)
                    throw std::runtime_error(
                        "ContextualTraitExecutorV1 phenotype/residual shape or name mismatch"
                    );
                copy_matrix(
                    phenotypes, elements_for({n_, l_}, "phenotype elements"),
                    phenotype_batch_
                );
                copy_matrix(
                    residuals, elements_for({n_, h_}, "residual basis elements"),
                    residual_basis_
                );
            } else if (!phenotype_batch.is_none() || !residual_basis.is_none() ||
                       !trait_names_.empty() || !residual_names_.empty()) {
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 trait inputs require trait_v1 mode"
                );
            }
            validate_maps_and_bim(bim_entries);
            validate_owned_science();
            build_component_maps();
            seal_annotation_and_group_maps();
            seal_scale_identity();
            if (trait_mode_) {
                if (!sample_probes.is_none() || !philox_keys.is_none() ||
                    sample_probe_count != 0 || !variant_probes.is_none() ||
                    !variant_philox_keys.is_none() || variant_probe_count != 0)
                    throw std::runtime_error(
                        "ContextualTraitExecutorV1 does not accept reference probes"
                    );
                seal_trait_inputs();
            } else {
                seal_probes(sample_probes, philox_keys, sample_probe_count);
                seal_variant_probes(
                    variant_probes, variant_philox_keys, variant_probe_count
                );
            }

            variant_block_ = resolve_tile(variant_block, m_, "variant_block");
            if (trait_mode_) {
                if (sample_probe_resident != 0 || sample_probe_tile != 0 ||
                    action_tile != 0 || annotation_tile != 0 || context_tile != 0 ||
                    variant_probe_tile != 0 || group_tile != 1 ||
                    grouped_algorithm_ != "group_restricted_action_v1" ||
                    direct_grouped_scaling_ != "action_scaled_v1" ||
                    enable_grouped_differential_)
                    throw std::runtime_error(
                        "ContextualTraitExecutorV1 received reference-only tile or grouped policy"
                    );
                trait_feature_tile_ = resolve_tile(
                    trait_feature_tile,
                    elements_for({q_, variant_block_},
                                 "trait feature block columns"),
                    "trait_feature_tile"
                );
            } else {
                sample_probe_resident_ = resolve_tile(
                    sample_probe_resident, bt_, "sample_probe_resident"
                );
                sample_probe_tile_ = resolve_tile(
                    sample_probe_tile, sample_probe_resident_, "sample_probe_tile"
                );
                action_tile_ = resolve_tile(action_tile, components_, "action_tile");
                annotation_tile_ = resolve_tile(annotation_tile, k_, "annotation_tile");
                context_tile_ = resolve_tile(context_tile, q_, "context_tile");
            }
            if (!trait_mode_ && stage3_enabled_) {
                variant_probe_tile_ = resolve_tile(
                    variant_probe_tile, bd_, "variant_probe_tile"
                );
                group_tile_ = resolve_tile(group_tile, groups_, "group_tile");
                seal_group_execution_order();
            } else if (!trait_mode_ && (variant_probe_tile != 0 || group_tile != 1 ||
                       grouped_algorithm_ != "group_restricted_action_v1" ||
                       direct_grouped_scaling_ != "action_scaled_v1" ||
                       enable_grouped_differential_)) {
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 Stage 3 options require variant probes"
                );
            }
            validate_fault_and_test_policy();
            estimate_ = build_preflight();
            seal_execution_plan();
            if (estimate_.required_workspace_bytes > workspace_cap_bytes_)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 admission: workspace cap is below required_workspace_bytes"
                );
            if (estimate_.required_telemetry_capacity > telemetry_capacity_)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 admission: telemetry capacity is below required_telemetry_capacity"
                );

            const std::size_t protected_count = checked_size(
                checked_add(estimate_.maximum_protected_output_elements, 16U,
                            "protected buffer canaries"),
                "protected buffer elements"
            );
            primary_buffer_.assign(protected_count, kCanaryV1);
            retry_buffer_.assign(protected_count, kCanaryV1);
            witness_buffer_.assign(protected_count, kCanaryV1);
            fallback_buffer_.assign(protected_count, kCanaryV1);
            decoded_scratch_.resize(elements_for(
                {n_, variant_block_}, "decoded scratch elements"));
            missing_scratch_.resize(elements_for(
                {n_, variant_block_}, "missing scratch elements"));
            observed_scratch_.resize(variant_block_);
            const std::size_t maximum_projection_columns = trait_mode_
                ? trait_feature_tile_
                : std::max({
                    sample_probe_tile_, elements_for(
                        {action_tile_, sample_probe_tile_},
                        "maximum projection columns"
                    ),
                    stage3_enabled_ ? elements_for(
                        {k_, q_, variant_probe_tile_},
                        "same-person projection columns"
                    ) : std::size_t{0}
                });
            projection_coeff_scratch_.resize(
                elements_for({rank_, maximum_projection_columns},
                             "projection coefficient elements")
            );
            projection_update_scratch_.resize(
                elements_for({n_, maximum_projection_columns},
                             "projection update elements")
            );
            if (trait_mode_) {
                normalized_phenotypes_.resize(elements_for(
                    {n_, l_}, "normalized phenotype elements"));
                phenotype_coefficient_scratch_.resize(elements_for(
                    {rank_, l_}, "phenotype coefficient elements"));
                leverage_scratch_.resize(n_);
                residual_compressed_scratch_.resize(elements_for(
                    {h_, rank_, rank_}, "residual compressed elements"));
                trait_score_right_scratch_.resize(elements_for(
                    {n_, q_, l_}, "trait score RHS elements"));
                trait_score_scratch_.resize(elements_for(
                    {variant_block_, q_, l_}, "trait score elements"));
                trait_raw_feature_scratch_.resize(elements_for(
                    {n_, q_, variant_block_}, "trait raw feature elements"));
                trait_projected_feature_scratch_.resize(
                    trait_raw_feature_scratch_.size());
                trait_projection_tile_scratch_.resize(elements_for(
                    {n_, trait_feature_tile_},
                    "trait projected feature tile elements"));
                genetic_rhs_.assign(elements_for(
                    {components_, l_}, "genetic RHS elements"), 0.0);
                genetic_traces_.assign(components_, 0.0);
                genetic_residual_.assign(elements_for(
                    {components_, h_}, "genetic residual elements"), 0.0);
                residual_rhs_.assign(elements_for(
                    {h_, l_}, "residual RHS elements"), 0.0);
                residual_traces_.assign(h_, 0.0);
                residual_gram_.assign(elements_for(
                    {h_, h_}, "residual Gram elements"), 0.0);
                group_rhs_.assign(elements_for(
                    {groups_, components_, l_}, "group RHS elements"), 0.0);
                group_trace_.assign(elements_for(
                    {groups_, components_}, "group trace elements"), 0.0);
                group_genetic_residual_.assign(elements_for(
                    {groups_, components_, h_},
                    "group genetic residual elements"), 0.0);
            } else {
            probe_resident_scratch_.resize(elements_for(
                {n_, sample_probe_resident_}, "resident probe elements"));
            projected_probe_scratch_.resize(elements_for(
                {n_, sample_probe_resident_}, "projected probe elements"));
            source_right_scratch_.resize(
                elements_for({n_, context_tile_, sample_probe_tile_},
                             "source RHS elements")
            );
            target_panel_scratch_.resize(
                elements_for({n_, k_, q_, sample_probe_resident_},
                             "target panel elements")
            );
            target_right_scratch_.resize(
                elements_for({variant_block_, annotation_tile_, context_tile_,
                              sample_probe_tile_}, "target RHS elements")
            );
            target_update_scratch_.resize(std::max(
                elements_for({variant_block_, context_tile_, sample_probe_tile_},
                             "source output elements"),
                elements_for({n_, annotation_tile_, context_tile_,
                              sample_probe_tile_}, "target output elements")
            ));
            action_panel_scratch_.resize(
                elements_for({n_, action_tile_, sample_probe_tile_},
                             "action panel elements")
            );
            action_projected_scratch_.resize(
                elements_for({n_, action_tile_, sample_probe_tile_},
                             "projected action elements")
            );
            gram_panel_scratch_.resize(
                elements_for({n_, components_, sample_probe_resident_},
                             "Gram panel elements")
            );
            gram_update_scratch_.resize(elements_for(
                {components_, components_}, "Gram update elements"));
            strict_genotype_scratch_.resize(elements_for(
                {n_, variant_block_}, "strict genotype elements"));
            if (stage3_enabled_) {
                group_target_scratch_.resize(elements_for(
                    {group_tile_, n_, k_, q_, bt_},
                    "group target elements"));
                group_action_scratch_.resize(elements_for(
                    {n_, bt_, action_tile_}, "group action elements"));
                group_cross_scratch_.resize(elements_for(
                    {action_tile_, components_}, "group cross elements"));
                direct_action_scratch_.resize(elements_for(
                    {n_, action_tile_, sample_probe_tile_},
                    "direct action elements"));
                direct_scaled_action_scratch_.resize(direct_action_scratch_.size());
                direct_scaled_genotype_scratch_.resize(elements_for(
                    {n_, variant_block_}, "direct scaled genotype elements"));
                direct_output_scratch_.resize(elements_for(
                    {variant_block_, action_tile_, sample_probe_tile_},
                    "direct grouped output elements"));
                same_right_scratch_.resize(elements_for(
                    {variant_block_, k_, variant_probe_tile_},
                    "same-person right elements"));
                variant_probe_scratch_.resize(elements_for(
                    {m_, variant_probe_tile_},
                    "variant-probe tile elements"));
                same_update_scratch_.resize(elements_for(
                    {n_, k_, variant_probe_tile_},
                    "same-person target update elements"));
                same_base_scratch_.resize(same_update_scratch_.size());
                same_contextual_scratch_.resize(elements_for(
                    {n_, k_, q_, variant_probe_tile_},
                    "same-person contextual elements"));
                same_projected_scratch_.resize(same_contextual_scratch_.size());
                same_g_scratch_.resize(elements_for(
                    {n_, variant_probe_tile_, components_},
                    "same-person component elements"));
                same_probe_sums_.assign(elements_for(
                    {n_, components_}, "same-person persistent sums"), 0.0);
                same_probe_cross_.assign(elements_for(
                    {components_, components_}, "same-person cross elements"), 0.0);
                if (enable_differential_snapshot_) {
                    differential_same_probe_sums_.assign(
                        same_probe_sums_.size(), 0.0
                    );
                    differential_same_probe_cross_.assign(
                        same_probe_cross_.size(), 0.0
                    );
                }
                group_gram_numerator_.assign(elements_for(
                    {groups_, components_, components_},
                    "group Gram numerator elements"), 0.0);
                same_person_.assign(elements_for(
                    {components_, components_}, "same-person elements"), 0.0);
                if (enable_grouped_differential_) {
                    group_restricted_differential_.assign(
                        group_gram_numerator_.size(), 0.0);
                    group_direct_action_.assign(
                        group_gram_numerator_.size(), 0.0);
                    group_direct_genotype_.assign(
                        group_gram_numerator_.size(), 0.0);
                }
                direct_directional_action_.assign(
                    group_gram_numerator_.size(), 0.0);
                direct_directional_genotype_.assign(
                    group_gram_numerator_.size(), 0.0);
                logical_missing_fingerprints_.assign(m_, 0);
            }
            source_scores_.assign(elements_for(
                {m_, q_, bt_}, "source score elements"), 0.0);
            raw_actions_.assign(elements_for(
                {n_, components_, bt_}, "raw action elements"), 0.0);
            gram_.assign(elements_for(
                {components_, components_}, "Gram elements"), 0.0);
            raw_gram_numerator_.assign(elements_for(
                {components_, components_}, "raw Gram elements"), 0.0);
            }
            events_.reserve(checked_size(
                estimate_.required_telemetry_capacity, "telemetry capacity"
            ));
            initial_output_fingerprint_ = fingerprint_outputs();

            bed_base_ = static_cast<unsigned char*>(::mmap(
                nullptr, bed_size_, PROT_READ, MAP_PRIVATE, bed_fd_, 0
            ));
            if (bed_base_ == MAP_FAILED) {
                bed_base_ = nullptr;
                throw std::runtime_error(
                    std::string("ContextualReferenceExecutorV1 BED mmap failed: ") +
                    std::strerror(errno)
                );
            }
            if (bed_base_[0] != 0x6c || bed_base_[1] != 0x1b ||
                bed_base_[2] != 0x01)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 requires SNP-major PLINK BED"
                );
            (void)::madvise(bed_base_, bed_size_, MADV_SEQUENTIAL);
            check_files_unchanged();
            bed_header_sha256_ = bed_record_sha256(bed_base_, 3U);
            retained_bed_record_sha256_.resize(m_);
            file_sample_fingerprint_ = fingerprint_file_samples();
            check_files_unchanged();
            input_fingerprint_ = fingerprint_inputs();
            state_ = "admitted";
        } catch (...) {
            close_internal();
            throw;
        }
    }

    ~ContextualReferenceExecutorV1() { close_internal(); }
    ContextualReferenceExecutorV1(const ContextualReferenceExecutorV1&) = delete;
    ContextualReferenceExecutorV1& operator=(
        const ContextualReferenceExecutorV1&
    ) = delete;

    nb::dict info() const;
    nb::dict preflight() const;
    nb::dict run();
    nb::dict performance_report() const;
    nb::dict failure_report() const;
    nb::dict differential_snapshot() const;
    nb::list semantic_anchors() const;
    std::string wait_for_test_checkpoint(std::uint64_t timeout_ms);
    void release_test_checkpoint();

private:
    template <typename T>
    static void copy_input(
        nb_vec1_ro<T> source, std::vector<T>& destination
    ) {
        destination.assign(source.data(), source.data() + source.shape(0));
    }

    static void copy_matrix(
        nb_mat2f_ro<double> source, std::size_t count,
        std::vector<double>& destination
    ) {
        destination.assign(source.data(), source.data() + count);
    }

    void validate_maps_and_bim(
        const std::vector<stage2_detail::BimEntryV1>& bim_entries
    );
    void validate_owned_science();
    void build_component_maps();
    void seal_annotation_and_group_maps();
    void seal_scale_identity();
    void seal_probes(nb::object sample_probes, nb::object philox_keys,
                     std::uint64_t requested_count);
    void seal_variant_probes(nb::object variant_probes,
                             nb::object variant_philox_keys,
                             std::uint64_t requested_count);
    void seal_trait_inputs();
    void seal_group_execution_order();
    void seal_execution_plan();
    void validate_fault_and_test_policy();
    stage2_detail::PreflightV1 build_preflight() const;
    stage2_detail::PreflightV1 build_trait_preflight() const;
    std::uint64_t fingerprint_inputs() const;
    std::uint64_t fingerprint_source() const;
    std::uint64_t fingerprint_source_action() const;
    std::uint64_t fingerprint_outputs() const;
    std::uint64_t runtime_fingerprint() const;
    void check_inputs_unchanged() const;
    void check_files_unchanged() const;
    Sha256 begin_retained_bed_content_digest() const;
    void finish_retained_bed_content_digest(Sha256& digest);
    void check_hardened_file_content_unchanged() const;
    std::uint64_t fingerprint_file_samples() const;
    void close_internal() noexcept;

    nb::dict publish_compact() const;
    nb::dict publish_trait_compact() const;
    nb::dict file_content_identity_dict() const;
    void execute();
    void source_phase();
    void action_phase();
    void gram_phase();
    void group_phase();
    void group_restricted_phase(std::vector<double>& destination);
    void direct_grouped_phase(bool action_scaled,
                              std::vector<double>& destination);
    void same_person_phase();
    void residual_moments_phase();
    void trait_phase();
    void finalize_and_release();
    void validate_event_ledger_for_publication();
    void checkpoint(const char* name);
    void inject_owned_mutation();

    std::size_t decode_block(
        std::size_t logical_begin, std::size_t logical_end,
        Sha256& missing_digest, Sha256& bed_content_digest
    );
    std::size_t decode_group_execution_block(
        std::size_t execution_begin, std::size_t execution_end
    );
    void fill_variant_probe_tile(
        std::size_t probe_begin, std::size_t probe_count,
        std::vector<double>& probes
    ) const;
    void fill_probe_resident(
        std::size_t resident_begin, std::size_t resident_count,
        std::vector<double>& probes
    ) const;
    void project_panel(
        const double* input, std::size_t columns,
        const stage2_detail::SemanticCoordinateV1& base_coordinate,
        ContextualDenseSemanticV1 tn_operation,
        ContextualDenseSemanticV1 nn_operation,
        std::vector<double>& output
    );
    void protected_tn(
        const stage2_detail::SemanticCoordinateV1& coordinate,
        int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output, int output_stride
    );
    void protected_nn(
        const stage2_detail::SemanticCoordinateV1& coordinate,
        int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output, int output_stride
    );
    void protected_gemm(
        bool transpose_left,
        const stage2_detail::SemanticCoordinateV1& coordinate,
        int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output, int output_stride
    );
    static void scalar_gemm(
        bool transpose_left, int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output
    );
    static void scalar_fallback_gemm(
        bool transpose_left, int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output
    );
    void record_event(
        stage2_detail::EventClassV1 event_class,
        const stage2_detail::SemanticCoordinateV1& coordinate,
        const char* resolution, std::uint32_t attempt,
        int rows = 0, int columns = 0, int reduction = 0,
        int left_stride = 0, int right_stride = 0,
        int output_stride = 0, bool transpose_left = false,
        const stage2_detail::EventEvidenceV1& evidence = {}
    );
    void bind_variant_membership(
        stage2_detail::SemanticCoordinateV1& coordinate,
        stage2_detail::VariantMembershipV1 membership,
        std::size_t source_begin, std::size_t source_end,
        std::size_t annotation = 0
    ) const;
    bool variant_selector_matches(
        std::uint64_t point,
        const stage2_detail::SemanticCoordinateV1& coordinate
    ) const;
    bool fault_matches(const stage2_detail::SemanticCoordinateV1& coordinate);

    std::size_t source_index(std::size_t variant, std::size_t context,
                             std::size_t probe) const {
        return variant + m_ * (context * bt_ + probe);
    }
    std::size_t action_index(std::size_t row, std::size_t component,
                             std::size_t probe) const {
        return row + n_ * (component * bt_ + probe);
    }
    double annotation_value(std::size_t variant, std::size_t annotation) const {
        if (!strict_annotation_ids_.empty())
            return strict_annotation_ids_[variant] == annotation ? 1.0 : 0.0;
        return annotation_weights_[variant + m_ * annotation];
    }

    std::size_t n_ = 0, m_ = 0, rank_ = 0, q_ = 0, k_ = 0;
    std::size_t groups_ = 0, pairs_ = 0, components_ = 0, bt_ = 0, bd_ = 0;
    std::size_t l_ = 0, h_ = 0;
    std::size_t variant_block_ = 0, sample_probe_resident_ = 0;
    std::size_t sample_probe_tile_ = 0, action_tile_ = 0;
    std::size_t annotation_tile_ = 0, context_tile_ = 0;
    std::size_t variant_probe_tile_ = 0, group_tile_ = 1;
    std::size_t trait_feature_tile_ = 0;
    int n_total_ = 0, m_total_ = 0;
    int decode_threads_ = 1, blas_threads_ = 1;
    std::uint64_t workspace_cap_bytes_ = 0, telemetry_capacity_ = 0;
    bool enable_differential_snapshot_ = false;
    bool stage3_enabled_ = false, enable_grouped_differential_ = false;
    bool trait_mode_ = false;
    std::string annotation_mode_, probe_policy_, allele_orientation_;
    std::string probe_identity_sha256_;
    std::string variant_probe_policy_, variant_probe_identity_sha256_;
    std::string grouped_algorithm_, direct_grouped_scaling_;
    std::string execution_mode_, phenotype_normalization_;
    std::string phenotype_batch_sha256_, residual_basis_sha256_;
    std::string group_execution_order_sha256_;
    std::string execution_plan_sha256_;
    std::string retained_sample_map_sha256_, variant_order_allele_sha256_;
    std::string fixed_basis_sha256_, evaluated_phi_sha256_;
    std::string annotation_map_sha256_, group_map_sha256_;
    std::string bim_file_sha256_, fam_file_sha256_;
    std::string bed_header_sha256_;
    std::string retained_bed_content_sha256_;
    std::string gram_phase_sha256_, residual_derived_state_sha256_;
    std::string trait_output_phase_sha256_;
    std::string sealed_plan_sha256_;
    std::string centering_source_, numa_policy_;
    int output_numa_node_ = -1;
    std::string expected_retained_order_sha256_, expected_mean_sha256_;
    std::string expected_inverse_sha256_, expected_missingness_sha256_;
    std::string expected_scale_plan_sha256_;
    std::string observed_retained_order_sha256_, observed_mean_sha256_;
    std::string observed_inverse_sha256_, observed_missingness_sha256_;
    std::string observed_scale_plan_sha256_;
    std::string fault_operation_, fault_mode_, fault_semantic_anchor_;
    stage2_detail::SemanticCoordinateV1 fault_selector_{};
    bool fault_selector_valid_ = false;
    bool fault_phase_selector_valid_ = false;
    bool fault_role_selector_valid_ = false;
    bool fault_placement_selector_valid_ = false;
    bool fault_canonical_selector_valid_ = false;
    std::string test_checkpoint_, test_mutation_target_;
    std::string state_ = "validation_and_sealing";
    const char* phase_ = "admission";

    std::vector<std::int64_t> retained_samples_, retained_variants_;
    std::vector<std::int64_t> expected_missing_counts_, group_index_;
    std::vector<std::uint8_t> counted_a1_;
    std::vector<std::string> expected_variant_ids_, counted_alleles_;
    std::vector<std::string> other_alleles_, annotation_names_, group_names_;
    std::vector<std::string> trait_names_, residual_names_;
    std::vector<double> affine_mean_, affine_inverse_scale_;
    std::vector<double> fixed_basis_, context_basis_, annotation_weights_;
    std::vector<double> phenotype_batch_, residual_basis_;
    std::vector<double> explicit_probes_;
    std::vector<std::uint64_t> philox_keys_;
    std::vector<double> explicit_variant_probes_, variant_probe_scratch_;
    std::vector<std::uint64_t> variant_philox_keys_;
    std::vector<std::size_t> strict_annotation_ids_;
    std::vector<std::int64_t> pair_q_, pair_r_, pair_eta_;
    std::vector<std::int64_t> component_annotation_, component_pair_;
    std::vector<double> annotation_masses_, group_annotation_masses_;
    std::vector<std::int64_t> group_variant_counts_;
    std::vector<std::int64_t> group_execution_order_;
    std::vector<std::uint64_t> logical_missing_fingerprints_;
    std::vector<std::string> retained_bed_record_sha256_;

    stage2_detail::PreflightV1 estimate_;
    std::vector<double> primary_buffer_, retry_buffer_, witness_buffer_;
    std::vector<double> fallback_buffer_, decoded_scratch_;
    std::vector<std::uint8_t> missing_scratch_;
    std::vector<int> observed_scratch_;
    std::vector<double> probe_resident_scratch_, projected_probe_scratch_;
    std::vector<double> projection_coeff_scratch_, projection_update_scratch_;
    std::vector<double> source_right_scratch_, target_panel_scratch_;
    std::vector<double> target_right_scratch_, target_update_scratch_;
    std::vector<double> action_panel_scratch_, action_projected_scratch_;
    std::vector<double> gram_panel_scratch_, gram_update_scratch_;
    std::vector<double> strict_genotype_scratch_;
    std::vector<double> group_target_scratch_, group_action_scratch_;
    std::vector<double> group_cross_scratch_, direct_action_scratch_;
    std::vector<double> direct_scaled_action_scratch_;
    std::vector<double> direct_scaled_genotype_scratch_;
    std::vector<double> direct_output_scratch_;
    std::vector<double> same_right_scratch_, same_update_scratch_;
    std::vector<double> same_base_scratch_, same_contextual_scratch_;
    std::vector<double> same_projected_scratch_, same_g_scratch_;
    std::vector<double> same_probe_sums_, same_probe_cross_;
    std::vector<double> differential_same_probe_sums_;
    std::vector<double> differential_same_probe_cross_;
    std::vector<double> normalized_phenotypes_;
    std::vector<double> phenotype_coefficient_scratch_, leverage_scratch_;
    std::vector<double> residual_compressed_scratch_;
    std::vector<double> trait_score_right_scratch_, trait_score_scratch_;
    std::vector<double> trait_raw_feature_scratch_;
    std::vector<double> trait_projected_feature_scratch_;
    std::vector<double> trait_projection_tile_scratch_;
    std::vector<double> source_scores_, raw_actions_;
    std::vector<double> gram_, raw_gram_numerator_;
    std::vector<double> group_gram_numerator_, same_person_;
    std::vector<double> genetic_rhs_, genetic_traces_, genetic_residual_;
    std::vector<double> residual_rhs_, residual_traces_, residual_gram_;
    std::vector<double> group_rhs_, group_trace_, group_genetic_residual_;
    std::vector<double> group_restricted_differential_;
    std::vector<double> group_direct_action_, group_direct_genotype_;
    std::vector<double> direct_directional_action_, direct_directional_genotype_;
    std::vector<stage2_detail::EventV1> events_;
    std::array<std::uint64_t, kContextualDenseSemanticCountV1> observed_calls_{};
    std::array<stage2_detail::PerformancePhaseRecordV1, 10>
        performance_phases_{};
    std::array<stage2_detail::PerformanceCategoryRecordV1, 5>
        performance_categories_{};
    std::array<stage2_detail::PerformanceOperationRecordV1,
               kContextualDenseSemanticCountV1> performance_operations_{};
    std::uint64_t input_fingerprint_ = 0;
    std::uint64_t source_fingerprint_ = 0, action_fingerprint_ = 0;
    std::uint64_t group_output_fingerprint_ = 0;
    std::uint64_t output_fingerprint_ = 0;
    std::uint64_t initial_output_fingerprint_ = 0;
    std::uint64_t file_sample_fingerprint_ = 0;
    std::uint64_t observed_descriptor_passes_ = 0;
    std::uint64_t observed_decoded_blocks_ = 0;
    std::uint64_t observed_variant_visits_ = 0;
    std::uint64_t observed_group_execution_visits_ = 0;
    std::uint64_t observed_variant_probe_coverage_ = 0;
    std::uint64_t observed_phenotype_projections_ = 0;
    std::uint64_t injection_count_ = 0, repair_count_ = 0;
    std::uint64_t retry_count_ = 0, fallback_count_ = 0;
    std::uint64_t event_sequence_ = 0;
    bool fault_consumed_ = false, scratch_released_ = false;
    bool event_ledger_validated_ = false;
    bool performance_complete_ = false;
    std::string failure_message_;
    std::string failure_type_;
    std::string failure_phase_;
    stage2_detail::SemanticCoordinateV1 last_coordinate_{};
    double maximum_projection_leakage_ = 0.0;
    double gram_pre_symmetry_max_abs_ = 0.0;
    double same_person_pre_symmetry_max_abs_ = 0.0;
    double group_reconstruction_max_abs_ = 0.0;
    double direct_action_restricted_max_abs_ = 0.0;
    double direct_genotype_restricted_max_abs_ = 0.0;
    double direct_placement_max_abs_ = 0.0;
    double residual_gram_pre_symmetry_max_abs_ = 0.0;
    double trait_group_reconstruction_max_abs_ = 0.0;
    double phenotype_projection_leakage_max_abs_ = 0.0;
    double phenotype_normalized_norm_error_max_abs_ = 0.0;

    int bed_fd_ = -1, bim_fd_ = -1, fam_fd_ = -1;
    FileState bed_state_{}, bim_state_{}, fam_state_{};
    unsigned char* bed_base_ = nullptr;
    std::size_t bed_size_ = 0, bytes_per_snp_ = 0;

    mutable std::mutex call_mutex_;
    mutable std::mutex checkpoint_mutex_;
    mutable std::condition_variable checkpoint_condition_;
    std::string reached_checkpoint_;
    bool checkpoint_released_ = false;
};

void ContextualReferenceExecutorV1::validate_maps_and_bim(
    const std::vector<stage2_detail::BimEntryV1>& bim_entries
) {
    std::unordered_set<std::int64_t> sample_seen;
    std::unordered_set<std::int64_t> variant_seen;
    for (std::int64_t row : retained_samples_) {
        if (row < 0 || row >= n_total_ || !sample_seen.insert(row).second)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained sample map is invalid"
            );
    }
    for (std::size_t logical = 0; logical < m_; ++logical) {
        const std::int64_t row = retained_variants_[logical];
        if (row < 0 || row >= m_total_ || !variant_seen.insert(row).second)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained variant order is invalid"
            );
        if (counted_a1_[logical] > 1)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 counted-allele orientation must be binary"
            );
        const auto& observed = bim_entries[static_cast<std::size_t>(row)];
        const std::string& expected_counted = counted_a1_[logical]
            ? observed.allele_1 : observed.allele_2;
        const std::string& expected_other = counted_a1_[logical]
            ? observed.allele_2 : observed.allele_1;
        if (expected_variant_ids_[logical] != observed.variant_id)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained variant ID/order disagrees with BIM"
            );
        if (counted_alleles_[logical] != expected_counted ||
            other_alleles_[logical] != expected_other)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained allele orientation disagrees with BIM"
            );
    }
}

void ContextualReferenceExecutorV1::validate_owned_science() {
    const auto finite_vector = [](const std::vector<double>& values,
                                  const char* label) {
        for (double value : values) {
            if (!std::isfinite(value))
                throw std::runtime_error(
                    std::string("ContextualReferenceExecutorV1 nonfinite ") + label
                );
        }
    };
    finite_vector(affine_mean_, "affine mean");
    finite_vector(affine_inverse_scale_, "affine inverse scale");
    finite_vector(fixed_basis_, "fixed basis");
    finite_vector(context_basis_, "context basis");
    finite_vector(annotation_weights_, "annotation weight");
    for (double value : affine_inverse_scale_) {
        if (!(value > 0.0))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 affine inverse scales must be positive"
            );
    }
    for (std::int64_t value : expected_missing_counts_) {
        if (value < 0 || value > static_cast<std::int64_t>(n_))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 expected missing counts are invalid"
            );
    }
    for (std::int64_t value : group_index_) {
        if (value < 0 || value >= static_cast<std::int64_t>(groups_))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group coverage is invalid"
            );
    }
    std::unordered_set<std::string> annotation_labels;
    std::unordered_set<std::string> group_labels;
    for (const auto& value : annotation_names_) {
        if (value.empty() || !annotation_labels.insert(value).second)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 annotation names must be nonempty and unique"
            );
    }
    for (const auto& value : group_names_) {
        if (value.empty() || !group_labels.insert(value).second)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group names must be nonempty and unique"
            );
    }
    for (std::size_t first = 0; first < rank_; ++first) {
        for (std::size_t second = 0; second < rank_; ++second) {
            double dot = 0.0;
            for (std::size_t row = 0; row < n_; ++row)
                dot += fixed_basis_[row + n_ * first] *
                       fixed_basis_[row + n_ * second];
            const double expected = first == second ? 1.0 : 0.0;
            if (std::abs(dot - expected) > 1.0e-10)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 fixed basis is not orthonormal"
                );
        }
    }
}

void ContextualReferenceExecutorV1::build_component_maps() {
    for (std::size_t coordinate = 0; coordinate < q_; ++coordinate) {
        pair_q_.push_back(static_cast<std::int64_t>(coordinate));
        pair_r_.push_back(static_cast<std::int64_t>(coordinate));
        pair_eta_.push_back(1);
    }
    for (std::size_t first = 0; first < q_; ++first) {
        for (std::size_t second = first + 1; second < q_; ++second) {
            pair_q_.push_back(static_cast<std::int64_t>(first));
            pair_r_.push_back(static_cast<std::int64_t>(second));
            pair_eta_.push_back(2);
        }
    }
    pairs_ = pair_q_.size();
    components_ = stage2_detail::checked_size(
        stage2_detail::checked_mul(k_, pairs_, "component count"),
        "component count"
    );
    component_annotation_.reserve(components_);
    component_pair_.reserve(components_);
    for (std::size_t annotation = 0; annotation < k_; ++annotation) {
        for (std::size_t pair = 0; pair < pairs_; ++pair) {
            component_annotation_.push_back(
                static_cast<std::int64_t>(annotation)
            );
            component_pair_.push_back(static_cast<std::int64_t>(pair));
        }
    }
}

void ContextualReferenceExecutorV1::seal_annotation_and_group_maps() {
    if (annotation_mode_ != "generic_nonnegative_weights_v1" &&
        annotation_mode_ != "strict_disjoint_binary_v1")
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 unsupported annotation mode"
        );
    annotation_masses_.assign(k_, 0.0);
    group_annotation_masses_.assign(groups_ * k_, 0.0);
    group_variant_counts_.assign(groups_, 0);
    if (annotation_mode_ == "strict_disjoint_binary_v1")
        strict_annotation_ids_.resize(m_);
    for (std::size_t variant = 0; variant < m_; ++variant) {
        std::size_t strict_count = 0;
        std::size_t strict_id = 0;
        const std::size_t group = static_cast<std::size_t>(group_index_[variant]);
        ++group_variant_counts_[group];
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            const double value = annotation_weights_[variant + m_ * annotation];
            if (!std::isfinite(value) || value < 0.0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 annotations must be finite and nonnegative"
                );
            annotation_masses_[annotation] += value;
            group_annotation_masses_[group + groups_ * annotation] += value;
            if (annotation_mode_ == "strict_disjoint_binary_v1") {
                if (value != 0.0 && value != 1.0)
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 strict annotation membership must be binary"
                    );
                if (value == 1.0) {
                    strict_id = annotation;
                    ++strict_count;
                }
            }
        }
        if (annotation_mode_ == "strict_disjoint_binary_v1") {
            if (strict_count != 1)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 strict annotation membership must contain exactly one one per variant"
                );
            strict_annotation_ids_[variant] = strict_id;
        }
    }
    for (double mass : annotation_masses_) {
        if (!(mass > 0.0) || !std::isfinite(mass))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 annotation mass must be positive"
            );
    }
    for (std::int64_t count : group_variant_counts_) {
        if (count <= 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 every deletion group must be nonempty"
            );
    }
}

void ContextualReferenceExecutorV1::seal_scale_identity() {
    using namespace stage2_detail;
    observed_retained_order_sha256_ = array_sha256_i64(retained_variants_);
    observed_mean_sha256_ = array_sha256_f64(affine_mean_);
    observed_inverse_sha256_ = array_sha256_f64(affine_inverse_scale_);
    bool all_a1 = true, all_a2 = true;
    for (std::uint8_t value : counted_a1_) {
        all_a1 = all_a1 && value == 1;
        all_a2 = all_a2 && value == 0;
    }
    allele_orientation_ = all_a1 ? "bim_a1_counted_v1" :
        (all_a2 ? "bim_a2_counted_v1" :
                  "mixed_bim_a1_a2_per_variant_v1");
    observed_scale_plan_sha256_ = scale_plan_sha256(
        observed_retained_order_sha256_, allele_orientation_,
        observed_mean_sha256_, observed_inverse_sha256_, centering_source_
    );
    const auto require_digest = [](const std::string& expected,
                                   const std::string& observed,
                                   const char* label) {
        if (!is_sha256(expected) || expected != observed)
            throw std::runtime_error(
                std::string("ContextualReferenceExecutorV1 ") + label +
                " SHA256 digest mismatch"
            );
    };
    require_digest(expected_retained_order_sha256_,
                   observed_retained_order_sha256_, "retained variant order");
    require_digest(expected_mean_sha256_, observed_mean_sha256_, "affine mean");
    require_digest(expected_inverse_sha256_, observed_inverse_sha256_,
                   "affine inverse scale");
    require_digest(expected_scale_plan_sha256_, observed_scale_plan_sha256_,
                   "scale plan");
    if (!is_sha256(expected_missingness_sha256_))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 missingness SHA256 is required"
        );
}

void ContextualReferenceExecutorV1::seal_probes(
    nb::object sample_probes, nb::object philox_keys,
    std::uint64_t requested_count
) {
    const bool has_explicit = !sample_probes.is_none();
    const bool has_counter = !philox_keys.is_none();
    if (has_explicit == has_counter)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 requires exactly one sample-probe representation"
        );
    if (has_explicit) {
        const auto probes = nb::cast<nb_mat2f_ro<double>>(sample_probes);
        if (probes.shape(0) != n_ || probes.shape(1) == 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 explicit probes must be Fortran-contiguous [N,B]"
            );
        bt_ = probes.shape(1);
        if (requested_count != 0 && requested_count != bt_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 explicit sample probe count mismatch"
            );
        explicit_probes_.assign(probes.data(), probes.data() + n_ * bt_);
        for (double value : explicit_probes_) {
            if (value != -1.0 && value != 1.0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 explicit probes must be Rademacher"
                );
        }
        probe_policy_ = "explicit_rademacher_v1";
        probe_identity_sha256_ = stage2_detail::array_sha256_f64_matrix_c(
            explicit_probes_, n_, bt_
        );
    } else {
        const auto keys = nb::cast<nb_mat2c_ro<std::uint64_t>>(philox_keys);
        if (keys.shape(0) == 0 || keys.shape(1) != 2 ||
            requested_count == 0 || requested_count != keys.shape(0))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 Philox keys must be C-contiguous [B,2] with matching sample_probe_count"
            );
        bt_ = keys.shape(0);
        philox_keys_.assign(keys.data(), keys.data() + bt_ * 2U);
        probe_policy_ = "numpy_philox_per_probe_key_v1";
        probe_identity_sha256_ = stage2_detail::array_sha256_u64_matrix_c(
            philox_keys_, bt_, 2
        );
    }
}

void ContextualReferenceExecutorV1::seal_variant_probes(
    nb::object variant_probes, nb::object variant_philox_keys,
    std::uint64_t requested_count
) {
    const bool has_explicit = !variant_probes.is_none();
    const bool has_counter = !variant_philox_keys.is_none();
    stage3_enabled_ = has_explicit || has_counter || requested_count != 0;
    if (!stage3_enabled_) return;
    if (has_explicit == has_counter)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 Stage 3 requires exactly one variant-probe representation"
        );
    if (grouped_algorithm_ != "group_restricted_action_v1" &&
        grouped_algorithm_ != "direct_grouped_tn_v1")
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 unsupported grouped algorithm"
        );
    if (direct_grouped_scaling_ != "action_scaled_v1" &&
        direct_grouped_scaling_ != "genotype_scaled_v1" &&
        direct_grouped_scaling_ != "both_differential_v1")
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 unsupported direct grouped scaling policy"
        );
    if (direct_grouped_scaling_ == "both_differential_v1" &&
        !enable_grouped_differential_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 both direct scaling placements require differential mode"
        );
    if (has_explicit) {
        const auto probes = nb::cast<nb_mat2f_ro<double>>(variant_probes);
        if (probes.shape(0) != m_ || probes.shape(1) < 2)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 explicit variant probes must be Fortran-contiguous [M,B_D] with B_D>=2"
            );
        bd_ = probes.shape(1);
        if (requested_count != 0 && requested_count != bd_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 explicit variant probe count mismatch"
            );
        explicit_variant_probes_.assign(
            probes.data(), probes.data() + m_ * bd_
        );
        for (double value : explicit_variant_probes_) {
            if (value != -1.0 && value != 1.0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 explicit variant probes must be Rademacher"
                );
        }
        variant_probe_policy_ = "explicit_variant_rademacher_v1";
        variant_probe_identity_sha256_ =
            stage2_detail::array_sha256_f64_matrix_c(
                explicit_variant_probes_, m_, bd_
            );
    } else {
        const auto keys = nb::cast<nb_mat2c_ro<std::uint64_t>>(
            variant_philox_keys
        );
        if (keys.shape(0) < 2 || keys.shape(1) != 2 ||
            requested_count != keys.shape(0))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 variant Philox keys must be C-contiguous [B_D,2] with matching count >=2"
            );
        bd_ = keys.shape(0);
        variant_philox_keys_.assign(keys.data(), keys.data() + bd_ * 2U);
        variant_probe_policy_ = "numpy_philox_variant_per_probe_key_v1";
        variant_probe_identity_sha256_ =
            stage2_detail::array_sha256_u64_matrix_c(
                variant_philox_keys_, bd_, 2
            );
    }
    for (std::size_t group = 0; group < groups_; ++group) {
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            const double retained = annotation_masses_[annotation] -
                group_annotation_masses_[group + groups_ * annotation];
            if (!(retained > 0.0) || !std::isfinite(retained))
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 deletion empties an annotation"
                );
        }
    }
}

void ContextualReferenceExecutorV1::seal_trait_inputs() {
    using namespace stage2_detail;
    if (phenotype_normalization_ !=
        "project_then_unit_residual_variance_v1")
        throw std::runtime_error(
            "ContextualTraitExecutorV1 unsupported phenotype normalization"
        );
    const auto validate_names = [](const std::vector<std::string>& names,
                                   const char* label) {
        std::unordered_set<std::string> seen;
        for (const std::string& name : names) {
            if (name.empty() || !seen.insert(name).second)
                throw std::runtime_error(
                    std::string("ContextualTraitExecutorV1 ") + label +
                    " names must be nonempty and unique"
                );
        }
    };
    validate_names(trait_names_, "trait");
    validate_names(residual_names_, "residual");
    for (double value : phenotype_batch_)
        if (!std::isfinite(value))
            throw std::runtime_error(
                "ContextualTraitExecutorV1 phenotype batch contains nonfinite values"
            );
    for (double value : residual_basis_)
        if (!std::isfinite(value))
            throw std::runtime_error(
                "ContextualTraitExecutorV1 residual basis contains nonfinite values"
            );
    for (std::size_t group = 0; group < groups_; ++group) {
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            const double retained = annotation_masses_[annotation] -
                group_annotation_masses_[group + groups_ * annotation];
            if (!(retained > 0.0) || !std::isfinite(retained))
                throw std::runtime_error(
                    "ContextualTraitExecutorV1 deletion empties an annotation"
                );
        }
    }
    seal_group_execution_order();
    phenotype_batch_sha256_ = array_sha256_f64_matrix_c(
        phenotype_batch_, n_, l_
    );
    residual_basis_sha256_ = array_sha256_f64_matrix_c(
        residual_basis_, n_, h_
    );
    Sha256 digest;
    sha_update_string(digest, "contextual_trait_sealed_plan_v1");
    for (const std::string* value : {
            &retained_sample_map_sha256_, &variant_order_allele_sha256_,
            &fixed_basis_sha256_, &evaluated_phi_sha256_,
            &annotation_map_sha256_, &group_map_sha256_,
            &observed_scale_plan_sha256_, &phenotype_batch_sha256_,
            &residual_basis_sha256_, &annotation_mode_,
            &phenotype_normalization_}) {
        sha_update_u64_le(digest, value->size());
        sha_update(digest, value->data(), value->size());
    }
    for (const auto* names : {&annotation_names_, &group_names_,
                              &trait_names_, &residual_names_}) {
        for (const std::string& value : *names) {
            sha_update_u64_le(digest, value.size());
            sha_update(digest, value.data(), value.size());
        }
    }
    sealed_plan_sha256_ = digest.finish();
}

void ContextualReferenceExecutorV1::seal_group_execution_order() {
    using namespace stage2_detail;
    group_execution_order_.resize(m_);
    std::iota(group_execution_order_.begin(), group_execution_order_.end(), 0);
    std::stable_sort(
        group_execution_order_.begin(), group_execution_order_.end(),
        [&](std::int64_t left, std::int64_t right) {
            return group_index_[static_cast<std::size_t>(left)] <
                   group_index_[static_cast<std::size_t>(right)];
        }
    );
    std::vector<std::uint8_t> seen(m_, 0);
    std::int64_t previous_group = -1;
    for (std::int64_t logical_value : group_execution_order_) {
        if (logical_value < 0 ||
            logical_value >= static_cast<std::int64_t>(m_))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group execution order has an unknown variant"
            );
        const std::size_t logical = static_cast<std::size_t>(logical_value);
        if (seen[logical] != 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group execution order duplicates a variant"
            );
        seen[logical] = 1;
        const std::int64_t group = group_index_[logical];
        if (group < previous_group)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group execution order is not contiguous"
            );
        previous_group = group;
    }
    if (std::find(seen.begin(), seen.end(), std::uint8_t{0}) != seen.end())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 group execution order has a coverage gap"
        );
    group_execution_order_sha256_ = array_sha256_i64(group_execution_order_);
    retained_sample_map_sha256_ = array_sha256_i64(retained_samples_);
    fixed_basis_sha256_ = array_sha256_f64_matrix_c(
        fixed_basis_, n_, rank_
    );
    evaluated_phi_sha256_ = array_sha256_f64_matrix_c(
        context_basis_, n_, q_
    );
    annotation_map_sha256_ = array_sha256_f64_matrix_c(
        annotation_weights_, m_, k_
    );
    group_map_sha256_ = array_sha256_i64(group_index_);
    Sha256 allele_digest;
    sha_update_string(allele_digest, "variant_order_allele_v1");
    for (std::size_t variant = 0; variant < m_; ++variant) {
        sha_update_i64_le(allele_digest, retained_variants_[variant]);
        for (const std::string* value : {
                &expected_variant_ids_[variant], &counted_alleles_[variant],
                &other_alleles_[variant]}) {
            sha_update_u64_le(allele_digest, value->size());
            sha_update(allele_digest, value->data(), value->size());
        }
        sha_update(allele_digest, &counted_a1_[variant], 1);
    }
    variant_order_allele_sha256_ = allele_digest.finish();
    Sha256 plan_digest;
    sha_update_string(plan_digest, "contextual_reference_sealed_plan_v1");
    for (const std::string* value : {
            &retained_sample_map_sha256_, &variant_order_allele_sha256_,
            &fixed_basis_sha256_, &evaluated_phi_sha256_,
            &annotation_map_sha256_, &group_map_sha256_,
            &observed_scale_plan_sha256_, &probe_identity_sha256_,
            &variant_probe_identity_sha256_, &group_execution_order_sha256_,
            &annotation_mode_, &grouped_algorithm_,
            &direct_grouped_scaling_}) {
        sha_update_u64_le(plan_digest, value->size());
        sha_update(plan_digest, value->data(), value->size());
    }
    sealed_plan_sha256_ = plan_digest.finish();
}

void ContextualReferenceExecutorV1::seal_execution_plan() {
    using namespace stage2_detail;
    retained_sample_map_sha256_ = array_sha256_i64(retained_samples_);
    fixed_basis_sha256_ = array_sha256_f64_matrix_c(
        fixed_basis_, n_, rank_
    );
    evaluated_phi_sha256_ = array_sha256_f64_matrix_c(
        context_basis_, n_, q_
    );
    annotation_map_sha256_ = array_sha256_f64_matrix_c(
        annotation_weights_, m_, k_
    );
    group_map_sha256_ = array_sha256_i64(group_index_);
    Sha256 allele_digest;
    sha_update_string(allele_digest, "variant_order_allele_v1");
    for (std::size_t variant = 0; variant < m_; ++variant) {
        sha_update_i64_le(allele_digest, retained_variants_[variant]);
        for (const std::string* value : {
                &expected_variant_ids_[variant], &counted_alleles_[variant],
                &other_alleles_[variant]}) {
            sha_update_u64_le(allele_digest, value->size());
            sha_update(allele_digest, value->data(), value->size());
        }
        sha_update(allele_digest, &counted_a1_[variant], 1);
    }
    variant_order_allele_sha256_ = allele_digest.finish();
    Sha256 digest;
    sha_update_string(digest, "contextual_execution_plan_v2");
    const auto add_string = [&](const std::string& value) {
        sha_update_u64_le(digest, value.size());
        sha_update(digest, value.data(), value.size());
    };
    for (const char* value : {
            kBackendV1, kTraitBackendV1, kExecutionBackendV1})
        add_string(value);
    for (const std::string* value : {
            &execution_mode_, &annotation_mode_, &probe_policy_,
            &probe_identity_sha256_, &variant_probe_policy_,
            &variant_probe_identity_sha256_, &grouped_algorithm_,
            &direct_grouped_scaling_, &phenotype_normalization_,
            &phenotype_batch_sha256_, &residual_basis_sha256_,
            &retained_sample_map_sha256_, &variant_order_allele_sha256_,
            &fixed_basis_sha256_, &evaluated_phi_sha256_,
            &annotation_map_sha256_, &group_map_sha256_,
            &observed_scale_plan_sha256_, &observed_missingness_sha256_,
            &group_execution_order_sha256_, &numa_policy_})
        add_string(*value);
    for (std::uint64_t value : {
            kContextualBackendVersionV1, n_, m_, rank_, q_, k_, groups_,
            pairs_, components_, bt_, bd_, l_, h_, variant_block_,
            sample_probe_resident_, sample_probe_tile_, action_tile_,
            annotation_tile_, context_tile_, variant_probe_tile_, group_tile_,
            trait_feature_tile_, static_cast<std::uint64_t>(decode_threads_),
            static_cast<std::uint64_t>(blas_threads_),
            static_cast<std::uint64_t>(enable_differential_snapshot_),
            static_cast<std::uint64_t>(enable_grouped_differential_),
            estimate_.total_protected_calls,
            estimate_.maximum_protected_output_elements,
            estimate_.required_workspace_bytes,
            estimate_.required_telemetry_capacity})
        sha_update_u64_le(digest, value);
    for (std::size_t index = 0;
         index < kContextualDenseSemanticCountV1; ++index) {
        add_string(kContextualDenseSemanticNamesV1[index]);
        sha_update_u64_le(digest, estimate_.calls[index]);
    }
    execution_plan_sha256_ = digest.finish();
}

void ContextualReferenceExecutorV1::validate_fault_and_test_policy() {
    using namespace stage2_detail;
    static const std::unordered_set<std::string> modes = {
        "none", "one_shot", "repeated", "repair_corruption",
        "force_fallback", "fallback_corruption", "nan", "inf",
        "canary", "operand_mutation", "fallback_failure",
        "runtime_mutation", "semantic_group_reconstruction"
    };
    if (modes.find(fault_mode_) == modes.end())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 unsupported fault mode"
        );
    const std::size_t fault_index = operation_index(fault_operation_);
    if (fault_mode_ == "none") {
        if (!fault_operation_.empty() || !fault_semantic_anchor_.empty())
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 fault selector requires a non-none mode"
            );
    } else if (fault_mode_ == "semantic_group_reconstruction") {
        if (trait_mode_ || !stage3_enabled_ || !fault_operation_.empty() ||
            !fault_semantic_anchor_.empty())
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 semantic group reconstruction fault requires Stage 3 and no protected selector"
            );
    } else {
        const bool active_semantic = trait_mode_
            ? fault_index >= static_cast<std::size_t>(
                  ContextualDenseSemanticV1::trait_score_tn) &&
              fault_index < kContextualDenseSemanticCountV1
            : fault_index < (stage3_enabled_ ? 14U : 7U);
        if (!active_semantic)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 fault operation is not active in this plan"
            );
        if (fault_semantic_anchor_.empty())
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 fault injection requires a semantic point anchor"
            );
    }
    static const std::unordered_set<std::string> reference_checkpoints = {
        "", "post_seal", "post_admission", "post_source", "post_action",
        "post_gram", "post_group", "post_same_person",
        "pre_finalization", "pre_publication"
    };
    static const std::unordered_set<std::string> trait_checkpoints = {
        "", "post_seal", "post_admission", "post_residual_moments",
        "post_trait", "pre_finalization", "pre_publication"
    };
    const auto& checkpoints = trait_mode_
        ? trait_checkpoints : reference_checkpoints;
    if (checkpoints.find(test_checkpoint_) == checkpoints.end())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 unsupported test checkpoint"
        );
    static const std::unordered_set<std::string> reference_mutation_targets = {
        "none", "fixed_basis", "context_basis", "annotations",
        "affine_mean", "affine_inverse_scale", "sample_probes", "probe_keys",
        "variant_probes", "variant_probe_keys", "group_execution_order",
        "retained_samples", "retained_variants", "groups",
        "counted_allele_flags", "variant_metadata",
        "expected_missing_counts", "annotation_names", "group_names",
        "pair_component_maps", "output_buffers", "retained_bed_record"
    };
    static const std::unordered_set<std::string> trait_mutation_targets = {
        "none", "fixed_basis", "context_basis", "annotations",
        "affine_mean", "affine_inverse_scale", "group_execution_order",
        "retained_samples", "retained_variants", "groups",
        "counted_allele_flags", "variant_metadata",
        "expected_missing_counts", "annotation_names", "group_names",
        "phenotype_batch", "residual_basis", "trait_names",
        "residual_names", "pair_component_maps", "output_buffers",
        "retained_bed_record"
    };
    const auto& mutation_targets = trait_mode_
        ? trait_mutation_targets : reference_mutation_targets;
    if (mutation_targets.find(test_mutation_target_) == mutation_targets.end())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 unsupported owned mutation target"
        );
    if (test_mutation_target_ != "none" && test_checkpoint_.empty())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 owned mutation target requires a test checkpoint"
        );
    if (!fault_semantic_anchor_.empty()) {
        // Legacy operation + global start-point selectors remain accepted.
        // Optional identity fields disambiguate intentional operation reuse.
        std::vector<std::string> fields;
        std::istringstream input(fault_semantic_anchor_);
        std::string field;
        while (std::getline(input, field, '|')) fields.push_back(field);
        if (fields.size() < 7U)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 malformed semantic point anchor"
            );
        const auto parse = [](const std::string& field, const char* prefix) {
            const std::string expected(prefix);
            if (field.rfind(expected, 0) != 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 malformed semantic point anchor"
                );
            std::size_t consumed = 0;
            const auto value = std::stoull(field.substr(expected.size()), &consumed);
            if (consumed != field.size() - expected.size())
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 malformed semantic point anchor"
                );
            return static_cast<std::uint64_t>(value);
        };
        if (fields[0] != fault_operation_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 fault operation/anchor mismatch"
            );
        fault_selector_.operation = static_cast<ContextualDenseSemanticV1>(fault_index);
        fault_selector_.resident_begin = parse(fields[1], "resident=");
        fault_selector_.probe_begin = parse(fields[2], "probe=");
        fault_selector_.variant_begin = parse(fields[3], "variant=");
        fault_selector_.annotation_begin = parse(fields[4], "annotation=");
        fault_selector_.context_begin = parse(fields[5], "context=");
        fault_selector_.action_begin = parse(fields[6], "action=");
        const bool grouped = fault_index >= static_cast<std::size_t>(
                                 ContextualDenseSemanticV1::group_target_nn) &&
            fault_index <= static_cast<std::size_t>(
                                 ContextualDenseSemanticV1::direct_grouped_tn);
        bool group_seen = false;
        for (std::size_t index = 7; index < fields.size(); ++index) {
            const std::string& token = fields[index];
            if (token.rfind("group=", 0) == 0) {
                if (group_seen)
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 duplicate group selector"
                    );
                fault_selector_.group_begin = parse(token, "group=");
                group_seen = true;
            } else if (token.rfind("phase=", 0) == 0) {
                const std::string value = token.substr(6);
                static const std::array<std::pair<const char*, SemanticPhaseV1>, 9>
                    phases{{
                        {"source", SemanticPhaseV1::source},
                        {"action", SemanticPhaseV1::action},
                        {"gram", SemanticPhaseV1::gram},
                        {"group", SemanticPhaseV1::group},
                        {"same_person", SemanticPhaseV1::same_person},
                        {"residual_moments", SemanticPhaseV1::residual_moments},
                        {"trait", SemanticPhaseV1::trait},
                        {"finalization", SemanticPhaseV1::finalization},
                        {"publication", SemanticPhaseV1::publication},
                    }};
                const auto match = std::find_if(
                    phases.begin(), phases.end(), [&](const auto& candidate) {
                        return value == candidate.first;
                    }
                );
                if (match == phases.end())
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 unknown phase selector"
                    );
                fault_selector_.phase = match->second;
                fault_phase_selector_valid_ = true;
            } else if (token.rfind("role=", 0) == 0) {
                const std::string value = token.substr(5);
                if (value == "ordinary")
                    fault_selector_.role = SemanticRoleV1::ordinary;
                else if (value == "tile")
                    fault_selector_.role = SemanticRoleV1::tile;
                else if (value == "global_merge")
                    fault_selector_.role = SemanticRoleV1::global_merge;
                else
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 unknown role selector"
                    );
                fault_role_selector_valid_ = true;
            } else if (token.rfind("placement=", 0) == 0) {
                const std::string value = token.substr(10);
                if (value == "none")
                    fault_selector_.placement = SemanticPlacementV1::none;
                else if (value == "action_scaled")
                    fault_selector_.placement = SemanticPlacementV1::action_scaled;
                else if (value == "genotype_scaled")
                    fault_selector_.placement = SemanticPlacementV1::genotype_scaled;
                else
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 unknown placement selector"
                    );
                fault_placement_selector_valid_ = true;
            } else if (token.rfind("canonical=", 0) == 0) {
                fault_selector_.canonical_begin = parse(token, "canonical=");
                fault_canonical_selector_valid_ = true;
            } else {
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 unknown semantic selector field"
                );
            }
        }
        if (grouped && !group_seen)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 grouped semantic anchor requires group="
            );
        fault_selector_valid_ = true;
    }
}

stage2_detail::PreflightV1 ContextualReferenceExecutorV1::build_preflight() const {
    using namespace stage2_detail;
    if (trait_mode_) return build_trait_preflight();
    PreflightV1 out;
    const std::uint64_t blocks = contextual_ceil_div_u64_v1(
        m_, variant_block_, "Stage 2 variant blocks"
    );
    const std::uint64_t residents = contextual_ceil_div_u64_v1(
        bt_, sample_probe_resident_, "Stage 2 resident batches"
    );
    std::uint64_t probe_tiles = 0;
    for (std::size_t begin = 0; begin < bt_; begin += sample_probe_resident_) {
        const std::size_t width = std::min(sample_probe_resident_, bt_ - begin);
        probe_tiles = checked_add(
            probe_tiles,
            contextual_ceil_div_u64_v1(
                width, std::min(sample_probe_tile_, width),
                "Stage 2 probe tiles"
            ),
            "Stage 2 probe tiles"
        );
    }
    const std::uint64_t context_tiles = contextual_ceil_div_u64_v1(
        q_, context_tile_, "Stage 2 context tiles"
    );
    const std::uint64_t annotation_tiles = contextual_ceil_div_u64_v1(
        k_, annotation_tile_, "Stage 2 annotation tiles"
    );
    const std::uint64_t action_tiles = contextual_ceil_div_u64_v1(
        components_, action_tile_, "Stage 2 action tiles"
    );
    out.resident_batches = residents;
    out.source_descriptor_passes = residents;
    out.action_descriptor_passes = residents;
    out.source_decoded_blocks = checked_mul(residents, blocks, "source blocks");
    out.action_decoded_blocks = checked_mul(residents, blocks, "action blocks");
    out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::sample_probe_projection_tn)] = probe_tiles;
    out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::sample_probe_projection_nn)] = probe_tiles;
    out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::source_tn)] =
        checked_mul(checked_mul(probe_tiles, blocks, "source calls"),
                    context_tiles, "source calls");
    if (strict_annotation_ids_.empty()) {
        out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::full_target_nn)] =
            checked_mul(
                checked_mul(checked_mul(probe_tiles, blocks, "target calls"),
                            annotation_tiles, "target calls"),
                context_tiles, "target calls"
            );
    } else {
        std::uint64_t active_annotation_blocks = 0;
        for (std::size_t block = 0; block < m_; block += variant_block_) {
            const std::size_t end = std::min(m_, block + variant_block_);
            for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                bool present = false;
                for (std::size_t variant = block; variant < end; ++variant)
                    present = present || strict_annotation_ids_[variant] == annotation;
                if (present) ++active_annotation_blocks;
            }
        }
        out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::full_target_nn)] =
            checked_mul(checked_mul(probe_tiles, active_annotation_blocks,
                                    "strict target calls"),
                        context_tiles, "strict target calls");
    }
    out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::action_projection_tn)] =
        checked_mul(probe_tiles, action_tiles, "action projection calls");
    out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::action_projection_nn)] =
        checked_mul(probe_tiles, action_tiles, "action projection calls");
    out.calls[static_cast<std::size_t>(ContextualDenseSemanticV1::action_gram_tn)] = residents;
    if (stage3_enabled_) {
        const bool execute_restricted =
            grouped_algorithm_ == "group_restricted_action_v1" ||
            enable_grouped_differential_;
        const bool execute_direct =
            grouped_algorithm_ == "direct_grouped_tn_v1" ||
            enable_grouped_differential_;
        const std::uint64_t group_probe_tiles = contextual_ceil_div_u64_v1(
            bt_, sample_probe_tile_, "Stage 3 grouped sample-probe tiles"
        );
        std::uint64_t restricted_blocks = 0;
        for (std::int64_t count : group_variant_counts_)
            restricted_blocks = checked_add(
                restricted_blocks,
                contextual_ceil_div_u64_v1(
                    static_cast<std::uint64_t>(count), variant_block_,
                    "Stage 3 group execution blocks"
                ), "Stage 3 group execution blocks"
            );
        if (execute_restricted) {
            out.group_descriptor_passes = 1;
            out.group_decoded_blocks = restricted_blocks;
            std::uint64_t target_calls = 0;
            if (strict_annotation_ids_.empty()) {
                target_calls = checked_mul(
                    checked_mul(
                        checked_mul(restricted_blocks, group_probe_tiles,
                                    "group target calls"),
                        annotation_tiles, "group target calls"),
                    context_tiles, "group target calls"
                );
            } else {
                std::size_t cursor = 0;
                for (std::size_t group = 0; group < groups_; ++group) {
                    const std::size_t group_end = cursor +
                        static_cast<std::size_t>(group_variant_counts_[group]);
                    for (std::size_t begin = cursor; begin < group_end;
                         begin += variant_block_) {
                        const std::size_t end = std::min(
                            group_end, begin + variant_block_
                        );
                        std::uint64_t active = 0;
                        for (std::size_t annotation = 0; annotation < k_;
                             ++annotation) {
                            bool present = false;
                            for (std::size_t execution = begin;
                                 execution < end; ++execution) {
                                const std::size_t logical =
                                    static_cast<std::size_t>(
                                        group_execution_order_[execution]);
                                present = present ||
                                    strict_annotation_ids_[logical] == annotation;
                            }
                            if (present) ++active;
                        }
                        target_calls = checked_add(
                            target_calls,
                            checked_mul(
                                checked_mul(active, group_probe_tiles,
                                            "strict group target calls"),
                                context_tiles, "strict group target calls"
                            ), "strict group target calls"
                        );
                    }
                    cursor = group_end;
                }
            }
            out.calls[static_cast<std::size_t>(
                ContextualDenseSemanticV1::group_target_nn)] = target_calls;
            out.calls[static_cast<std::size_t>(
                ContextualDenseSemanticV1::group_cross_gram_tn)] =
                checked_mul(groups_, action_tiles, "group cross-Gram calls");
        }
        if (execute_direct) {
            const std::uint64_t placements =
                enable_grouped_differential_ ||
                direct_grouped_scaling_ == "both_differential_v1" ? 2U : 1U;
            out.group_descriptor_passes = checked_add(
                out.group_descriptor_passes, placements,
                "group descriptor passes"
            );
            out.group_decoded_blocks = checked_add(
                out.group_decoded_blocks,
                checked_mul(blocks, placements, "group decoded blocks"),
                "group decoded blocks"
            );
            out.calls[static_cast<std::size_t>(
                ContextualDenseSemanticV1::direct_grouped_tn)] = checked_mul(
                    checked_mul(
                        checked_mul(
                            checked_mul(blocks, q_, "direct grouped calls"),
                            action_tiles, "direct grouped calls"),
                        group_probe_tiles, "direct grouped calls"),
                    placements, "direct grouped calls"
                );
        }
        const std::uint64_t variant_tiles = contextual_ceil_div_u64_v1(
            bd_, variant_probe_tile_, "Stage 3 variant-probe tiles"
        );
        out.same_person_descriptor_passes = variant_tiles;
        out.same_person_decoded_blocks = checked_mul(
            variant_tiles, blocks, "same-person decoded blocks"
        );
        out.calls[static_cast<std::size_t>(
            ContextualDenseSemanticV1::same_person_target_nn)] =
                out.same_person_decoded_blocks;
        out.calls[static_cast<std::size_t>(
            ContextualDenseSemanticV1::same_person_projection_tn)] = variant_tiles;
        out.calls[static_cast<std::size_t>(
            ContextualDenseSemanticV1::same_person_projection_nn)] = variant_tiles;
        out.calls[static_cast<std::size_t>(
            ContextualDenseSemanticV1::same_person_gram_tn)] =
                checked_add(variant_tiles, 1, "same-person Gram calls");
    }
    for (std::uint64_t count : out.calls)
        out.total_protected_calls = checked_add(
            out.total_protected_calls, count, "protected call total"
        );
    out.required_telemetry_capacity = checked_add(
        checked_mul(out.total_protected_calls, 8, "telemetry events"),
        32, "telemetry events"
    );

    const auto b8 = [](std::initializer_list<std::uint64_t> dimensions,
                       const char* label) {
        return bytes_for(dimensions, sizeof(double), label);
    };
    const auto bi8 = [](std::uint64_t count, const char* label) {
        return bytes_for({count}, sizeof(std::int64_t), label);
    };
    const auto bu8 = [](std::uint64_t count, const char* label) {
        return bytes_for({count}, sizeof(std::uint8_t), label);
    };
    out.permanent_bytes = 0;
    const auto add_permanent = [&](std::uint64_t value) {
        out.permanent_bytes = checked_add(
            out.permanent_bytes, value, "permanent bytes"
        );
    };
    add_permanent(bi8(n_, "sample map"));
    add_permanent(bi8(m_, "variant map"));
    add_permanent(bu8(m_, "allele flags"));
    add_permanent(b8({m_, 2}, "affine vectors"));
    add_permanent(bi8(m_, "missing counts"));
    add_permanent(b8({n_, rank_}, "fixed basis"));
    add_permanent(b8({n_, q_}, "context basis"));
    add_permanent(b8({m_, k_}, "annotations"));
    add_permanent(bi8(m_, "groups"));
    add_permanent(b8({m_, q_, bt_}, "source scores"));
    add_permanent(b8({n_, components_, bt_}, "raw actions"));
    add_permanent(b8({components_, components_, 2}, "Gram outputs"));
    add_permanent(b8({k_}, "annotation masses"));
    add_permanent(b8({groups_, k_}, "group annotation masses"));
    add_permanent(bi8(groups_, "group counts"));
    add_permanent(bi8(checked_add(
        checked_mul(pairs_, 3U, "pair map elements"),
        checked_mul(components_, 2U, "component map elements"),
        "component map elements"), "component maps"));
    add_permanent(strict_annotation_ids_.empty()
        ? 0 : bytes_for({m_}, sizeof(std::size_t), "strict annotation IDs"));
    add_permanent(explicit_probes_.empty()
        ? bytes_for({bt_, 2}, sizeof(std::uint64_t), "Philox keys")
        : b8({n_, bt_}, "explicit probes"));
    add_permanent(bytes_for(
        {m_, 64U}, sizeof(char), "retained BED record SHA256 ledger"
    ));
    if (stage3_enabled_) {
        add_permanent(explicit_variant_probes_.empty()
            ? bytes_for({bd_, 2}, sizeof(std::uint64_t),
                        "variant Philox keys")
            : b8({m_, bd_}, "explicit variant probes"));
        add_permanent(bi8(m_, "group execution order"));
        add_permanent(bytes_for({m_}, sizeof(std::uint64_t),
                                "logical missing fingerprints"));
        add_permanent(b8({groups_, components_, components_},
                         "group Gram numerator"));
        add_permanent(b8({components_, components_}, "same-person output"));
        if (enable_differential_snapshot_)
            add_permanent(b8(
                {checked_add(checked_mul(n_, components_,
                                         "differential same-person snapshot"),
                             checked_mul(components_, components_,
                                         "differential same-person snapshot"),
                             "differential same-person snapshot")},
                "differential same-person snapshot"
            ));
        if (enable_grouped_differential_)
            add_permanent(b8({3, groups_, components_, components_},
                             "grouped differential outputs"));
    }

    const std::uint64_t max_projection_columns = std::max<std::uint64_t>({
        sample_probe_tile_, checked_mul(
            action_tile_, sample_probe_tile_, "maximum projection columns"
        ),
        stage3_enabled_ ? checked_mul(
            checked_mul(k_, q_, "same-person projection columns"),
            variant_probe_tile_, "same-person projection columns"
        ) : 0
    });
    out.source_phase_bytes = 0;
    const auto add_source = [&](std::uint64_t value) {
        out.source_phase_bytes = checked_add(
            out.source_phase_bytes, value, "source phase bytes"
        );
    };
    add_source(b8({n_, variant_block_}, "decoded block"));
    add_source(bu8(n_ * variant_block_, "missing block"));
    add_source(bytes_for({variant_block_}, sizeof(int), "missing counts scratch"));
    add_source(b8({n_, sample_probe_resident_, 2}, "resident probes"));
    add_source(b8({rank_, max_projection_columns}, "projection coefficients"));
    add_source(b8({n_, max_projection_columns}, "projection update"));
    add_source(b8({n_, context_tile_, sample_probe_tile_}, "source RHS"));

    out.action_phase_bytes = 0;
    const auto add_action = [&](std::uint64_t value) {
        out.action_phase_bytes = checked_add(
            out.action_phase_bytes, value, "action phase bytes"
        );
    };
    add_action(b8({n_, k_, q_, sample_probe_resident_}, "target panel"));
    add_action(b8({variant_block_, annotation_tile_, context_tile_,
                   sample_probe_tile_}, "target RHS"));
    add_action(b8({std::max<std::uint64_t>(
                       variant_block_, checked_mul(
                           n_, annotation_tile_, "target update rows")),
                   context_tile_, sample_probe_tile_}, "target update"));
    add_action(b8({n_, action_tile_, sample_probe_tile_, 2}, "action panels"));
    add_action(b8({n_, components_, sample_probe_resident_}, "Gram panel"));
    add_action(b8({components_, components_}, "Gram update"));
    add_action(b8({n_, variant_block_}, "strict genotype pack"));
    // All execution scratch is reserved before decode and remains live through
    // both phases.  Report that single tracked arena as the live payload for
    // either phase; max(source, action) therefore counts it exactly once.
    const std::uint64_t execution_arena_bytes = checked_add(
        out.source_phase_bytes, out.action_phase_bytes,
        "execution arena bytes"
    );
    out.source_phase_bytes = execution_arena_bytes;
    out.action_phase_bytes = execution_arena_bytes;

    if (stage3_enabled_) {
        out.group_phase_bytes = 0;
        const auto add_group = [&](std::uint64_t value) {
            out.group_phase_bytes = checked_add(
                out.group_phase_bytes, value, "group phase bytes"
            );
        };
        add_group(b8({group_tile_, n_, k_, q_, bt_}, "group targets"));
        add_group(b8({n_, bt_, action_tile_}, "group actions"));
        add_group(b8({action_tile_, components_}, "group cross"));
        add_group(b8({n_, action_tile_, sample_probe_tile_, 2},
                     "direct action panels"));
        add_group(b8({n_, variant_block_}, "direct scaled genotype"));
        add_group(b8({variant_block_, action_tile_, sample_probe_tile_},
                     "direct output"));
        add_group(b8({2, groups_, components_, components_},
                     "direct directional accumulators"));

        out.same_person_phase_bytes = 0;
        const auto add_same = [&](std::uint64_t value) {
            out.same_person_phase_bytes = checked_add(
                out.same_person_phase_bytes, value,
                "same-person phase bytes"
            );
        };
        add_same(b8({m_, variant_probe_tile_}, "variant-probe tile"));
        add_same(b8({variant_block_, k_, variant_probe_tile_},
                    "same-person right"));
        add_same(b8({n_, k_, variant_probe_tile_, 2},
                    "same-person target panels"));
        add_same(b8({n_, k_, q_, variant_probe_tile_, 2},
                    "same-person contextual panels"));
        add_same(b8({n_, variant_probe_tile_, components_},
                    "same-person component panel"));
        add_same(b8({n_, components_}, "same-person persistent sums"));
        add_same(b8({components_, components_},
                    "same-person cross accumulator"));
    }

    out.maximum_protected_output_elements = std::max<std::uint64_t>({
        checked_mul(rank_, max_projection_columns, "projection TN output"),
        checked_mul(n_, max_projection_columns, "projection NN output"),
        checked_mul(checked_mul(variant_block_, context_tile_, "source output"),
                    sample_probe_tile_, "source output"),
        checked_mul(checked_mul(checked_mul(n_, annotation_tile_, "target output"),
                                context_tile_, "target output"),
                    sample_probe_tile_, "target output"),
        checked_mul(components_, components_, "Gram output"),
        stage3_enabled_ ? checked_mul(
            checked_mul(n_, k_, "same target output"), variant_probe_tile_,
            "same target output") : 0,
        stage3_enabled_ ? checked_mul(
            checked_mul(checked_mul(n_, k_, "same projection output"), q_,
                        "same projection output"), variant_probe_tile_,
            "same projection output") : 0,
        stage3_enabled_ ? checked_mul(
            checked_mul(variant_block_, action_tile_, "direct output"),
            sample_probe_tile_, "direct output") : 0,
        stage3_enabled_ ? checked_mul(action_tile_, components_,
                                      "group cross output") : 0
    });
    out.integrity_reserve_bytes = b8({4,
        checked_add(out.maximum_protected_output_elements, 16,
                    "protected canaries")}, "integrity reserve");
    out.compact_output_bytes = checked_add(
        b8({components_, components_, 2}, "published Grams"),
        checked_add(
            b8({k_}, "published masses"),
            bi8(checked_add(
                    checked_mul(pairs_, 3U, "published pair maps"),
                    checked_mul(components_, 2U, "published component maps"),
                    "published maps"), "published maps"),
            "compact output bytes"
        ), "compact output bytes"
    );
    if (stage3_enabled_)
        out.compact_output_bytes = checked_add(
            out.compact_output_bytes,
            checked_add(
                b8({components_, components_}, "published same-person"),
                checked_add(
                    b8({groups_, components_, components_},
                       "published grouped numerator"),
                    checked_add(
                        b8({2, groups_, k_},
                           "published group masses plus row-major staging"),
                        bi8(groups_, "published group counts"),
                        "Stage 3 compact output bytes"
                    ), "Stage 3 compact output bytes"
                ), "Stage 3 compact output bytes"
            ), "compact output bytes"
        );
    out.telemetry_bytes = bytes_for(
        {out.required_telemetry_capacity}, sizeof(stage2_detail::EventV1),
        "telemetry bytes"
    );
    out.required_workspace_bytes = out.permanent_bytes;
    for (std::uint64_t value : {
            std::max(out.source_phase_bytes, out.action_phase_bytes),
            out.group_phase_bytes, out.same_person_phase_bytes,
            out.integrity_reserve_bytes, out.compact_output_bytes,
            out.telemetry_bytes})
        out.required_workspace_bytes = checked_add(
            out.required_workspace_bytes, value, "required workspace bytes"
        );
    return out;
}

stage2_detail::PreflightV1
ContextualReferenceExecutorV1::build_trait_preflight() const {
    using namespace stage2_detail;
    PreflightV1 out;
    const std::uint64_t blocks = contextual_ceil_div_u64_v1(
        m_, variant_block_, "Stage 4 trait variant blocks"
    );
    out.trait_descriptor_passes = 1;
    out.trait_decoded_blocks = blocks;
    out.calls[static_cast<std::size_t>(
        ContextualDenseSemanticV1::trait_score_tn)] = blocks;
    std::uint64_t projection_calls = 0;
    for (std::size_t begin = 0; begin < m_; begin += variant_block_) {
        const std::size_t width = std::min(variant_block_, m_ - begin);
        projection_calls = checked_add(
            projection_calls,
            checked_mul(
                q_,
                contextual_ceil_div_u64_v1(
                    width, trait_feature_tile_,
                    "Stage 4 trait within-context feature tiles"
                ),
                "Stage 4 trait context-bounded projection calls"
            ),
            "Stage 4 trait projection calls"
        );
    }
    out.calls[static_cast<std::size_t>(
        ContextualDenseSemanticV1::trait_feature_projection_tn)] =
        projection_calls;
    out.calls[static_cast<std::size_t>(
        ContextualDenseSemanticV1::trait_feature_projection_nn)] =
        projection_calls;
    for (std::uint64_t count : out.calls)
        out.total_protected_calls = checked_add(
            out.total_protected_calls, count, "Stage 4 protected calls"
        );
    out.required_telemetry_capacity = checked_add(
        checked_mul(out.total_protected_calls, 8,
                    "Stage 4 telemetry events"),
        32, "Stage 4 telemetry events"
    );
    const auto b8 = [](std::initializer_list<std::uint64_t> dims,
                       const char* label) {
        return bytes_for(dims, sizeof(double), label);
    };
    const auto bi8 = [](std::uint64_t count, const char* label) {
        return bytes_for({count}, sizeof(std::int64_t), label);
    };
    const auto bu8 = [](std::uint64_t count, const char* label) {
        return bytes_for({count}, sizeof(std::uint8_t), label);
    };
    const auto add = [](std::uint64_t& destination, std::uint64_t value,
                        const char* label) {
        destination = checked_add(destination, value, label);
    };
    for (std::uint64_t value : {
            bi8(n_, "trait retained samples"),
            bi8(m_, "trait retained variants"),
            bu8(m_, "trait allele flags"),
            b8({m_, 2}, "trait affine scale"),
            bi8(m_, "trait missing counts"),
            b8({n_, rank_}, "trait fixed basis"),
            b8({n_, q_}, "trait context basis"),
            b8({m_, k_}, "trait annotations"),
            bi8(m_, "trait groups"),
            bi8(m_, "trait group execution order"),
            bytes_for({m_, 64U}, sizeof(char),
                      "trait retained BED record SHA256 ledger"),
            bytes_for({strict_annotation_ids_.size()}, sizeof(std::size_t),
                      "trait strict annotation ids"),
            b8({n_, l_}, "trait phenotype batch"),
            b8({n_, h_}, "trait residual basis"),
            b8({k_}, "trait annotation masses"),
            b8({groups_, k_}, "trait group masses"),
            bi8(groups_, "trait group counts"),
            bi8(checked_add(checked_mul(pairs_, 3, "trait pair maps"),
                            checked_mul(components_, 2, "trait component maps"),
                            "trait maps"), "trait maps"),
            b8({components_, l_}, "trait genetic RHS"),
            b8({components_}, "trait genetic traces"),
            b8({components_, h_}, "trait genetic residual"),
            b8({h_, l_}, "trait residual RHS"),
            b8({h_}, "trait residual traces"),
            b8({h_, h_}, "trait residual Gram"),
            b8({groups_, components_, l_}, "trait group RHS"),
            b8({groups_, components_}, "trait group traces"),
            b8({groups_, components_, h_}, "trait group residual")})
        add(out.permanent_bytes, value, "Stage 4 permanent bytes");

    for (std::uint64_t value : {
            b8({n_, l_}, "normalized phenotypes"),
            b8({rank_, l_}, "phenotype coefficients"),
            b8({n_}, "leverage"),
            b8({h_, rank_, rank_}, "compressed residual matrices")})
        add(out.residual_phase_bytes, value, "Stage 4 residual bytes");

    for (std::uint64_t value : {
            b8({n_, variant_block_}, "trait decoded block"),
            bu8(checked_mul(n_, variant_block_, "trait missing block"),
                "trait missing block"),
            bytes_for({variant_block_}, sizeof(int), "trait missing counts"),
            b8({rank_, trait_feature_tile_}, "trait projection coefficients"),
            b8({n_, trait_feature_tile_}, "trait projection update"),
            b8({n_, q_, l_}, "trait score RHS"),
            b8({variant_block_, q_, l_}, "trait score output"),
            b8({n_, q_, variant_block_, 2}, "trait raw/projected features"),
            b8({n_, trait_feature_tile_}, "trait projected tile")})
        add(out.trait_phase_bytes, value, "Stage 4 trait bytes");

    out.maximum_protected_output_elements = std::max<std::uint64_t>({
        checked_mul(checked_mul(variant_block_, q_, "trait score output"),
                    l_, "trait score output"),
        checked_mul(n_, trait_feature_tile_, "trait feature output"),
        checked_mul(rank_, trait_feature_tile_, "trait projection output")
    });
    out.integrity_reserve_bytes = b8({
        4, checked_add(out.maximum_protected_output_elements, 16,
                       "trait integrity canaries")}, "trait integrity reserve");
    out.compact_output_bytes = 0;
    for (std::uint64_t value : {
            b8({components_, l_}, "published genetic RHS"),
            b8({components_}, "published genetic traces"),
            b8({components_, h_}, "published genetic residual"),
            b8({h_, l_}, "published residual RHS"),
            b8({h_}, "published residual traces"),
            b8({h_, h_}, "published residual Gram"),
            b8({groups_, components_, l_}, "published group RHS"),
            b8({groups_, components_}, "published group traces"),
            b8({groups_, components_, h_}, "published group residual"),
            b8({k_}, "published annotation masses"),
            b8({groups_, k_}, "published group masses"),
            bi8(groups_, "published group counts")})
        add(out.compact_output_bytes, value, "Stage 4 compact output bytes");
    out.telemetry_bytes = bytes_for(
        {out.required_telemetry_capacity}, sizeof(stage2_detail::EventV1),
        "Stage 4 telemetry bytes"
    );
    for (std::uint64_t value : {
            out.permanent_bytes, out.residual_phase_bytes,
            out.trait_phase_bytes, out.integrity_reserve_bytes,
            out.compact_output_bytes, out.telemetry_bytes})
        add(out.required_workspace_bytes, value,
            "Stage 4 required workspace bytes");
    return out;
}

std::uint64_t ContextualReferenceExecutorV1::fingerprint_inputs() const {
    using namespace stage2_detail;
    std::uint64_t hash = UINT64_C(1469598103934665603);
    hash = fnv_vector(hash, retained_samples_);
    hash = fnv_vector(hash, retained_variants_);
    hash = fnv_vector(hash, counted_a1_);
    hash = fnv_vector(hash, affine_mean_);
    hash = fnv_vector(hash, affine_inverse_scale_);
    hash = fnv_vector(hash, expected_missing_counts_);
    hash = fnv_vector(hash, fixed_basis_);
    hash = fnv_vector(hash, context_basis_);
    hash = fnv_vector(hash, annotation_weights_);
    hash = fnv_vector(hash, group_index_);
    hash = fnv_vector(hash, explicit_probes_);
    hash = fnv_vector(hash, philox_keys_);
    hash = fnv_vector(hash, explicit_variant_probes_);
    hash = fnv_vector(hash, variant_philox_keys_);
    hash = fnv_vector(hash, group_execution_order_);
    hash = fnv_vector(hash, strict_annotation_ids_);
    hash = fnv_vector(hash, pair_q_);
    hash = fnv_vector(hash, pair_r_);
    hash = fnv_vector(hash, pair_eta_);
    hash = fnv_vector(hash, component_annotation_);
    hash = fnv_vector(hash, component_pair_);
    hash = fnv_vector(hash, annotation_masses_);
    hash = fnv_vector(hash, group_annotation_masses_);
    hash = fnv_vector(hash, group_variant_counts_);
    hash = fnv_strings(hash, expected_variant_ids_);
    hash = fnv_strings(hash, counted_alleles_);
    hash = fnv_strings(hash, other_alleles_);
    hash = fnv_strings(hash, annotation_names_);
    hash = fnv_strings(hash, group_names_);
    for (const std::string* value : {
            &annotation_mode_, &probe_policy_, &variant_probe_policy_,
            &probe_identity_sha256_, &variant_probe_identity_sha256_,
            &allele_orientation_, &grouped_algorithm_,
            &direct_grouped_scaling_, &group_execution_order_sha256_,
            &centering_source_, &numa_policy_,
            &expected_retained_order_sha256_, &expected_mean_sha256_,
            &expected_inverse_sha256_, &expected_missingness_sha256_,
            &expected_scale_plan_sha256_, &fault_operation_, &fault_mode_,
            &fault_semantic_anchor_, &test_checkpoint_, &test_mutation_target_})
        hash = fnv_string(hash, *value);
    hash = fnv_array(hash, std::array<std::uint64_t, 19>{
        static_cast<std::uint64_t>(n_), static_cast<std::uint64_t>(m_),
        static_cast<std::uint64_t>(rank_), static_cast<std::uint64_t>(q_),
        static_cast<std::uint64_t>(k_), static_cast<std::uint64_t>(groups_),
        static_cast<std::uint64_t>(bt_), static_cast<std::uint64_t>(variant_block_),
        static_cast<std::uint64_t>(sample_probe_resident_),
        static_cast<std::uint64_t>(sample_probe_tile_),
        static_cast<std::uint64_t>(action_tile_),
        static_cast<std::uint64_t>(annotation_tile_),
        static_cast<std::uint64_t>(context_tile_),
        static_cast<std::uint64_t>(bd_),
        static_cast<std::uint64_t>(variant_probe_tile_),
        static_cast<std::uint64_t>(group_tile_),
        static_cast<std::uint64_t>(enable_grouped_differential_),
        static_cast<std::uint64_t>(output_numa_node_),
        file_sample_fingerprint_
    });
    hash = fnv_array(hash, std::array<int, 2>{decode_threads_, blas_threads_});
    if (trait_mode_) {
        hash = fnv_vector(hash, phenotype_batch_);
        hash = fnv_vector(hash, residual_basis_);
        hash = fnv_strings(hash, trait_names_);
        hash = fnv_strings(hash, residual_names_);
        for (const std::string* value : {
                &execution_mode_, &phenotype_normalization_,
                &phenotype_batch_sha256_, &residual_basis_sha256_})
            hash = fnv_string(hash, *value);
        hash = fnv_array(hash, std::array<std::uint64_t, 3>{
            static_cast<std::uint64_t>(l_),
            static_cast<std::uint64_t>(h_),
            static_cast<std::uint64_t>(trait_feature_tile_)
        });
    }
    return hash;
}

std::uint64_t ContextualReferenceExecutorV1::fingerprint_outputs() const {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    hash = stage2_detail::fnv_vector(hash, source_scores_);
    hash = stage2_detail::fnv_vector(hash, raw_actions_);
    hash = stage2_detail::fnv_vector(hash, gram_);
    hash = stage2_detail::fnv_vector(hash, raw_gram_numerator_);
    hash = stage2_detail::fnv_vector(hash, group_gram_numerator_);
    hash = stage2_detail::fnv_vector(hash, same_person_);
    hash = stage2_detail::fnv_vector(hash, group_restricted_differential_);
    hash = stage2_detail::fnv_vector(hash, group_direct_action_);
    hash = stage2_detail::fnv_vector(hash, group_direct_genotype_);
    if (trait_mode_) {
        hash = stage2_detail::fnv_vector(hash, genetic_rhs_);
        hash = stage2_detail::fnv_vector(hash, genetic_traces_);
        hash = stage2_detail::fnv_vector(hash, genetic_residual_);
        hash = stage2_detail::fnv_vector(hash, residual_rhs_);
        hash = stage2_detail::fnv_vector(hash, residual_traces_);
        hash = stage2_detail::fnv_vector(hash, residual_gram_);
        hash = stage2_detail::fnv_vector(hash, group_rhs_);
        hash = stage2_detail::fnv_vector(hash, group_trace_);
        hash = stage2_detail::fnv_vector(hash, group_genetic_residual_);
    }
    return hash;
}

std::uint64_t ContextualReferenceExecutorV1::fingerprint_source() const {
    return stage2_detail::fnv_vector(
        UINT64_C(1469598103934665603), source_scores_
    );
}

std::uint64_t ContextualReferenceExecutorV1::fingerprint_source_action() const {
    std::uint64_t hash = fingerprint_source();
    return stage2_detail::fnv_vector(hash, raw_actions_);
}

std::uint64_t ContextualReferenceExecutorV1::runtime_fingerprint() const {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    hash = stage2_detail::fnv_array(hash, std::array<std::uint64_t, 1>{
        static_cast<std::uint64_t>(blas_threads_)
    });
#ifdef _OPENMP
    hash = stage2_detail::fnv_array(hash, std::array<int, 4>{
        omp_get_dynamic(), omp_get_max_threads(), omp_get_max_active_levels(),
        effective_openmp_capacity()
    });
#endif
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (::sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 cannot authenticate CPU affinity"
        );
    const auto* bytes = reinterpret_cast<const unsigned char*>(&affinity);
    for (std::size_t index = 0; index < sizeof(affinity); ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
#endif
    return hash;
}

void ContextualReferenceExecutorV1::check_inputs_unchanged() const {
    if (fingerprint_inputs() != input_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 owned plan input mutation detected"
        );
}

std::uint64_t ContextualReferenceExecutorV1::fingerprint_file_samples() const {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    std::array<unsigned char, 4096> buffer{};
    const auto update = [&](const unsigned char* bytes, std::size_t size) {
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= UINT64_C(1099511628211);
        }
    };
    const auto sample = [&](int descriptor, const FileState& state) {
        const std::uint64_t size = static_cast<std::uint64_t>(state.size);
        update(reinterpret_cast<const unsigned char*>(&size), sizeof(size));
        const std::size_t width = std::min<std::uint64_t>(buffer.size(), size);
        const auto read_at = [&](std::uint64_t offset) {
            std::size_t consumed = 0;
            while (consumed < width) {
                const ssize_t observed = ::pread(
                    descriptor, buffer.data() + consumed, width - consumed,
                    static_cast<off_t>(offset + consumed)
                );
                if (observed < 0) {
                    if (errno == EINTR) continue;
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 file sample fingerprint read failed"
                    );
                }
                if (observed == 0)
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 file changed during sample fingerprint"
                    );
                consumed += static_cast<std::size_t>(observed);
            }
            update(buffer.data(), width);
        };
        if (width != 0) {
            read_at(0);
            if (size > width) read_at(size - width);
        }
    };
    sample(bed_fd_, bed_state_);
    sample(bim_fd_, bim_state_);
    sample(fam_fd_, fam_state_);
    return hash;
}

Sha256 ContextualReferenceExecutorV1::begin_retained_bed_content_digest()
    const {
    using namespace stage2_detail;
    Sha256 digest;
    sha_update_string(digest, "retained_bed_logical_record_stream_v1");
    sha_update_u64_le(digest, m_);
    sha_update_u64_le(digest, bytes_per_snp_);
    digest.update(bed_base_, 3U);
    return digest;
}

void ContextualReferenceExecutorV1::finish_retained_bed_content_digest(
    Sha256& digest
) {
    const std::string observed = digest.finish();
    if (retained_bed_content_sha256_.empty())
        retained_bed_content_sha256_ = observed;
    else if (observed != retained_bed_content_sha256_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 retained BED logical record stream changed between authoritative decode passes"
        );
}

void ContextualReferenceExecutorV1::check_hardened_file_content_unchanged()
    const {
    using namespace stage2_detail;
    if (bim_file_sha256_.empty() || fam_file_sha256_.empty() ||
        bed_header_sha256_.empty() || retained_bed_content_sha256_.empty())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 hardened file identity is unsealed"
        );
    if (full_file_sha256(bim_fd_, bim_state_, "BIM") != bim_file_sha256_ ||
        full_file_sha256(fam_fd_, fam_state_, "FAM") != fam_file_sha256_ ||
        bed_record_sha256(bed_base_, 3U) != bed_header_sha256_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 hardened file content mutation detected"
        );
}

void ContextualReferenceExecutorV1::check_files_unchanged() const {
    const auto states_match = [&]() {
        return same_state(
                   validate_regular_fd(bed_fd_, "contextual BED"), bed_state_) &&
               same_state(
                   validate_regular_fd(bim_fd_, "contextual BIM"), bim_state_) &&
               same_state(
                   validate_regular_fd(fam_fd_, "contextual FAM"), fam_state_);
    };
    if (!states_match())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 descriptor file mutation detected"
        );
    if (file_sample_fingerprint_ != 0 &&
        fingerprint_file_samples() != file_sample_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 descriptor file boundary-sample mutation detected"
        );
    if (!states_match())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 descriptor file changed during boundary sampling"
        );
}

void ContextualReferenceExecutorV1::close_internal() noexcept {
    if (bed_base_ != nullptr) {
        (void)::munmap(bed_base_, bed_size_);
        bed_base_ = nullptr;
    }
    for (int* descriptor : {&bed_fd_, &bim_fd_, &fam_fd_}) {
        if (*descriptor >= 0) {
            (void)::close(*descriptor);
            *descriptor = -1;
        }
    }
}

std::size_t ContextualReferenceExecutorV1::decode_block(
    std::size_t logical_begin, std::size_t logical_end,
    Sha256& missing_digest, Sha256& bed_content_digest
) {
    if (logical_begin >= logical_end || logical_end > m_ ||
        logical_end - logical_begin > variant_block_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 invalid decode block"
        );
    const std::size_t width = logical_end - logical_begin;
    stage2_detail::PerformanceCategoryScopeV1 performance_scope(
        performance_categories_[static_cast<std::size_t>(
            stage2_detail::PerformanceCategoryV1::decode)],
        static_cast<std::uint64_t>(n_) * width,
        static_cast<std::uint64_t>(bytes_per_snp_) * width
    );
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
    for (std::int64_t local_signed = 0;
         local_signed < static_cast<std::int64_t>(width); ++local_signed) {
        const std::size_t local = static_cast<std::size_t>(local_signed);
        const std::size_t logical = logical_begin + local;
        const std::size_t physical = static_cast<std::size_t>(
            retained_variants_[logical]
        );
        const unsigned char* record = bed_base_ + 3U + physical * bytes_per_snp_;
        int missing = 0;
        for (std::size_t row = 0; row < n_; ++row) {
            const std::size_t physical_row = static_cast<std::size_t>(
                retained_samples_[row]
            );
            const unsigned char code = static_cast<unsigned char>(
                (record[physical_row / 4U] >> (2U * (physical_row % 4U))) & 3U
            );
            double dosage = 0.0;
            std::uint8_t is_missing = 0;
            if (code == 1U) {
                dosage = affine_mean_[logical];
                is_missing = 1;
                ++missing;
            } else {
                const double a1_dosage = code == 0U ? 2.0 :
                    (code == 2U ? 1.0 : 0.0);
                dosage = counted_a1_[logical]
                    ? a1_dosage : 2.0 - a1_dosage;
            }
            decoded_scratch_[row + n_ * local] =
                (dosage - affine_mean_[logical]) *
                affine_inverse_scale_[logical];
            missing_scratch_[local * n_ + row] = is_missing;
        }
        observed_scratch_[local] = missing;
    }
    for (std::size_t local = 0; local < width; ++local) {
        const std::size_t logical = logical_begin + local;
        const std::size_t physical = static_cast<std::size_t>(
            retained_variants_[logical]
        );
        const unsigned char* record =
            bed_base_ + 3U + physical * bytes_per_snp_;
        const std::string record_identity = stage2_detail::bed_record_sha256(
            record, bytes_per_snp_
        );
        if (logical >= retained_bed_record_sha256_.size())
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained BED record ledger bounds mismatch"
            );
        if (retained_bed_record_sha256_[logical].empty())
            retained_bed_record_sha256_[logical] = record_identity;
        else if (record_identity != retained_bed_record_sha256_[logical])
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained BED record content mutation detected during decode"
            );
        stage2_detail::sha_update_u64_le(bed_content_digest, physical);
        stage2_detail::sha_update(bed_content_digest, record, bytes_per_snp_);
        if (observed_scratch_[local] != expected_missing_counts_[logical])
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 missing-call count mismatch"
            );
        stage2_detail::sha_update(
            missing_digest, missing_scratch_.data() + local * n_, n_
        );
        if (stage3_enabled_) {
            std::uint64_t fingerprint = UINT64_C(1469598103934665603);
            for (std::size_t row = 0; row < n_; ++row) {
                fingerprint ^= missing_scratch_[local * n_ + row];
                fingerprint *= UINT64_C(1099511628211);
            }
            if (logical_missing_fingerprints_[logical] == 0)
                logical_missing_fingerprints_[logical] = fingerprint;
            else if (logical_missing_fingerprints_[logical] != fingerprint)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 logical missingness identity changed"
                );
        }
    }
    ++observed_decoded_blocks_;
    observed_variant_visits_ += width;
    return width;
}

std::size_t ContextualReferenceExecutorV1::decode_group_execution_block(
    std::size_t execution_begin, std::size_t execution_end
) {
    if (execution_begin >= execution_end || execution_end > m_ ||
        execution_end - execution_begin > variant_block_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 invalid group execution block"
        );
    const std::size_t width = execution_end - execution_begin;
    stage2_detail::PerformanceCategoryScopeV1 performance_scope(
        performance_categories_[static_cast<std::size_t>(
            stage2_detail::PerformanceCategoryV1::decode)],
        static_cast<std::uint64_t>(n_) * width,
        static_cast<std::uint64_t>(bytes_per_snp_) * width
    );
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
    for (std::int64_t local_signed = 0;
         local_signed < static_cast<std::int64_t>(width); ++local_signed) {
        const std::size_t local = static_cast<std::size_t>(local_signed);
        const std::size_t logical = static_cast<std::size_t>(
            group_execution_order_[execution_begin + local]
        );
        const std::size_t physical = static_cast<std::size_t>(
            retained_variants_[logical]
        );
        const unsigned char* record = bed_base_ + 3U + physical * bytes_per_snp_;
        int missing = 0;
        for (std::size_t row = 0; row < n_; ++row) {
            const std::size_t physical_row = static_cast<std::size_t>(
                retained_samples_[row]
            );
            const unsigned char code = static_cast<unsigned char>(
                (record[physical_row / 4U] >>
                 (2U * (physical_row % 4U))) & 3U
            );
            double dosage = 0.0;
            std::uint8_t is_missing = 0;
            if (code == 1U) {
                dosage = affine_mean_[logical];
                is_missing = 1;
                ++missing;
            } else {
                const double a1_dosage = code == 0U ? 2.0 :
                    (code == 2U ? 1.0 : 0.0);
                dosage = counted_a1_[logical]
                    ? a1_dosage : 2.0 - a1_dosage;
            }
            decoded_scratch_[row + n_ * local] =
                (dosage - affine_mean_[logical]) *
                affine_inverse_scale_[logical];
            missing_scratch_[local * n_ + row] = is_missing;
        }
        observed_scratch_[local] = missing;
    }
    for (std::size_t local = 0; local < width; ++local) {
        const std::size_t logical = static_cast<std::size_t>(
            group_execution_order_[execution_begin + local]
        );
        const std::size_t physical = static_cast<std::size_t>(
            retained_variants_[logical]
        );
        if (logical >= retained_bed_record_sha256_.size() ||
            stage2_detail::bed_record_sha256(
                bed_base_ + 3U + physical * bytes_per_snp_, bytes_per_snp_
            ) != retained_bed_record_sha256_[logical])
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 retained BED record content mutation detected during group decode"
            );
        if (observed_scratch_[local] != expected_missing_counts_[logical])
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group pass missing-call count mismatch"
            );
        std::uint64_t fingerprint = UINT64_C(1469598103934665603);
        for (std::size_t row = 0; row < n_; ++row) {
            fingerprint ^= missing_scratch_[local * n_ + row];
            fingerprint *= UINT64_C(1099511628211);
        }
        if (logical_missing_fingerprints_[logical] == 0 ||
            logical_missing_fingerprints_[logical] != fingerprint)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group execution missingness identity mismatch"
            );
    }
    ++observed_decoded_blocks_;
    observed_variant_visits_ += width;
    observed_group_execution_visits_ += width;
    return width;
}

void ContextualReferenceExecutorV1::fill_variant_probe_tile(
    std::size_t probe_begin, std::size_t probe_count,
    std::vector<double>& probes
) const {
    if (probes.size() < m_ * probe_count || probe_begin + probe_count > bd_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 variant-probe scratch is undersized"
        );
    if (!explicit_variant_probes_.empty()) {
        std::copy(
            explicit_variant_probes_.data() + m_ * probe_begin,
            explicit_variant_probes_.data() + m_ * (probe_begin + probe_count),
            probes.data()
        );
    } else {
        fill_numpy_philox_rademacher(
            probes.data(), checked_blas_dim(m_, "variant rows"),
            variant_philox_keys_.data() + probe_begin * 2U,
            checked_blas_dim(probe_count, "variant probes"),
            decode_threads_
        );
    }
}

void ContextualReferenceExecutorV1::fill_probe_resident(
    std::size_t resident_begin, std::size_t resident_count,
    std::vector<double>& probes
) const {
    if (probes.size() < n_ * resident_count)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 probe scratch is undersized"
        );
    if (!explicit_probes_.empty()) {
        std::copy(
            explicit_probes_.data() + n_ * resident_begin,
            explicit_probes_.data() + n_ * (resident_begin + resident_count),
            probes.data()
        );
    } else {
        fill_numpy_philox_rademacher(
            probes.data(), checked_blas_dim(n_, "sample rows"),
            philox_keys_.data() + resident_begin * 2U,
            checked_blas_dim(resident_count, "resident probes"),
            decode_threads_
        );
    }
}

void ContextualReferenceExecutorV1::project_panel(
    const double* input, std::size_t columns,
    const stage2_detail::SemanticCoordinateV1& base_coordinate,
    ContextualDenseSemanticV1 tn_operation,
    ContextualDenseSemanticV1 nn_operation,
    std::vector<double>& output
) {
    if (columns == 0 || output.size() < n_ * columns ||
        projection_coeff_scratch_.size() < rank_ * columns ||
        projection_update_scratch_.size() < n_ * columns)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 projection scratch is undersized"
        );
    stage2_detail::PerformanceCategoryScopeV1 performance_scope(
        performance_categories_[static_cast<std::size_t>(
            stage2_detail::PerformanceCategoryV1::projection)],
        static_cast<std::uint64_t>(n_) * columns,
        static_cast<std::uint64_t>(n_) * columns * sizeof(double) * 2U
    );
    auto tn_coordinate = base_coordinate;
    tn_coordinate.operation = tn_operation;
    protected_tn(
        tn_coordinate, checked_blas_dim(rank_, "projection rank"),
        checked_blas_dim(columns, "projection columns"),
        checked_blas_dim(n_, "projection rows"),
        fixed_basis_.data(), checked_blas_dim(n_, "fixed basis stride"),
        input, checked_blas_dim(n_, "projection input stride"),
        projection_coeff_scratch_.data(),
        checked_blas_dim(rank_, "projection coefficient stride")
    );
    auto nn_coordinate = base_coordinate;
    nn_coordinate.operation = nn_operation;
    protected_nn(
        nn_coordinate, checked_blas_dim(n_, "projection rows"),
        checked_blas_dim(columns, "projection columns"),
        checked_blas_dim(rank_, "projection rank"),
        fixed_basis_.data(), checked_blas_dim(n_, "fixed basis stride"),
        projection_coeff_scratch_.data(),
        checked_blas_dim(rank_, "projection coefficient stride"),
        projection_update_scratch_.data(),
        checked_blas_dim(n_, "projection update stride")
    );
    double maximum_input = 0.0;
    for (std::size_t index = 0; index < n_ * columns; ++index) {
        output[index] = input[index] - projection_update_scratch_[index];
        maximum_input = std::max(maximum_input, std::abs(input[index]));
        if (!std::isfinite(output[index]))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 nonfinite projected panel"
            );
    }
    double leakage = 0.0;
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t basis = 0; basis < rank_; ++basis) {
            double dot = 0.0;
            for (std::size_t row = 0; row < n_; ++row)
                dot += fixed_basis_[row + n_ * basis] *
                       output[row + n_ * column];
            leakage = std::max(leakage, std::abs(dot));
        }
    }
    maximum_projection_leakage_ = std::max(
        maximum_projection_leakage_, leakage
    );
    if (leakage > 2.0e-10 * (1.0 + maximum_input * n_))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 projection leakage verification failed"
        );
}

void ContextualReferenceExecutorV1::scalar_gemm(
    bool transpose_left, int rows, int columns, int reduction,
    const double* left, int left_stride,
    const double* right, int right_stride,
    double* output
) {
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int inner = 0; inner < reduction; ++inner) {
                const double left_value = transpose_left
                    ? left[static_cast<std::size_t>(inner) +
                           static_cast<std::size_t>(row) * left_stride]
                    : left[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(inner) * left_stride];
                sum += left_value * right[
                    static_cast<std::size_t>(inner) +
                    static_cast<std::size_t>(column) * right_stride
                ];
            }
            output[static_cast<std::size_t>(row) +
                   static_cast<std::size_t>(column) * rows] = sum;
        }
    }
}

void ContextualReferenceExecutorV1::scalar_fallback_gemm(
    bool transpose_left, int rows, int columns, int reduction,
    const double* left, int left_stride,
    const double* right, int right_stride,
    double* output
) {
    std::fill_n(output, static_cast<std::size_t>(rows) * columns, 0.0);
    // Independent trusted implementation: reduction-major accumulation,
    // deliberately distinct from both the cache-tiled primary and the
    // output-entry-major scalar witness.
    for (int inner = 0; inner < reduction; ++inner) {
        for (int column = 0; column < columns; ++column) {
            const double right_value = right[
                static_cast<std::size_t>(inner) +
                static_cast<std::size_t>(column) * right_stride
            ];
            for (int row = 0; row < rows; ++row) {
                const double left_value = transpose_left
                    ? left[static_cast<std::size_t>(inner) +
                           static_cast<std::size_t>(row) * left_stride]
                    : left[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(inner) * left_stride];
                output[static_cast<std::size_t>(row) +
                       static_cast<std::size_t>(column) * rows] +=
                    left_value * right_value;
            }
        }
    }
}

void ContextualReferenceExecutorV1::record_event(
    stage2_detail::EventClassV1 event_class,
    const stage2_detail::SemanticCoordinateV1& coordinate,
    const char* resolution, std::uint32_t attempt,
    int rows, int columns, int reduction,
    int left_stride, int right_stride, int output_stride,
    bool transpose_left,
    const stage2_detail::EventEvidenceV1& evidence
) {
    if (events_.size() >= telemetry_capacity_ ||
        events_.size() >= estimate_.required_telemetry_capacity)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 telemetry overflow"
        );
    stage2_detail::EventV1 event;
    event.event_class = event_class;
    event.phase = phase_;
    event.coordinate = coordinate;
    if (event.coordinate.phase == stage2_detail::SemanticPhaseV1::unspecified &&
        static_cast<std::size_t>(event.coordinate.operation) <
            kContextualDenseSemanticCountV1)
        event.coordinate.phase = stage2_detail::semantic_phase_for_operation(
            event.coordinate.operation
        );
    event.resolution = resolution;
    event.attempt = attempt;
    event.rows = rows;
    event.columns = columns;
    event.reduction = reduction;
    event.left_stride = left_stride;
    event.right_stride = right_stride;
    event.output_stride = output_stride;
    event.transpose_left = transpose_left;
    event.sequence = ++event_sequence_;
    event.process_id = stage2_detail::current_process_id();
    event.thread_id = stage2_detail::current_thread_id();
    event.elapsed_ns = evidence.elapsed_ns;
    event.operand_fingerprint = evidence.operand_fingerprint;
    event.witness_fingerprint = evidence.witness_fingerprint;
    event.accepted_output_fingerprint =
        evidence.accepted_output_fingerprint;
    event.runtime_before_fingerprint = evidence.runtime_before_fingerprint;
    event.runtime_after_fingerprint = evidence.runtime_after_fingerprint;
    event.prefix_canary_verified = evidence.prefix_canary_verified;
    event.suffix_canary_verified = evidence.suffix_canary_verified;
    event.finiteness_verified = evidence.finiteness_verified;
    event.serialized_entry_verified = evidence.serialized_entry_verified;
    event.deterministic_non_vendor_backend =
        evidence.deterministic_non_vendor_backend;
    last_coordinate_ = event.coordinate;
    events_.push_back(event);
}

void ContextualReferenceExecutorV1::bind_variant_membership(
    stage2_detail::SemanticCoordinateV1& coordinate,
    stage2_detail::VariantMembershipV1 membership,
    std::size_t source_begin, std::size_t source_end,
    std::size_t annotation
) const {
    using namespace stage2_detail;
    const bool grouped = membership ==
            VariantMembershipV1::group_execution_slice ||
        membership == VariantMembershipV1::strict_group_execution_slice;
    const bool strict = membership == VariantMembershipV1::strict_logical_range ||
        membership == VariantMembershipV1::strict_group_execution_slice;
    if (membership == VariantMembershipV1::logical_half_open_range ||
        source_begin >= source_end || source_end > m_ ||
        (strict && annotation >= k_))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 invalid exact variant membership"
        );
    const auto logical_at = [&](std::size_t source) {
        return grouped ? static_cast<std::size_t>(
            group_execution_order_[source]
        ) : source;
    };
    const auto retained = [&](std::size_t logical) {
        return !strict || strict_annotation_ids_[logical] == annotation;
    };

    std::size_t count = 0;
    std::size_t minimum = m_;
    std::size_t maximum = 0;
    bool contiguous = true;
    std::size_t previous = 0;
    for (std::size_t source = source_begin; source < source_end; ++source) {
        const std::size_t logical = logical_at(source);
        if (logical >= m_ || !retained(logical)) continue;
        if (count != 0 && logical != previous + 1U) contiguous = false;
        minimum = std::min(minimum, logical);
        maximum = std::max(maximum, logical);
        previous = logical;
        ++count;
    }
    if (count == 0)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 exact variant membership is empty"
        );

    Sha256 digest;
    sha_update_string(digest, "contextual_exact_logical_variant_membership_v1");
    sha_update_u64_le(digest, count);
    for (std::size_t source = source_begin; source < source_end; ++source) {
        const std::size_t logical = logical_at(source);
        if (logical < m_ && retained(logical))
            sha_update_u64_le(digest, logical);
    }
    const std::string identity = digest.finish();
    if (identity.size() != 64U)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 variant membership digest failure"
        );
    coordinate.variant_membership = membership;
    coordinate.variant_membership_source_begin = source_begin;
    coordinate.variant_membership_source_end = source_end;
    coordinate.variant_membership_count = count;
    std::copy(identity.begin(), identity.end(),
              coordinate.variant_membership_sha256.begin());
    coordinate.variant_membership_sha256[64] = '\0';
    coordinate.variant_begin = minimum;
    coordinate.variant_end = maximum + 1U;
    coordinate.variant_range_is_exact = contiguous &&
        maximum - minimum + 1U == count;
    coordinate.canonical_begin = minimum;
    coordinate.canonical_end = maximum + 1U;
}

bool ContextualReferenceExecutorV1::variant_selector_matches(
    std::uint64_t point,
    const stage2_detail::SemanticCoordinateV1& coordinate
) const {
    using namespace stage2_detail;
    if (coordinate.variant_membership ==
            VariantMembershipV1::logical_half_open_range)
        return coordinate.variant_begin == coordinate.variant_end
            ? point == coordinate.variant_begin
            : point >= coordinate.variant_begin &&
                point < coordinate.variant_end;
    const bool grouped = coordinate.variant_membership ==
            VariantMembershipV1::group_execution_slice ||
        coordinate.variant_membership ==
            VariantMembershipV1::strict_group_execution_slice;
    const bool strict = coordinate.variant_membership ==
            VariantMembershipV1::strict_logical_range ||
        coordinate.variant_membership ==
            VariantMembershipV1::strict_group_execution_slice;
    if (coordinate.variant_membership_source_begin >=
            coordinate.variant_membership_source_end ||
        coordinate.variant_membership_source_end > m_)
        return false;
    for (std::size_t source = coordinate.variant_membership_source_begin;
         source < coordinate.variant_membership_source_end; ++source) {
        const std::size_t logical = grouped ? static_cast<std::size_t>(
            group_execution_order_[source]
        ) : source;
        if (logical != point) continue;
        return !strict || (
            coordinate.annotation_begin < k_ &&
            strict_annotation_ids_[logical] == coordinate.annotation_begin
        );
    }
    return false;
}

bool ContextualReferenceExecutorV1::fault_matches(
    const stage2_detail::SemanticCoordinateV1& coordinate
) {
    using namespace stage2_detail;
    if (fault_mode_ == "none" || fault_consumed_ ||
        operation_index(fault_operation_) !=
            static_cast<std::size_t>(coordinate.operation))
        return false;
    if (fault_selector_valid_) {
        const auto contains = [](std::uint64_t point, std::uint64_t begin,
                                 std::uint64_t end) {
            return begin == end ? point == begin : point >= begin && point < end;
        };
        if (!contains(fault_selector_.resident_begin,
                      coordinate.resident_begin, coordinate.resident_end) ||
            !contains(fault_selector_.probe_begin,
                      coordinate.probe_begin, coordinate.probe_end) ||
            !variant_selector_matches(
                fault_selector_.variant_begin, coordinate) ||
            !contains(fault_selector_.annotation_begin,
                      coordinate.annotation_begin, coordinate.annotation_end) ||
            !contains(fault_selector_.context_begin,
                      coordinate.context_begin, coordinate.context_end) ||
            !contains(fault_selector_.action_begin,
                      coordinate.action_begin, coordinate.action_end) ||
            !contains(fault_selector_.group_begin,
                      coordinate.group_begin, coordinate.group_end))
            return false;
        if (coordinate.operation ==
                ContextualDenseSemanticV1::direct_grouped_tn &&
            (fault_selector_.variant_begin >= m_ ||
             static_cast<std::uint64_t>(group_index_[
                 static_cast<std::size_t>(fault_selector_.variant_begin)]) !=
                 fault_selector_.group_begin))
            return false;
        const SemanticPhaseV1 phase = coordinate.phase ==
                SemanticPhaseV1::unspecified
            ? semantic_phase_for_operation(coordinate.operation)
            : coordinate.phase;
        if ((fault_phase_selector_valid_ &&
             fault_selector_.phase != phase) ||
            (fault_role_selector_valid_ &&
             fault_selector_.role != coordinate.role) ||
            (fault_placement_selector_valid_ &&
             fault_selector_.placement != coordinate.placement) ||
            (fault_canonical_selector_valid_ &&
             !contains(fault_selector_.canonical_begin,
                       coordinate.canonical_begin,
                       coordinate.canonical_end)))
            return false;
    }
    fault_consumed_ = true;
    return true;
}

void ContextualReferenceExecutorV1::protected_tn(
    const stage2_detail::SemanticCoordinateV1& coordinate,
    int rows, int columns, int reduction,
    const double* left, int left_stride,
    const double* right, int right_stride,
    double* output, int output_stride
) {
    protected_gemm(true, coordinate, rows, columns, reduction,
                   left, left_stride, right, right_stride,
                   output, output_stride);
}

void ContextualReferenceExecutorV1::protected_nn(
    const stage2_detail::SemanticCoordinateV1& coordinate,
    int rows, int columns, int reduction,
    const double* left, int left_stride,
    const double* right, int right_stride,
    double* output, int output_stride
) {
    protected_gemm(false, coordinate, rows, columns, reduction,
                   left, left_stride, right, right_stride,
                   output, output_stride);
}

void ContextualReferenceExecutorV1::protected_gemm(
    bool transpose_left,
    const stage2_detail::SemanticCoordinateV1& coordinate,
    int rows, int columns, int reduction,
    const double* left, int left_stride,
    const double* right, int right_stride,
    double* output, int output_stride
) {
    using namespace stage2_detail;
    const auto protected_begin = std::chrono::steady_clock::now();
    std::uint64_t operation_retry_calls = 0;
    std::uint64_t operation_fallback_calls = 0;
    std::uint64_t operation_recovery_wall_ns = 0;
    std::unique_lock<std::mutex> dispatcher_lock(contextual_dispatcher_mutex());
    ContextualDispatcherActivityV1 dispatcher_activity;
    if (rows <= 0 || columns <= 0 || reduction <= 0 ||
        output_stride < rows || right_stride < reduction ||
        (transpose_left ? left_stride < reduction : left_stride < rows))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 protected operation shape is invalid"
        );
    const std::size_t count = checked_size(
        checked_mul(rows, columns, "protected output elements"),
        "protected output elements"
    );
    constexpr std::size_t prefix = kProtectedPrefixCanariesV1;
    constexpr std::size_t suffix = kProtectedSuffixCanariesV1;
    const std::size_t protected_count = count + prefix + suffix;
    if (protected_count > primary_buffer_.size())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 protected output exceeds admitted reserve"
        );
    for (std::vector<double>* buffer : {
            &primary_buffer_, &witness_buffer_})
        std::fill(buffer->begin(), buffer->begin() + protected_count, kCanaryV1);
    double* const primary = primary_buffer_.data() + prefix;
    double* const retry = retry_buffer_.data() + prefix;
    double* const witness = witness_buffer_.data() + prefix;
    double* const fallback = fallback_buffer_.data() + prefix;
    const auto operand_fingerprint = [&]() {
        std::uint64_t hash = UINT64_C(1469598103934665603);
        const auto update = [&](double value) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
            for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
                hash ^= bytes[byte];
                hash *= UINT64_C(1099511628211);
            }
        };
        for (int row = 0; row < rows; ++row)
            for (int inner = 0; inner < reduction; ++inner)
                update(transpose_left
                    ? left[static_cast<std::size_t>(inner) +
                           static_cast<std::size_t>(row) * left_stride]
                    : left[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(inner) * left_stride]);
        for (int column = 0; column < columns; ++column)
            for (int inner = 0; inner < reduction; ++inner)
                update(right[static_cast<std::size_t>(inner) +
                             static_cast<std::size_t>(column) * right_stride]);
        return hash;
    };
    const std::uint64_t sealed_operands = operand_fingerprint();
    const std::uint64_t sealed_runtime = runtime_fingerprint();
    const auto run_primary = [&](double* destination) {
        const auto begin = std::chrono::steady_clock::now();
        if (transpose_left)
            dgemm_tn_tiled(rows, columns, reduction, left, left_stride,
                           right, right_stride, destination, rows,
                           blas_threads_);
        else
            dgemm_nn_tiled(rows, columns, reduction, left, left_stride,
                           right, right_stride, destination, rows,
                           blas_threads_);
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin
            ).count()
        );
    };
    const std::uint64_t primary_elapsed = run_primary(primary);
    const auto witness_begin = std::chrono::steady_clock::now();
    scalar_gemm(transpose_left, rows, columns, reduction, left, left_stride,
                right, right_stride, witness);
    const std::uint64_t witness_elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - witness_begin
        ).count()
    );
    const auto canaries = [&](const std::vector<double>& candidate) {
        bool prefix_ok = true;
        bool suffix_ok = true;
        for (std::size_t index = 0; index < prefix; ++index)
            prefix_ok = prefix_ok && candidate[index] == kCanaryV1;
        for (std::size_t index = prefix + count;
             index < protected_count; ++index)
            suffix_ok = suffix_ok && candidate[index] == kCanaryV1;
        return std::pair<bool, bool>{prefix_ok, suffix_ok};
    };
    const auto finite = [&](const double* candidate) {
        return std::all_of(candidate, candidate + count, [](double value) {
            return std::isfinite(value);
        });
    };
    const auto primary_canaries = canaries(primary_buffer_);
    const auto witness_canaries = canaries(witness_buffer_);
    const std::uint64_t runtime_after_primary = runtime_fingerprint();
    EventEvidenceV1 primary_evidence;
    primary_evidence.elapsed_ns = primary_elapsed;
    primary_evidence.operand_fingerprint = sealed_operands;
    primary_evidence.witness_fingerprint = fingerprint_doubles(witness, count);
    primary_evidence.accepted_output_fingerprint =
        fingerprint_doubles(primary, count);
    primary_evidence.runtime_before_fingerprint = sealed_runtime;
    primary_evidence.runtime_after_fingerprint = runtime_after_primary;
    primary_evidence.prefix_canary_verified = primary_canaries.first &&
        witness_canaries.first;
    primary_evidence.suffix_canary_verified = primary_canaries.second &&
        witness_canaries.second;
    primary_evidence.finiteness_verified = finite(primary) && finite(witness);
    primary_evidence.serialized_entry_verified =
        dispatcher_activity.singleton_process_entry_observed();
    primary_evidence.deterministic_non_vendor_backend = true;
    ++observed_calls_[static_cast<std::size_t>(coordinate.operation)];
    record_event(EventClassV1::protected_call, coordinate,
                 "primary_deterministic_tiled_fp64", 1, rows, columns,
                 reduction, left_stride, right_stride, output_stride,
                 transpose_left, primary_evidence);

    const auto agrees = [&](const std::vector<double>& candidate) {
        const double* values = candidate.data() + prefix;
        const auto guard = canaries(candidate);
        if (!guard.first || !guard.second) return false;
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(values[index]) ||
                !std::isfinite(witness[index])) return false;
            const double tolerance = 4.0e-12 *
                (1.0 + std::abs(witness[index]));
            if (std::abs(values[index] - witness[index]) > tolerance)
                return false;
        }
        return true;
    };
    const auto measured_evidence = [&](
        const std::vector<double>& candidate, std::uint64_t elapsed
    ) {
        const auto guard = canaries(candidate);
        const double* values = candidate.data() + prefix;
        EventEvidenceV1 evidence = primary_evidence;
        evidence.elapsed_ns = elapsed;
        evidence.operand_fingerprint = operand_fingerprint();
        evidence.accepted_output_fingerprint =
            fingerprint_doubles(values, count);
        evidence.runtime_after_fingerprint = runtime_fingerprint();
        evidence.prefix_canary_verified = guard.first && witness_canaries.first;
        evidence.suffix_canary_verified = guard.second && witness_canaries.second;
        evidence.finiteness_verified = finite(values) && finite(witness);
        return evidence;
    };

    const bool inject = fault_matches(coordinate);
#if defined(__linux__)
    cpu_set_t saved_runtime_affinity;
    CPU_ZERO(&saved_runtime_affinity);
    bool runtime_affinity_mutated = false;
#endif
    if (inject) {
        ++injection_count_;
        if (fault_mode_ == "operand_mutation")
            const_cast<double*>(right)[0] = std::nextafter(
                right[0], std::numeric_limits<double>::infinity()
            );
        if (fault_mode_ == "runtime_mutation") {
#if defined(__linux__)
            if (::sched_getaffinity(
                    0, sizeof(saved_runtime_affinity),
                    &saved_runtime_affinity) != 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 cannot capture affinity for runtime mutation test"
                );
            int first_cpu = -1;
            std::size_t cpu_count = 0;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (!CPU_ISSET(cpu, &saved_runtime_affinity)) continue;
                if (first_cpu < 0) first_cpu = cpu;
                ++cpu_count;
            }
            if (cpu_count < 2U || first_cpu < 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 actual runtime affinity mutation requires at least two admitted CPUs"
                );
            cpu_set_t restricted_affinity;
            CPU_ZERO(&restricted_affinity);
            CPU_SET(first_cpu, &restricted_affinity);
            if (::sched_setaffinity(
                    0, sizeof(restricted_affinity),
                    &restricted_affinity) != 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 actual runtime affinity mutation failed"
                );
            runtime_affinity_mutated = true;
#else
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 actual runtime affinity mutation is unavailable on this platform"
            );
#endif
        }
        if (fault_mode_ == "nan")
            primary[0] = std::numeric_limits<double>::quiet_NaN();
        else if (fault_mode_ == "inf")
            primary[0] = std::numeric_limits<double>::infinity();
        else if (fault_mode_ == "canary")
            primary_buffer_[0] = 0.0;
        else if (fault_mode_ != "force_fallback" &&
                 fault_mode_ != "fallback_failure" &&
                 fault_mode_ != "fallback_corruption")
            primary[0] += 1.0;
        const EventEvidenceV1 injected_evidence = measured_evidence(
            primary_buffer_, primary_elapsed
        );
#if defined(__linux__)
        if (runtime_affinity_mutated && ::sched_setaffinity(
                0, sizeof(saved_runtime_affinity),
                &saved_runtime_affinity) != 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 actual runtime affinity restoration failed"
            );
#endif
        record_event(EventClassV1::fault_injection, coordinate,
                     fault_mode_.c_str(), 1, rows, columns, reduction,
                     left_stride, right_stride, output_stride,
                     transpose_left, injected_evidence);
        const char* detection = "scalar_witness_mismatch";
        if (injected_evidence.operand_fingerprint != sealed_operands)
            detection = "operand_fingerprint_mismatch";
        else if (injected_evidence.runtime_after_fingerprint != sealed_runtime)
            detection = "runtime_fingerprint_mismatch";
        else if (!injected_evidence.prefix_canary_verified)
            detection = "prefix_canary_mismatch";
        else if (!injected_evidence.suffix_canary_verified)
            detection = "suffix_canary_mismatch";
        else if (!injected_evidence.finiteness_verified)
            detection = "nonfinite_output";
        else if (fault_mode_ == "force_fallback" ||
                 fault_mode_ == "fallback_failure" ||
                 fault_mode_ == "fallback_corruption")
            detection = "forced_fallback_policy";
        record_event(EventClassV1::fault_detection, coordinate,
                     detection, 1, rows, columns, reduction,
                     left_stride, right_stride, output_stride,
                     transpose_left, injected_evidence);
        if (fault_mode_ == "runtime_mutation") {
            if (injected_evidence.runtime_after_fingerprint == sealed_runtime)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 actual runtime affinity mutation was not observed"
                );
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 authenticated runtime affinity mutation detected and restored"
            );
        }
    }

    if (operand_fingerprint() != sealed_operands)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 protected operand mutation detected"
        );
    if (runtime_fingerprint() != sealed_runtime)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 authenticated runtime thread/affinity mutation detected"
        );

    const bool force_fallback = inject &&
        (fault_mode_ == "force_fallback" ||
         fault_mode_ == "fallback_failure" ||
         fault_mode_ == "fallback_corruption");
    const double* accepted = nullptr;
    std::uint32_t accepted_attempt = 0;
    if (!force_fallback && agrees(primary_buffer_)) {
        accepted = primary;
        accepted_attempt = 1;
    } else if (!force_fallback) {
        ++retry_count_;
        ++operation_retry_calls;
        std::fill(
            retry_buffer_.begin(),
            retry_buffer_.begin() + protected_count,
            kCanaryV1
        );
        const std::uint64_t retry_elapsed = run_primary(retry);
        operation_recovery_wall_ns = saturating_add_u64(
            operation_recovery_wall_ns, retry_elapsed
        );
        if (inject && (fault_mode_ == "repeated" ||
                       fault_mode_ == "repair_corruption"))
            retry[0] += 1.0;
        const EventEvidenceV1 retry_evidence = measured_evidence(
            retry_buffer_, retry_elapsed
        );
        record_event(EventClassV1::retry, coordinate,
                     "deterministic_tiled_retry", 2, rows, columns,
                     reduction, left_stride, right_stride, output_stride,
                     transpose_left, retry_evidence);
        if (inject && (fault_mode_ == "repeated" ||
                       fault_mode_ == "repair_corruption"))
            record_event(EventClassV1::fault_detection, coordinate,
                         "retry_scalar_witness_mismatch", 2,
                         rows, columns, reduction, left_stride, right_stride,
                         output_stride, transpose_left, retry_evidence);
        if (agrees(retry_buffer_)) {
            ++repair_count_;
            record_event(EventClassV1::repair, coordinate,
                         "verified_retry", 2, rows, columns, reduction,
                         left_stride, right_stride, output_stride,
                         transpose_left, retry_evidence);
            accepted = retry;
            accepted_attempt = 2;
        }
    }
    if (accepted == nullptr) {
        ++fallback_count_;
        ++operation_fallback_calls;
        std::fill(
            fallback_buffer_.begin(),
            fallback_buffer_.begin() + protected_count,
            kCanaryV1
        );
        const auto fallback_begin = std::chrono::steady_clock::now();
        scalar_fallback_gemm(transpose_left, rows, columns, reduction, left,
                             left_stride, right, right_stride,
                             fallback);
        const std::uint64_t fallback_elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - fallback_begin
            ).count()
        );
        operation_recovery_wall_ns = saturating_add_u64(
            operation_recovery_wall_ns, fallback_elapsed
        );
        if (inject && fault_mode_ == "fallback_corruption")
            fallback[0] += 1.0;
        const EventEvidenceV1 fallback_evidence = measured_evidence(
            fallback_buffer_, fallback_elapsed
        );
        record_event(EventClassV1::trusted_fallback, coordinate,
                     "one_thread_scalar_fp64", 3, rows, columns, reduction,
                     left_stride, right_stride, output_stride,
                     transpose_left, fallback_evidence);
        if (inject && fault_mode_ == "fallback_corruption")
            record_event(EventClassV1::fault_detection, coordinate,
                         "fallback_scalar_witness_mismatch", 3,
                         rows, columns, reduction, left_stride, right_stride,
                         output_stride, transpose_left, fallback_evidence);
        if (inject && fault_mode_ == "fallback_failure") {
            record_event(EventClassV1::fault_detection, coordinate,
                         "injected_trusted_fallback_failure", 3,
                         rows, columns, reduction, left_stride, right_stride,
                         output_stride, transpose_left, fallback_evidence);
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 trusted fallback failure"
            );
        }
        if (!agrees(fallback_buffer_))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 trusted fallback verification failed"
            );
        accepted = fallback;
        accepted_attempt = 3;
    }
    if (operand_fingerprint() != sealed_operands ||
        runtime_fingerprint() != sealed_runtime)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 protected operand/runtime mutation detected"
        );
    for (int column = 0; column < columns; ++column) {
        std::copy_n(
            accepted + static_cast<std::size_t>(column) * rows,
            rows,
            output + static_cast<std::size_t>(column) * output_stride
        );
    }
    const auto accepted_buffer = accepted_attempt == 1 ? &primary_buffer_ :
        (accepted_attempt == 2 ? &retry_buffer_ : &fallback_buffer_);
    const auto accepted_canaries = canaries(*accepted_buffer);
    EventEvidenceV1 accepted_evidence = primary_evidence;
    accepted_evidence.elapsed_ns = primary_elapsed + witness_elapsed;
    accepted_evidence.accepted_output_fingerprint =
        fingerprint_doubles(accepted, count);
    accepted_evidence.runtime_after_fingerprint = runtime_fingerprint();
    accepted_evidence.prefix_canary_verified = accepted_canaries.first;
    accepted_evidence.suffix_canary_verified = accepted_canaries.second;
    accepted_evidence.finiteness_verified = finite(accepted);
    record_event(EventClassV1::semantic_verification, coordinate,
                 "scalar_witness_agreement", accepted_attempt,
                 rows, columns, reduction, left_stride, right_stride,
                 output_stride, transpose_left, accepted_evidence);
    const std::size_t operation_index_value = static_cast<std::size_t>(
        coordinate.operation
    );
    if (operation_index_value >= performance_operations_.size())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 protected operation identity is invalid"
        );
    auto& performance = performance_operations_[operation_index_value];
    performance.calls = saturating_add_u64(performance.calls, 1);
    performance.logical_flop_count += 2.0 * static_cast<double>(rows) *
        static_cast<double>(columns) * static_cast<double>(reduction);
    performance.output_elements = saturating_add_u64(
        performance.output_elements,
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns)
    );
    performance.primary_wall_ns = saturating_add_u64(
        performance.primary_wall_ns, primary_elapsed
    );
    performance.witness_wall_ns = saturating_add_u64(
        performance.witness_wall_ns, witness_elapsed
    );
    performance.retry_calls = saturating_add_u64(
        performance.retry_calls, operation_retry_calls
    );
    performance.fallback_calls = saturating_add_u64(
        performance.fallback_calls, operation_fallback_calls
    );
    performance.recovery_wall_ns = saturating_add_u64(
        performance.recovery_wall_ns, operation_recovery_wall_ns
    );
    performance.protected_total_wall_ns = saturating_add_u64(
        performance.protected_total_wall_ns,
        elapsed_ns_since(protected_begin)
    );
    const std::uint64_t rows_u64 = static_cast<std::uint64_t>(rows);
    const std::uint64_t columns_u64 = static_cast<std::uint64_t>(columns);
    const std::uint64_t reduction_u64 = static_cast<std::uint64_t>(reduction);
    if (performance.calls == 1) {
        performance.minimum_rows = rows_u64;
        performance.minimum_columns = columns_u64;
        performance.minimum_reduction = reduction_u64;
        performance.transpose_left = transpose_left;
    }
    performance.minimum_rows = std::min(performance.minimum_rows, rows_u64);
    performance.maximum_rows = std::max(performance.maximum_rows, rows_u64);
    performance.minimum_columns = std::min(
        performance.minimum_columns, columns_u64
    );
    performance.maximum_columns = std::max(
        performance.maximum_columns, columns_u64
    );
    performance.minimum_reduction = std::min(
        performance.minimum_reduction, reduction_u64
    );
    performance.maximum_reduction = std::max(
        performance.maximum_reduction, reduction_u64
    );
    if (performance.transpose_left != transpose_left)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 protected operation transpose identity changed"
        );
}

void ContextualReferenceExecutorV1::source_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::source)]
    );
    phase_ = "source";
    state_ = "source_running";
    SemanticCoordinateV1 phase_coordinate{};
    record_event(EventClassV1::phase_transition, phase_coordinate,
                 "source_running", 0);
    for (std::size_t resident_begin = 0; resident_begin < bt_;
         resident_begin += sample_probe_resident_) {
        const std::size_t resident_end = std::min(
            bt_, resident_begin + sample_probe_resident_
        );
        const std::size_t resident_width = resident_end - resident_begin;
        fill_probe_resident(
            resident_begin, resident_width, probe_resident_scratch_
        );
        for (std::size_t local_probe = 0; local_probe < resident_width;
             local_probe += sample_probe_tile_) {
            const std::size_t width = std::min(
                sample_probe_tile_, resident_width - local_probe
            );
            SemanticCoordinateV1 coordinate{};
            coordinate.resident_begin = resident_begin;
            coordinate.resident_end = resident_end;
            coordinate.probe_begin = resident_begin + local_probe;
            coordinate.probe_end = coordinate.probe_begin + width;
            project_panel(
                probe_resident_scratch_.data() + n_ * local_probe,
                width, coordinate,
                ContextualDenseSemanticV1::sample_probe_projection_tn,
                ContextualDenseSemanticV1::sample_probe_projection_nn,
                action_projected_scratch_
            );
            {
                PerformanceCategoryScopeV1 category_scope(
                    performance_categories_[static_cast<std::size_t>(
                        PerformanceCategoryV1::packing)],
                    n_ * width, n_ * width * sizeof(double)
                );
                std::copy_n(
                    action_projected_scratch_.data(), n_ * width,
                    projected_probe_scratch_.data() + n_ * local_probe
                );
            }
        }

        Sha256 missing_digest = begin_missingness_sha256(m_, n_);
        Sha256 bed_content_digest = begin_retained_bed_content_digest();
        for (std::size_t variant_begin = 0; variant_begin < m_;
             variant_begin += variant_block_) {
            const std::size_t variant_end = std::min(
                m_, variant_begin + variant_block_
            );
            const std::size_t variant_width = decode_block(
                variant_begin, variant_end, missing_digest,
                bed_content_digest
            );
            for (std::size_t local_probe = 0; local_probe < resident_width;
                 local_probe += sample_probe_tile_) {
                const std::size_t probe_width = std::min(
                    sample_probe_tile_, resident_width - local_probe
                );
                for (std::size_t context_begin = 0; context_begin < q_;
                     context_begin += context_tile_) {
                    const std::size_t context_end = std::min(
                        q_, context_begin + context_tile_
                    );
                    const std::size_t context_width = context_end - context_begin;
                    const std::size_t columns = context_width * probe_width;
                    {
                        PerformanceCategoryScopeV1 category_scope(
                            performance_categories_[static_cast<std::size_t>(
                                PerformanceCategoryV1::row_scale)],
                            n_ * columns,
                            n_ * columns * sizeof(double) * 3U
                        );
                        for (std::size_t context_local = 0;
                             context_local < context_width; ++context_local) {
                            const std::size_t context = context_begin + context_local;
                            for (std::size_t probe_local = 0;
                                 probe_local < probe_width; ++probe_local) {
                                const std::size_t column =
                                    context_local * probe_width + probe_local;
                                const double* projected = projected_probe_scratch_.data() +
                                    n_ * (local_probe + probe_local);
                                double* packed = source_right_scratch_.data() + n_ * column;
                                const double* phi = context_basis_.data() + n_ * context;
                                for (std::size_t row = 0; row < n_; ++row)
                                    packed[row] = phi[row] * projected[row];
                            }
                        }
                    }
                    SemanticCoordinateV1 coordinate{};
                    coordinate.operation = ContextualDenseSemanticV1::source_tn;
                    coordinate.resident_begin = resident_begin;
                    coordinate.resident_end = resident_end;
                    coordinate.probe_begin = resident_begin + local_probe;
                    coordinate.probe_end = coordinate.probe_begin + probe_width;
                    coordinate.variant_begin = variant_begin;
                    coordinate.variant_end = variant_end;
                    coordinate.context_begin = context_begin;
                    coordinate.context_end = context_end;
                    protected_tn(
                        coordinate,
                        checked_blas_dim(variant_width, "source block rows"),
                        checked_blas_dim(columns, "source columns"),
                        checked_blas_dim(n_, "source reduction"),
                        decoded_scratch_.data(), checked_blas_dim(n_, "decoded stride"),
                        source_right_scratch_.data(), checked_blas_dim(n_, "source RHS stride"),
                        target_update_scratch_.data(),
                        checked_blas_dim(variant_width, "source output stride")
                    );
                    {
                        PerformanceCategoryScopeV1 category_scope(
                            performance_categories_[static_cast<std::size_t>(
                                PerformanceCategoryV1::packing)],
                            variant_width * columns,
                            variant_width * columns * sizeof(double)
                        );
                        for (std::size_t context_local = 0;
                             context_local < context_width; ++context_local) {
                            const std::size_t context = context_begin + context_local;
                            for (std::size_t probe_local = 0;
                                 probe_local < probe_width; ++probe_local) {
                                const std::size_t probe = resident_begin +
                                    local_probe + probe_local;
                                const std::size_t column =
                                    context_local * probe_width + probe_local;
                                for (std::size_t variant_local = 0;
                                     variant_local < variant_width; ++variant_local)
                                    source_scores_[source_index(
                                        variant_begin + variant_local, context, probe
                                    )] = target_update_scratch_[
                                        variant_local + variant_width * column
                                    ];
                            }
                        }
                    }
                }
            }
        }
        ++observed_descriptor_passes_;
        finish_retained_bed_content_digest(bed_content_digest);
        const std::string digest = missing_digest.finish();
        if (digest != expected_missingness_sha256_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 missingness SHA256 mismatch"
            );
        if (observed_missingness_sha256_.empty())
            observed_missingness_sha256_ = digest;
        else if (observed_missingness_sha256_ != digest)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 missingness changed between descriptor passes"
            );
    }
    state_ = "source_complete";
    source_fingerprint_ = fingerprint_source();
    checkpoint("post_source");
    if (fingerprint_source() != source_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 source output mutation detected"
        );
}

void ContextualReferenceExecutorV1::action_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::action)]
    );
    if (fingerprint_source() != source_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 sealed source identity changed before action"
        );
    phase_ = "action";
    state_ = "action_running";
    SemanticCoordinateV1 phase_coordinate{};
    record_event(EventClassV1::phase_transition, phase_coordinate,
                 "action_running", 0);
    for (std::size_t resident_begin = 0; resident_begin < bt_;
         resident_begin += sample_probe_resident_) {
        const std::size_t resident_end = std::min(
            bt_, resident_begin + sample_probe_resident_
        );
        const std::size_t resident_width = resident_end - resident_begin;
        const std::size_t target_elements = n_ * k_ * q_ * resident_width;
        std::fill_n(target_panel_scratch_.data(), target_elements, 0.0);
        Sha256 missing_digest = begin_missingness_sha256(m_, n_);
        Sha256 bed_content_digest = begin_retained_bed_content_digest();
        for (std::size_t variant_begin = 0; variant_begin < m_;
             variant_begin += variant_block_) {
            const std::size_t variant_end = std::min(
                m_, variant_begin + variant_block_
            );
            const std::size_t variant_width = decode_block(
                variant_begin, variant_end, missing_digest,
                bed_content_digest
            );
            for (std::size_t local_probe = 0; local_probe < resident_width;
                 local_probe += sample_probe_tile_) {
                const std::size_t probe_width = std::min(
                    sample_probe_tile_, resident_width - local_probe
                );
                for (std::size_t annotation_begin = 0; annotation_begin < k_;
                     annotation_begin += annotation_tile_) {
                    const std::size_t annotation_end = std::min(
                        k_, annotation_begin + annotation_tile_
                    );
                    for (std::size_t context_begin = 0; context_begin < q_;
                         context_begin += context_tile_) {
                        const std::size_t context_end = std::min(
                            q_, context_begin + context_tile_
                        );
                        const std::size_t context_width = context_end - context_begin;
                        if (strict_annotation_ids_.empty()) {
                            const std::size_t annotation_width =
                                annotation_end - annotation_begin;
                            const std::size_t columns = annotation_width *
                                context_width * probe_width;
                            PerformanceCategoryScopeV1 row_scale_scope(
                                performance_categories_[static_cast<std::size_t>(
                                    PerformanceCategoryV1::row_scale)],
                                variant_width * columns,
                                variant_width * columns * sizeof(double) * 3U
                            );
                            for (std::size_t annotation_local = 0;
                                 annotation_local < annotation_width;
                                 ++annotation_local) {
                                const std::size_t annotation =
                                    annotation_begin + annotation_local;
                                for (std::size_t context_local = 0;
                                     context_local < context_width;
                                     ++context_local) {
                                    const std::size_t context =
                                        context_begin + context_local;
                                    for (std::size_t probe_local = 0;
                                         probe_local < probe_width;
                                         ++probe_local) {
                                        const std::size_t probe = resident_begin +
                                            local_probe + probe_local;
                                        const std::size_t column =
                                            (annotation_local * context_width +
                                             context_local) * probe_width +
                                            probe_local;
                                        double* packed = target_right_scratch_.data() +
                                            variant_width * column;
                                        for (std::size_t variant_local = 0;
                                             variant_local < variant_width;
                                             ++variant_local) {
                                            const std::size_t variant =
                                                variant_begin + variant_local;
                                            packed[variant_local] =
                                                annotation_weights_[variant + m_ * annotation] *
                                                source_scores_[source_index(
                                                    variant, context, probe
                                                )];
                                        }
                                    }
                                }
                            }
                            row_scale_scope.finish();
                            SemanticCoordinateV1 coordinate{};
                            coordinate.operation = ContextualDenseSemanticV1::full_target_nn;
                            coordinate.resident_begin = resident_begin;
                            coordinate.resident_end = resident_end;
                            coordinate.probe_begin = resident_begin + local_probe;
                            coordinate.probe_end = coordinate.probe_begin + probe_width;
                            coordinate.variant_begin = variant_begin;
                            coordinate.variant_end = variant_end;
                            coordinate.annotation_begin = annotation_begin;
                            coordinate.annotation_end = annotation_end;
                            coordinate.context_begin = context_begin;
                            coordinate.context_end = context_end;
                            protected_nn(
                                coordinate,
                                checked_blas_dim(n_, "target rows"),
                                checked_blas_dim(columns, "target columns"),
                                checked_blas_dim(variant_width, "target reduction"),
                                decoded_scratch_.data(), checked_blas_dim(n_, "decoded stride"),
                                target_right_scratch_.data(),
                                checked_blas_dim(variant_width, "target RHS stride"),
                                target_update_scratch_.data(),
                                checked_blas_dim(n_, "target output stride")
                            );
                            PerformanceCategoryScopeV1 reduction_scope(
                                performance_categories_[static_cast<std::size_t>(
                                    PerformanceCategoryV1::reduction)],
                                n_ * columns,
                                n_ * columns * sizeof(double) * 2U
                            );
                            for (std::size_t annotation_local = 0;
                                 annotation_local < annotation_width;
                                 ++annotation_local) {
                                const std::size_t annotation =
                                    annotation_begin + annotation_local;
                                for (std::size_t context_local = 0;
                                     context_local < context_width;
                                     ++context_local) {
                                    const std::size_t context =
                                        context_begin + context_local;
                                    for (std::size_t probe_local = 0;
                                         probe_local < probe_width;
                                         ++probe_local) {
                                        const std::size_t h_column =
                                            (annotation * q_ + context) *
                                            resident_width + local_probe + probe_local;
                                        const std::size_t update_column =
                                            (annotation_local * context_width +
                                             context_local) * probe_width + probe_local;
                                        for (std::size_t row = 0; row < n_; ++row)
                                            target_panel_scratch_[row + n_ * h_column] +=
                                                target_update_scratch_[row + n_ * update_column];
                                    }
                                }
                            }
                        } else {
                            for (std::size_t annotation = annotation_begin;
                                 annotation < annotation_end; ++annotation) {
                                PerformanceCategoryScopeV1 packing_scope(
                                    performance_categories_[static_cast<std::size_t>(
                                        PerformanceCategoryV1::packing)]
                                );
                                std::size_t selected = 0;
                                for (std::size_t variant_local = 0;
                                     variant_local < variant_width; ++variant_local) {
                                    const std::size_t variant = variant_begin + variant_local;
                                    if (strict_annotation_ids_[variant] != annotation)
                                        continue;
                                    std::copy_n(
                                        decoded_scratch_.data() + n_ * variant_local,
                                        n_, strict_genotype_scratch_.data() + n_ * selected
                                    );
                                    for (std::size_t context_local = 0;
                                         context_local < context_width; ++context_local) {
                                        const std::size_t context = context_begin + context_local;
                                        for (std::size_t probe_local = 0;
                                             probe_local < probe_width; ++probe_local) {
                                            const std::size_t probe = resident_begin +
                                                local_probe + probe_local;
                                            const std::size_t column =
                                                context_local * probe_width + probe_local;
                                            target_right_scratch_[selected +
                                                variant_width * column] =
                                                source_scores_[source_index(
                                                    variant, context, probe
                                                )];
                                        }
                                    }
                                    ++selected;
                                }
                                if (selected == 0) continue;
                                // Compact right columns from the admitted
                                // variant-width stride to the selected stride.
                                for (std::size_t column = 1;
                                     column < context_width * probe_width;
                                     ++column)
                                    std::memmove(
                                        target_right_scratch_.data() + selected * column,
                                        target_right_scratch_.data() + variant_width * column,
                                        selected * sizeof(double)
                                    );
                                SemanticCoordinateV1 coordinate{};
                                coordinate.operation = ContextualDenseSemanticV1::full_target_nn;
                                coordinate.resident_begin = resident_begin;
                                coordinate.resident_end = resident_end;
                                coordinate.probe_begin = resident_begin + local_probe;
                                coordinate.probe_end = coordinate.probe_begin + probe_width;
                                coordinate.variant_begin = variant_begin;
                                coordinate.variant_end = variant_end;
                                coordinate.annotation_begin = annotation;
                                coordinate.annotation_end = annotation + 1U;
                                coordinate.context_begin = context_begin;
                                coordinate.context_end = context_end;
                                bind_variant_membership(
                                    coordinate,
                                    VariantMembershipV1::strict_logical_range,
                                    variant_begin, variant_end, annotation
                                );
                                const std::size_t columns = context_width * probe_width;
                                packing_scope.finish();
                                protected_nn(
                                    coordinate,
                                    checked_blas_dim(n_, "strict target rows"),
                                    checked_blas_dim(columns, "strict target columns"),
                                    checked_blas_dim(selected, "strict target reduction"),
                                    strict_genotype_scratch_.data(), checked_blas_dim(n_, "strict genotype stride"),
                                    target_right_scratch_.data(), checked_blas_dim(selected, "strict RHS stride"),
                                    target_update_scratch_.data(), checked_blas_dim(n_, "strict output stride")
                                );
                                PerformanceCategoryScopeV1 reduction_scope(
                                    performance_categories_[static_cast<std::size_t>(
                                        PerformanceCategoryV1::reduction)],
                                    n_ * columns,
                                    n_ * columns * sizeof(double) * 2U
                                );
                                for (std::size_t context_local = 0;
                                     context_local < context_width; ++context_local) {
                                    const std::size_t context = context_begin + context_local;
                                    for (std::size_t probe_local = 0;
                                         probe_local < probe_width; ++probe_local) {
                                        const std::size_t h_column =
                                            (annotation * q_ + context) * resident_width +
                                            local_probe + probe_local;
                                        const std::size_t update_column =
                                            context_local * probe_width + probe_local;
                                        for (std::size_t row = 0; row < n_; ++row)
                                            target_panel_scratch_[row + n_ * h_column] +=
                                                target_update_scratch_[row + n_ * update_column];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ++observed_descriptor_passes_;
        finish_retained_bed_content_digest(bed_content_digest);
        const std::string digest = missing_digest.finish();
        if (digest != expected_missingness_sha256_ ||
            digest != observed_missingness_sha256_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 missingness changed in action pass"
            );

        for (std::size_t local_probe = 0; local_probe < resident_width;
             local_probe += sample_probe_tile_) {
            const std::size_t probe_width = std::min(
                sample_probe_tile_, resident_width - local_probe
            );
            for (std::size_t action_begin = 0; action_begin < components_;
                 action_begin += action_tile_) {
                const std::size_t action_end = std::min(
                    components_, action_begin + action_tile_
                );
                const std::size_t action_width = action_end - action_begin;
                PerformanceCategoryScopeV1 row_scale_scope(
                    performance_categories_[static_cast<std::size_t>(
                        PerformanceCategoryV1::row_scale)],
                    n_ * action_width * probe_width,
                    n_ * action_width * probe_width * sizeof(double) * 4U
                );
                for (std::size_t action_local = 0;
                     action_local < action_width; ++action_local) {
                    const std::size_t component = action_begin + action_local;
                    const std::size_t annotation = static_cast<std::size_t>(
                        component_annotation_[component]
                    );
                    const std::size_t pair = static_cast<std::size_t>(
                        component_pair_[component]
                    );
                    const std::size_t first = static_cast<std::size_t>(pair_q_[pair]);
                    const std::size_t second = static_cast<std::size_t>(pair_r_[pair]);
                    for (std::size_t probe_local = 0;
                         probe_local < probe_width; ++probe_local) {
                        const std::size_t column =
                            action_local * probe_width + probe_local;
                        const std::size_t resident_probe = local_probe + probe_local;
                        const double* h_second = target_panel_scratch_.data() + n_ *
                            ((annotation * q_ + second) * resident_width + resident_probe);
                        const double* phi_first = context_basis_.data() + n_ * first;
                        const double* h_first = target_panel_scratch_.data() + n_ *
                            ((annotation * q_ + first) * resident_width + resident_probe);
                        const double* phi_second = context_basis_.data() + n_ * second;
                        double* packed = action_panel_scratch_.data() + n_ * column;
                        for (std::size_t row = 0; row < n_; ++row) {
                            packed[row] = phi_first[row] * h_second[row];
                            if (first != second)
                                packed[row] += phi_second[row] * h_first[row];
                        }
                    }
                }
                row_scale_scope.finish();
                SemanticCoordinateV1 coordinate{};
                coordinate.resident_begin = resident_begin;
                coordinate.resident_end = resident_end;
                coordinate.probe_begin = resident_begin + local_probe;
                coordinate.probe_end = coordinate.probe_begin + probe_width;
                coordinate.action_begin = action_begin;
                coordinate.action_end = action_end;
                project_panel(
                    action_panel_scratch_.data(), action_width * probe_width,
                    coordinate,
                    ContextualDenseSemanticV1::action_projection_tn,
                    ContextualDenseSemanticV1::action_projection_nn,
                    action_projected_scratch_
                );
                PerformanceCategoryScopeV1 packing_scope(
                    performance_categories_[static_cast<std::size_t>(
                        PerformanceCategoryV1::packing)],
                    n_ * action_width * probe_width,
                    n_ * action_width * probe_width * sizeof(double)
                );
                for (std::size_t action_local = 0;
                     action_local < action_width; ++action_local) {
                    const std::size_t component = action_begin + action_local;
                    for (std::size_t probe_local = 0;
                         probe_local < probe_width; ++probe_local) {
                        const std::size_t probe = resident_begin + local_probe + probe_local;
                        const std::size_t column =
                            action_local * probe_width + probe_local;
                        for (std::size_t row = 0; row < n_; ++row)
                            raw_actions_[action_index(row, component, probe)] =
                                action_projected_scratch_[row + n_ * column];
                    }
                }
            }
        }
    }
    state_ = "action_complete";
    if (fingerprint_source() != source_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 sealed source identity changed during action"
        );
    action_fingerprint_ = fingerprint_source_action();
    checkpoint("post_action");
    if (fingerprint_source_action() != action_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 source/action output mutation detected"
        );
}

void ContextualReferenceExecutorV1::gram_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::gram)]
    );
    if (fingerprint_source() != source_fingerprint_ ||
        fingerprint_source_action() != action_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 sealed source/action identity changed before Gram"
        );
    phase_ = "gram";
    state_ = "gram_running";
    SemanticCoordinateV1 phase_coordinate{};
    record_event(EventClassV1::phase_transition, phase_coordinate,
                 "gram_running", 0);
    std::fill(raw_gram_numerator_.begin(), raw_gram_numerator_.end(), 0.0);
    for (std::size_t resident_begin = 0; resident_begin < bt_;
         resident_begin += sample_probe_resident_) {
        const std::size_t resident_end = std::min(
            bt_, resident_begin + sample_probe_resident_
        );
        const std::size_t resident_width = resident_end - resident_begin;
        const std::size_t reduction = n_ * resident_width;
        PerformanceCategoryScopeV1 packing_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::packing)],
            reduction * components_, reduction * components_ * sizeof(double)
        );
        for (std::size_t component = 0; component < components_; ++component) {
            double* column = gram_panel_scratch_.data() + reduction * component;
            for (std::size_t local_probe = 0; local_probe < resident_width;
                 ++local_probe) {
                const std::size_t probe = resident_begin + local_probe;
                std::copy_n(
                    raw_actions_.data() + action_index(0, component, probe),
                    n_, column + n_ * local_probe
                );
            }
        }
        packing_scope.finish();
        SemanticCoordinateV1 coordinate{};
        coordinate.operation = ContextualDenseSemanticV1::action_gram_tn;
        coordinate.resident_begin = resident_begin;
        coordinate.resident_end = resident_end;
        coordinate.probe_begin = resident_begin;
        coordinate.probe_end = resident_end;
        coordinate.action_begin = 0;
        coordinate.action_end = components_;
        protected_tn(
            coordinate,
            checked_blas_dim(components_, "Gram rows"),
            checked_blas_dim(components_, "Gram columns"),
            checked_blas_dim(reduction, "Gram reduction"),
            gram_panel_scratch_.data(), checked_blas_dim(reduction, "Gram panel stride"),
            gram_panel_scratch_.data(), checked_blas_dim(reduction, "Gram panel stride"),
            gram_update_scratch_.data(), checked_blas_dim(components_, "Gram output stride")
        );
        PerformanceCategoryScopeV1 reduction_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::reduction)],
            components_ * components_,
            components_ * components_ * sizeof(double) * 2U
        );
        for (std::size_t index = 0; index < components_ * components_; ++index)
            raw_gram_numerator_[index] +=
                gram_update_scratch_[index] / static_cast<double>(bt_);
    }
    for (std::size_t column = 0; column < components_; ++column) {
        const std::size_t annotation_column = static_cast<std::size_t>(
            component_annotation_[column]
        );
        for (std::size_t row = 0; row < components_; ++row) {
            const std::size_t annotation_row = static_cast<std::size_t>(
                component_annotation_[row]
            );
            const std::size_t index = row + components_ * column;
            gram_[index] = raw_gram_numerator_[index] /
                (annotation_masses_[annotation_row] *
                 annotation_masses_[annotation_column]);
            if (!std::isfinite(gram_[index]) ||
                !std::isfinite(raw_gram_numerator_[index]))
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 nonfinite Gram output"
                );
        }
    }
    gram_pre_symmetry_max_abs_ = 0.0;
    for (std::size_t column = 0; column < components_; ++column) {
        for (std::size_t row = 0; row < column; ++row) {
            const std::size_t lower = row + components_ * column;
            const std::size_t upper = column + components_ * row;
            gram_pre_symmetry_max_abs_ = std::max(
                gram_pre_symmetry_max_abs_,
                std::abs(gram_[lower] - gram_[upper])
            );
            const double gram_average = 0.5 * (gram_[lower] + gram_[upper]);
            const double raw_average = 0.5 * (
                raw_gram_numerator_[lower] + raw_gram_numerator_[upper]
            );
            gram_[lower] = gram_[upper] = gram_average;
            raw_gram_numerator_[lower] =
                raw_gram_numerator_[upper] = raw_average;
        }
    }
    if (fingerprint_source() != source_fingerprint_ ||
        fingerprint_source_action() != action_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 sealed source/action identity changed during Gram"
        );
    state_ = "gram_complete";
    const auto gram_seal = [&]() {
        Sha256 digest;
        sha_update_string(digest, "contextual_post_gram_phase_v1");
        for (const std::string& value : {
                array_sha256_f64_c_order(
                    gram_, {components_, components_}),
                array_sha256_f64_c_order(
                    raw_gram_numerator_, {components_, components_}),
                std::to_string(source_fingerprint_),
                std::to_string(action_fingerprint_)}) {
            sha_update_u64_le(digest, value.size());
            sha_update(digest, value.data(), value.size());
        }
        return digest.finish();
    };
    gram_phase_sha256_ = gram_seal();
    checkpoint("post_gram");
    if (gram_seal() != gram_phase_sha256_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 post-Gram phase seal mutation detected"
        );
}

void ContextualReferenceExecutorV1::group_restricted_phase(
    std::vector<double>& destination
) {
    using namespace stage2_detail;
    std::fill(destination.begin(), destination.end(), 0.0);
    const std::size_t target_stride = elements_for(
        {n_, k_, q_, bt_}, "group target stride"
    );
    std::size_t execution_cursor = 0;
    for (std::size_t group_begin = 0; group_begin < groups_;
         group_begin += group_tile_) {
        const std::size_t group_end = std::min(
            groups_, group_begin + group_tile_
        );
        const std::size_t group_count = group_end - group_begin;
        std::fill_n(
            group_target_scratch_.data(), group_count * target_stride, 0.0
        );
        std::size_t batch_execution_cursor = execution_cursor;
        for (std::size_t group = group_begin; group < group_end; ++group) {
            const std::size_t local_group = group - group_begin;
            double* group_targets = group_target_scratch_.data() +
                local_group * target_stride;
            const std::size_t group_execution_end = batch_execution_cursor +
                static_cast<std::size_t>(group_variant_counts_[group]);
            for (std::size_t execution_block = batch_execution_cursor;
                 execution_block < group_execution_end;
                 execution_block += variant_block_) {
                const std::size_t execution_end = std::min(
                    group_execution_end, execution_block + variant_block_
                );
                const std::size_t variant_width =
                    decode_group_execution_block(
                        execution_block, execution_end
                    );
                for (std::size_t probe_begin = 0; probe_begin < bt_;
                     probe_begin += sample_probe_tile_) {
                    const std::size_t probe_end = std::min(
                        bt_, probe_begin + sample_probe_tile_
                    );
                    const std::size_t probe_width = probe_end - probe_begin;
                    for (std::size_t annotation_begin = 0;
                         annotation_begin < k_;
                         annotation_begin += annotation_tile_) {
                        const std::size_t annotation_end = std::min(
                            k_, annotation_begin + annotation_tile_
                        );
                        for (std::size_t context_begin = 0;
                             context_begin < q_;
                             context_begin += context_tile_) {
                            const std::size_t context_end = std::min(
                                q_, context_begin + context_tile_
                            );
                            const std::size_t context_width =
                                context_end - context_begin;
                            if (strict_annotation_ids_.empty()) {
                                const std::size_t annotation_width =
                                    annotation_end - annotation_begin;
                                const std::size_t columns = annotation_width *
                                    context_width * probe_width;
                                PerformanceCategoryScopeV1 packing_scope(
                                    performance_categories_[static_cast<std::size_t>(
                                        PerformanceCategoryV1::packing)],
                                    variant_width * columns,
                                    variant_width * columns * sizeof(double) * 3U
                                );
                                for (std::size_t annotation_local = 0;
                                     annotation_local < annotation_width;
                                     ++annotation_local) {
                                    const std::size_t annotation =
                                        annotation_begin + annotation_local;
                                    for (std::size_t context_local = 0;
                                         context_local < context_width;
                                         ++context_local) {
                                        const std::size_t context =
                                            context_begin + context_local;
                                        for (std::size_t probe_local = 0;
                                             probe_local < probe_width;
                                             ++probe_local) {
                                            const std::size_t probe =
                                                probe_begin + probe_local;
                                            const std::size_t column =
                                                (annotation_local * context_width +
                                                 context_local) * probe_width +
                                                probe_local;
                                            double* right =
                                                target_right_scratch_.data() +
                                                variant_width * column;
                                            for (std::size_t local = 0;
                                                 local < variant_width; ++local) {
                                                const std::size_t logical =
                                                    static_cast<std::size_t>(
                                                        group_execution_order_[
                                                            execution_block + local]);
                                                right[local] = annotation_value(
                                                    logical, annotation
                                                ) * source_scores_[source_index(
                                                    logical, context, probe
                                                )];
                                            }
                                        }
                                    }
                                }
                                packing_scope.finish();
                                SemanticCoordinateV1 coordinate{};
                                coordinate.operation =
                                    ContextualDenseSemanticV1::group_target_nn;
                                coordinate.resident_begin = 0;
                                coordinate.resident_end = 1;
                                coordinate.probe_begin = probe_begin;
                                coordinate.probe_end = probe_end;
                                coordinate.variant_begin = execution_block;
                                coordinate.variant_end = execution_end;
                                coordinate.annotation_begin = annotation_begin;
                                coordinate.annotation_end = annotation_end;
                                coordinate.context_begin = context_begin;
                                coordinate.context_end = context_end;
                                coordinate.group_begin = group;
                                coordinate.group_end = group + 1U;
                                bind_variant_membership(
                                    coordinate,
                                    VariantMembershipV1::group_execution_slice,
                                    execution_block, execution_end
                                );
                                protected_nn(
                                    coordinate,
                                    checked_blas_dim(n_, "group target rows"),
                                    checked_blas_dim(columns,
                                                     "group target columns"),
                                    checked_blas_dim(variant_width,
                                                     "group target reduction"),
                                    decoded_scratch_.data(),
                                    checked_blas_dim(n_,
                                                     "group decoded stride"),
                                    target_right_scratch_.data(),
                                    checked_blas_dim(variant_width,
                                                     "group RHS stride"),
                                    target_update_scratch_.data(),
                                    checked_blas_dim(n_,
                                                     "group output stride")
                                );
                                PerformanceCategoryScopeV1 reduction_scope(
                                    performance_categories_[static_cast<std::size_t>(
                                        PerformanceCategoryV1::reduction)],
                                    n_ * columns,
                                    n_ * columns * sizeof(double) * 2U
                                );
                                for (std::size_t annotation_local = 0;
                                     annotation_local < annotation_width;
                                     ++annotation_local) {
                                    const std::size_t annotation =
                                        annotation_begin + annotation_local;
                                    for (std::size_t context_local = 0;
                                         context_local < context_width;
                                         ++context_local) {
                                        const std::size_t context =
                                            context_begin + context_local;
                                        for (std::size_t probe_local = 0;
                                             probe_local < probe_width;
                                             ++probe_local) {
                                            const std::size_t probe =
                                                probe_begin + probe_local;
                                            const std::size_t source_column =
                                                (annotation_local * context_width +
                                                 context_local) * probe_width +
                                                probe_local;
                                            double* target = group_targets + n_ *
                                                ((annotation * q_ + context) * bt_ +
                                                 probe);
                                            const double* update =
                                                target_update_scratch_.data() +
                                                n_ * source_column;
                                            for (std::size_t row = 0; row < n_;
                                                 ++row)
                                                target[row] += update[row];
                                        }
                                    }
                                }
                            } else {
                                for (std::size_t annotation = annotation_begin;
                                     annotation < annotation_end; ++annotation) {
                                    PerformanceCategoryScopeV1 packing_scope(
                                        performance_categories_[static_cast<std::size_t>(
                                            PerformanceCategoryV1::packing)]
                                    );
                                    std::size_t selected = 0;
                                    for (std::size_t local = 0;
                                         local < variant_width; ++local) {
                                        const std::size_t logical =
                                            static_cast<std::size_t>(
                                                group_execution_order_[
                                                    execution_block + local]);
                                        if (strict_annotation_ids_[logical] !=
                                            annotation)
                                            continue;
                                        std::copy_n(
                                            decoded_scratch_.data() + n_ * local,
                                            n_, strict_genotype_scratch_.data() +
                                                n_ * selected
                                        );
                                        for (std::size_t context_local = 0;
                                             context_local < context_width;
                                             ++context_local) {
                                            const std::size_t context =
                                                context_begin + context_local;
                                            for (std::size_t probe_local = 0;
                                                 probe_local < probe_width;
                                                 ++probe_local) {
                                                const std::size_t probe =
                                                    probe_begin + probe_local;
                                                const std::size_t column =
                                                    context_local * probe_width +
                                                    probe_local;
                                                target_right_scratch_[
                                                    selected +
                                                    variant_width * column] =
                                                    source_scores_[source_index(
                                                        logical, context, probe
                                                    )];
                                            }
                                        }
                                        ++selected;
                                    }
                                    if (selected == 0) continue;
                                    for (std::size_t column = 1;
                                         column < context_width * probe_width;
                                         ++column)
                                        std::memmove(
                                            target_right_scratch_.data() +
                                                selected * column,
                                            target_right_scratch_.data() +
                                                variant_width * column,
                                            selected * sizeof(double)
                                        );
                                    SemanticCoordinateV1 coordinate{};
                                    coordinate.operation =
                                        ContextualDenseSemanticV1::group_target_nn;
                                    coordinate.resident_begin = 0;
                                    coordinate.resident_end = 1;
                                    coordinate.probe_begin = probe_begin;
                                    coordinate.probe_end = probe_end;
                                    coordinate.variant_begin = execution_block;
                                    coordinate.variant_end = execution_end;
                                    coordinate.annotation_begin = annotation;
                                    coordinate.annotation_end = annotation + 1U;
                                    coordinate.context_begin = context_begin;
                                    coordinate.context_end = context_end;
                                    coordinate.group_begin = group;
                                    coordinate.group_end = group + 1U;
                                    bind_variant_membership(
                                        coordinate,
                                        VariantMembershipV1::
                                            strict_group_execution_slice,
                                        execution_block, execution_end,
                                        annotation
                                    );
                                    const std::size_t columns =
                                        context_width * probe_width;
                                    packing_scope.finish();
                                    protected_nn(
                                        coordinate,
                                        checked_blas_dim(n_,
                                                         "strict group rows"),
                                        checked_blas_dim(columns,
                                                         "strict group columns"),
                                        checked_blas_dim(selected,
                                                         "strict group reduction"),
                                        strict_genotype_scratch_.data(),
                                        checked_blas_dim(n_,
                                                         "strict group genotype stride"),
                                        target_right_scratch_.data(),
                                        checked_blas_dim(selected,
                                                         "strict group RHS stride"),
                                        target_update_scratch_.data(),
                                        checked_blas_dim(n_,
                                                         "strict group output stride")
                                    );
                                    PerformanceCategoryScopeV1 reduction_scope(
                                        performance_categories_[static_cast<std::size_t>(
                                            PerformanceCategoryV1::reduction)],
                                        n_ * columns,
                                        n_ * columns * sizeof(double) * 2U
                                    );
                                    for (std::size_t context_local = 0;
                                         context_local < context_width;
                                         ++context_local) {
                                        const std::size_t context =
                                            context_begin + context_local;
                                        for (std::size_t probe_local = 0;
                                             probe_local < probe_width;
                                             ++probe_local) {
                                            const std::size_t probe =
                                                probe_begin + probe_local;
                                            const std::size_t column =
                                                context_local * probe_width +
                                                probe_local;
                                            double* target = group_targets + n_ *
                                                ((annotation * q_ + context) * bt_ +
                                                 probe);
                                            const double* update =
                                                target_update_scratch_.data() +
                                                n_ * column;
                                            for (std::size_t row = 0; row < n_;
                                                 ++row)
                                                target[row] += update[row];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            batch_execution_cursor = group_execution_end;
        }

        const std::size_t reduction = n_ * bt_;
        std::size_t cross_execution_cursor = execution_cursor;
        for (std::size_t group = group_begin; group < group_end; ++group) {
            const std::size_t cross_execution_end = cross_execution_cursor +
                static_cast<std::size_t>(group_variant_counts_[group]);
            const std::size_t local_group = group - group_begin;
            const double* group_targets = group_target_scratch_.data() +
                local_group * target_stride;
            for (std::size_t action_begin = 0; action_begin < components_;
                 action_begin += action_tile_) {
                const std::size_t action_end = std::min(
                    components_, action_begin + action_tile_
                );
                const std::size_t action_width = action_end - action_begin;
                PerformanceCategoryScopeV1 row_scale_scope(
                    performance_categories_[static_cast<std::size_t>(
                        PerformanceCategoryV1::row_scale)],
                    n_ * bt_ * action_width,
                    n_ * bt_ * action_width * sizeof(double) * 4U
                );
                for (std::size_t local = 0; local < action_width; ++local) {
                    const std::size_t component = action_begin + local;
                    const std::size_t annotation = static_cast<std::size_t>(
                        component_annotation_[component]
                    );
                    const std::size_t pair = static_cast<std::size_t>(
                        component_pair_[component]
                    );
                    const std::size_t first = static_cast<std::size_t>(
                        pair_q_[pair]
                    );
                    const std::size_t second = static_cast<std::size_t>(
                        pair_r_[pair]
                    );
                    for (std::size_t probe = 0; probe < bt_; ++probe) {
                        double* action = group_action_scratch_.data() +
                            n_ * (probe + bt_ * local);
                        const double* first_target = group_targets + n_ *
                            ((annotation * q_ + first) * bt_ + probe);
                        const double* second_target = group_targets + n_ *
                            ((annotation * q_ + second) * bt_ + probe);
                        const double* first_phi = context_basis_.data() + n_ * first;
                        const double* second_phi = context_basis_.data() + n_ * second;
                        for (std::size_t row = 0; row < n_; ++row) {
                            action[row] = first_phi[row] * second_target[row];
                            if (first != second)
                                action[row] += second_phi[row] * first_target[row];
                        }
                    }
                }
                row_scale_scope.finish();
                SemanticCoordinateV1 coordinate{};
                coordinate.operation =
                    ContextualDenseSemanticV1::group_cross_gram_tn;
                coordinate.resident_begin = 0;
                coordinate.resident_end = 1;
                coordinate.probe_begin = 0;
                coordinate.probe_end = bt_;
                coordinate.action_begin = action_begin;
                coordinate.action_end = action_end;
                coordinate.group_begin = group;
                coordinate.group_end = group + 1U;
                bind_variant_membership(
                    coordinate, VariantMembershipV1::group_execution_slice,
                    cross_execution_cursor, cross_execution_end
                );
                protected_tn(
                    coordinate,
                    checked_blas_dim(action_width, "group cross rows"),
                    checked_blas_dim(components_, "group cross columns"),
                    checked_blas_dim(reduction, "group cross reduction"),
                    group_action_scratch_.data(),
                    checked_blas_dim(reduction, "group action stride"),
                    raw_actions_.data(),
                    checked_blas_dim(reduction, "raw action stride"),
                    group_cross_scratch_.data(),
                    checked_blas_dim(action_width, "group cross stride")
                );
                PerformanceCategoryScopeV1 reduction_scope(
                    performance_categories_[static_cast<std::size_t>(
                        PerformanceCategoryV1::reduction)],
                    action_width * components_,
                    action_width * components_ * sizeof(double) * 2U
                );
                for (std::size_t local = 0; local < action_width; ++local) {
                    const std::size_t left = action_begin + local;
                    for (std::size_t right = 0; right < components_; ++right)
                        destination[(group * components_ + left) *
                                    components_ + right] +=
                            group_cross_scratch_[local + action_width * right] /
                            static_cast<double>(bt_);
                }
            }
            double* matrix = destination.data() +
                group * components_ * components_;
            PerformanceCategoryScopeV1 symmetry_scope(
                performance_categories_[static_cast<std::size_t>(
                    PerformanceCategoryV1::reduction)],
                components_ * components_,
                components_ * components_ * sizeof(double) * 2U
            );
            for (std::size_t left = 0; left < components_; ++left) {
                for (std::size_t right = left + 1; right < components_; ++right) {
                    const double average = 0.5 * (
                        matrix[left * components_ + right] +
                        matrix[right * components_ + left]
                    );
                    matrix[left * components_ + right] = average;
                    matrix[right * components_ + left] = average;
                }
            }
            if (!std::all_of(
                    matrix, matrix + components_ * components_,
                    [](double value) { return std::isfinite(value); }))
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 nonfinite group numerator"
                );
            cross_execution_cursor = cross_execution_end;
        }
        if (cross_execution_cursor != batch_execution_cursor)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 group cross membership coverage mismatch"
            );
        execution_cursor = batch_execution_cursor;
    }
    if (execution_cursor != m_ || observed_group_execution_visits_ != m_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 group execution permutation coverage mismatch"
        );
    ++observed_descriptor_passes_;
}

void ContextualReferenceExecutorV1::direct_grouped_phase(
    bool action_scaled, std::vector<double>& destination
) {
    using namespace stage2_detail;
    std::vector<double>& directional = action_scaled
        ? direct_directional_action_ : direct_directional_genotype_;
    std::fill(directional.begin(), directional.end(), 0.0);
    Sha256 missing_digest = begin_missingness_sha256(m_, n_);
    Sha256 bed_content_digest = begin_retained_bed_content_digest();
    for (std::size_t variant_begin = 0; variant_begin < m_;
         variant_begin += variant_block_) {
        const std::size_t variant_end = std::min(
            m_, variant_begin + variant_block_
        );
        const std::size_t variant_width = decode_block(
            variant_begin, variant_end, missing_digest,
            bed_content_digest
        );
        for (std::size_t context = 0; context < q_; ++context) {
            if (!action_scaled) {
                PerformanceCategoryScopeV1 genotype_scale_scope(
                    performance_categories_[static_cast<std::size_t>(
                        PerformanceCategoryV1::row_scale)],
                    n_ * variant_width,
                    n_ * variant_width * sizeof(double) * 3U
                );
                const double* phi = context_basis_.data() + n_ * context;
                for (std::size_t local = 0; local < variant_width; ++local)
                    for (std::size_t row = 0; row < n_; ++row)
                        direct_scaled_genotype_scratch_[row + n_ * local] =
                            phi[row] * decoded_scratch_[row + n_ * local];
            }
            for (std::size_t action_begin = 0; action_begin < components_;
                 action_begin += action_tile_) {
                const std::size_t action_end = std::min(
                    components_, action_begin + action_tile_
                );
                const std::size_t action_width = action_end - action_begin;
                for (std::size_t probe_begin = 0; probe_begin < bt_;
                     probe_begin += sample_probe_tile_) {
                    const std::size_t probe_end = std::min(
                        bt_, probe_begin + sample_probe_tile_
                    );
                    const std::size_t probe_width = probe_end - probe_begin;
                    PerformanceCategoryScopeV1 action_scale_scope(
                        performance_categories_[static_cast<std::size_t>(
                            PerformanceCategoryV1::row_scale)],
                        n_ * action_width * probe_width,
                        n_ * action_width * probe_width * sizeof(double) * 4U
                    );
                    for (std::size_t local_action = 0;
                         local_action < action_width; ++local_action) {
                        const std::size_t component =
                            action_begin + local_action;
                        for (std::size_t local_probe = 0;
                             local_probe < probe_width; ++local_probe) {
                            const std::size_t probe =
                                probe_begin + local_probe;
                            const std::size_t column =
                                local_action * probe_width + local_probe;
                            const double* source = raw_actions_.data() +
                                action_index(0, component, probe);
                            double* plain = direct_action_scratch_.data() +
                                n_ * column;
                            double* scaled =
                                direct_scaled_action_scratch_.data() +
                                n_ * column;
                            const double* phi =
                                context_basis_.data() + n_ * context;
                            for (std::size_t row = 0; row < n_; ++row) {
                                plain[row] = source[row];
                                scaled[row] = phi[row] * source[row];
                            }
                        }
                    }
                    action_scale_scope.finish();
                    SemanticCoordinateV1 coordinate{};
                    coordinate.operation =
                        ContextualDenseSemanticV1::direct_grouped_tn;
                    coordinate.resident_begin = 0;
                    coordinate.resident_end = 1;
                    coordinate.probe_begin = probe_begin;
                    coordinate.probe_end = probe_end;
                    coordinate.variant_begin = variant_begin;
                    coordinate.variant_end = variant_end;
                    coordinate.context_begin = context;
                    coordinate.context_end = context + 1U;
                    coordinate.action_begin = action_begin;
                    coordinate.action_end = action_end;
                    coordinate.group_begin = 0;
                    coordinate.group_end = groups_;
                    coordinate.phase = SemanticPhaseV1::group;
                    coordinate.placement = action_scaled
                        ? SemanticPlacementV1::action_scaled
                        : SemanticPlacementV1::genotype_scaled;
                    coordinate.canonical_begin = variant_begin;
                    coordinate.canonical_end = variant_end;
                    protected_tn(
                        coordinate,
                        checked_blas_dim(variant_width,
                                         "direct grouped rows"),
                        checked_blas_dim(action_width * probe_width,
                                         "direct grouped columns"),
                        checked_blas_dim(n_, "direct grouped reduction"),
                        action_scaled ? decoded_scratch_.data() :
                                        direct_scaled_genotype_scratch_.data(),
                        checked_blas_dim(n_, "direct grouped left stride"),
                        action_scaled ? direct_scaled_action_scratch_.data() :
                                        direct_action_scratch_.data(),
                        checked_blas_dim(n_, "direct grouped right stride"),
                        direct_output_scratch_.data(),
                        checked_blas_dim(variant_width,
                                         "direct grouped output stride")
                    );
                    PerformanceCategoryScopeV1 reduction_scope(
                        performance_categories_[static_cast<std::size_t>(
                            PerformanceCategoryV1::reduction)],
                        variant_width * action_width * probe_width,
                        variant_width * action_width * probe_width *
                            sizeof(double) * 3U
                    );
                    for (std::size_t source_component = 0;
                         source_component < components_; ++source_component) {
                        const std::size_t annotation =
                            static_cast<std::size_t>(
                                component_annotation_[source_component]);
                        const std::size_t pair = static_cast<std::size_t>(
                            component_pair_[source_component]
                        );
                        const std::size_t first = static_cast<std::size_t>(
                            pair_q_[pair]
                        );
                        const std::size_t second = static_cast<std::size_t>(
                            pair_r_[pair]
                        );
                        if (context != first && context != second) continue;
                        const std::size_t source_context = context == first
                            ? second : first;
                        for (std::size_t local_action = 0;
                             local_action < action_width; ++local_action) {
                            const std::size_t target_component =
                                action_begin + local_action;
                            for (std::size_t local_probe = 0;
                                 local_probe < probe_width; ++local_probe) {
                                const std::size_t probe =
                                    probe_begin + local_probe;
                                const std::size_t column =
                                    local_action * probe_width + local_probe;
                                const double* contraction =
                                    direct_output_scratch_.data() +
                                    variant_width * column;
                                for (std::size_t local = 0;
                                     local < variant_width; ++local) {
                                    const std::size_t variant =
                                        variant_begin + local;
                                    const std::size_t group =
                                        static_cast<std::size_t>(
                                            group_index_[variant]);
                                    const double coefficient = annotation_value(
                                        variant, annotation
                                    ) * source_scores_[source_index(
                                        variant, source_context, probe
                                    )];
                                    directional[
                                        (group * components_ +
                                         source_component) * components_ +
                                        target_component] +=
                                            coefficient * contraction[local];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    finish_retained_bed_content_digest(bed_content_digest);
    const std::string digest = missing_digest.finish();
    if (digest != expected_missingness_sha256_ ||
        digest != observed_missingness_sha256_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 missingness changed in direct grouped pass"
    );
    ++observed_descriptor_passes_;
    const double denominator = 2.0 * static_cast<double>(bt_);
    PerformanceCategoryScopeV1 final_reduction_scope(
        performance_categories_[static_cast<std::size_t>(
            PerformanceCategoryV1::reduction)],
        groups_ * components_ * components_,
        groups_ * components_ * components_ * sizeof(double) * 3U
    );
    for (std::size_t group = 0; group < groups_; ++group) {
        for (std::size_t left = 0; left < components_; ++left) {
            for (std::size_t right = 0; right < components_; ++right) {
                const std::size_t index =
                    (group * components_ + left) * components_ + right;
                const std::size_t transpose =
                    (group * components_ + right) * components_ + left;
                destination[index] =
                    (directional[index] + directional[transpose]) /
                    denominator;
                if (!std::isfinite(destination[index]))
                    throw std::runtime_error(
                        "ContextualReferenceExecutorV1 nonfinite direct grouped numerator"
                    );
            }
        }
    }
}

void ContextualReferenceExecutorV1::group_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::group)]
    );
    if (fingerprint_source_action() != action_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 source/action identity changed before group phase"
        );
    phase_ = "group";
    state_ = "group_running";
    record_event(EventClassV1::phase_transition, SemanticCoordinateV1{},
                 "group_running", 0);
    observed_group_execution_visits_ = 0;
    if (grouped_algorithm_ == "group_restricted_action_v1") {
        group_restricted_phase(group_gram_numerator_);
        if (enable_grouped_differential_)
            group_restricted_differential_ = group_gram_numerator_;
    } else if (enable_grouped_differential_) {
        group_restricted_phase(group_restricted_differential_);
    }
    if (enable_grouped_differential_) {
        direct_grouped_phase(true, group_direct_action_);
        direct_grouped_phase(false, group_direct_genotype_);
        if (grouped_algorithm_ == "direct_grouped_tn_v1")
            group_gram_numerator_ =
                direct_grouped_scaling_ == "genotype_scaled_v1"
                    ? group_direct_genotype_ : group_direct_action_;
    } else if (grouped_algorithm_ == "direct_grouped_tn_v1") {
        direct_grouped_phase(
            direct_grouped_scaling_ == "action_scaled_v1",
            group_gram_numerator_
        );
    }

    if (fault_mode_ == "semantic_group_reconstruction") {
        double scale = 1.0;
        for (double value : raw_gram_numerator_)
            scale = std::max(scale, std::abs(value));
        group_gram_numerator_[0] += 1.0 + 1.0e-6 * scale;
        fault_consumed_ = true;
        ++injection_count_;
        SemanticCoordinateV1 coordinate{};
        coordinate.group_begin = 0;
        coordinate.group_end = 1;
        record_event(EventClassV1::fault_injection, coordinate,
                     "semantic_group_reconstruction", 0);
    }

    group_reconstruction_max_abs_ = 0.0;
    double scientific_scale = 1.0;
    for (double value : raw_gram_numerator_)
        scientific_scale = std::max(scientific_scale, std::abs(value));
    for (std::size_t left = 0; left < components_; ++left) {
        for (std::size_t right = 0; right < components_; ++right) {
            double reconstructed = 0.0;
            for (std::size_t group = 0; group < groups_; ++group)
                reconstructed += group_gram_numerator_[
                    (group * components_ + left) * components_ + right];
            group_reconstruction_max_abs_ = std::max(
                group_reconstruction_max_abs_,
                std::abs(reconstructed -
                         raw_gram_numerator_[left + components_ * right])
            );
        }
    }
    if (enable_grouped_differential_) {
        for (std::size_t index = 0; index < group_gram_numerator_.size();
             ++index) {
            direct_action_restricted_max_abs_ = std::max(
                direct_action_restricted_max_abs_,
                std::abs(group_direct_action_[index] -
                         group_restricted_differential_[index])
            );
            direct_genotype_restricted_max_abs_ = std::max(
                direct_genotype_restricted_max_abs_,
                std::abs(group_direct_genotype_[index] -
                         group_restricted_differential_[index])
            );
            direct_placement_max_abs_ = std::max(
                direct_placement_max_abs_,
                std::abs(group_direct_action_[index] -
                         group_direct_genotype_[index])
            );
        }
    }
    const double tolerance = 1.0e-9 + 1.0e-11 * scientific_scale;
    if (group_reconstruction_max_abs_ > tolerance ||
        direct_action_restricted_max_abs_ > tolerance ||
        direct_genotype_restricted_max_abs_ > tolerance ||
        direct_placement_max_abs_ > tolerance) {
        if (fault_mode_ == "semantic_group_reconstruction")
            record_event(EventClassV1::fault_detection,
                         SemanticCoordinateV1{},
                         "group_reconstruction_mismatch", 0);
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 semantic group reconstruction failed"
        );
    }
    record_event(EventClassV1::semantic_verification,
                 SemanticCoordinateV1{},
                 "group_reconstruction_verified", 0);
    for (std::vector<double>* values : {
            &group_target_scratch_, &group_action_scratch_,
            &group_cross_scratch_, &direct_action_scratch_,
            &direct_scaled_action_scratch_,
            &direct_scaled_genotype_scratch_, &direct_output_scratch_,
            &direct_directional_action_, &direct_directional_genotype_})
        std::vector<double>().swap(*values);
    state_ = "group_complete";
    group_output_fingerprint_ = fingerprint_outputs();
    checkpoint("post_group");
    if (fingerprint_outputs() != group_output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 grouped output mutation detected"
        );
}

void ContextualReferenceExecutorV1::same_person_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::same_person)]
    );
    if (fingerprint_outputs() != group_output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 grouped identity changed before same-person phase"
        );
    phase_ = "same_person";
    state_ = "same_person_running";
    record_event(EventClassV1::phase_transition, SemanticCoordinateV1{},
                 "same_person_running", 0);
    std::fill(same_probe_sums_.begin(), same_probe_sums_.end(), 0.0);
    std::fill(same_probe_cross_.begin(), same_probe_cross_.end(), 0.0);
    observed_variant_probe_coverage_ = 0;
    for (std::size_t probe_begin = 0; probe_begin < bd_;
         probe_begin += variant_probe_tile_) {
        const std::size_t probe_end = std::min(
            bd_, probe_begin + variant_probe_tile_
        );
        const std::size_t probe_width = probe_end - probe_begin;
        fill_variant_probe_tile(
            probe_begin, probe_width, variant_probe_scratch_
        );
        std::fill_n(
            same_base_scratch_.data(), n_ * k_ * probe_width, 0.0
        );
        Sha256 missing_digest = begin_missingness_sha256(m_, n_);
        Sha256 bed_content_digest = begin_retained_bed_content_digest();
        for (std::size_t variant_begin = 0; variant_begin < m_;
             variant_begin += variant_block_) {
            const std::size_t variant_end = std::min(
                m_, variant_begin + variant_block_
            );
            const std::size_t variant_width = decode_block(
                variant_begin, variant_end, missing_digest,
                bed_content_digest
            );
            PerformanceCategoryScopeV1 packing_scope(
                performance_categories_[static_cast<std::size_t>(
                    PerformanceCategoryV1::packing)],
                variant_width * k_ * probe_width,
                variant_width * k_ * probe_width * sizeof(double) * 3U
            );
            for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                for (std::size_t local_probe = 0;
                     local_probe < probe_width; ++local_probe) {
                    double* right = same_right_scratch_.data() +
                        variant_width * (annotation * probe_width + local_probe);
                    for (std::size_t local = 0; local < variant_width; ++local) {
                        const std::size_t variant = variant_begin + local;
                        const double weight = annotation_value(
                            variant, annotation
                        );
                        const double root = strict_annotation_ids_.empty()
                            ? std::sqrt(weight) : weight;
                        right[local] = root * variant_probe_scratch_[
                            variant + m_ * local_probe];
                    }
                }
            }
            packing_scope.finish();
            SemanticCoordinateV1 coordinate{};
            coordinate.operation =
                ContextualDenseSemanticV1::same_person_target_nn;
            coordinate.resident_begin = 0;
            coordinate.resident_end = 1;
            coordinate.probe_begin = probe_begin;
            coordinate.probe_end = probe_end;
            coordinate.variant_begin = variant_begin;
            coordinate.variant_end = variant_end;
            coordinate.annotation_begin = 0;
            coordinate.annotation_end = k_;
            protected_nn(
                coordinate,
                checked_blas_dim(n_, "same-person target rows"),
                checked_blas_dim(k_ * probe_width,
                                 "same-person target columns"),
                checked_blas_dim(variant_width,
                                 "same-person target reduction"),
                decoded_scratch_.data(),
                checked_blas_dim(n_, "same-person genotype stride"),
                same_right_scratch_.data(),
                checked_blas_dim(variant_width,
                                 "same-person right stride"),
                same_update_scratch_.data(),
                checked_blas_dim(n_, "same-person target output stride")
            );
            PerformanceCategoryScopeV1 target_reduction_scope(
                performance_categories_[static_cast<std::size_t>(
                    PerformanceCategoryV1::reduction)],
                n_ * k_ * probe_width,
                n_ * k_ * probe_width * sizeof(double) * 2U
            );
            for (std::size_t column = 0; column < k_ * probe_width; ++column)
                for (std::size_t row = 0; row < n_; ++row)
                    same_base_scratch_[row + n_ * column] +=
                        same_update_scratch_[row + n_ * column];
        }
        finish_retained_bed_content_digest(bed_content_digest);
        const std::string digest = missing_digest.finish();
        if (digest != expected_missingness_sha256_ ||
            digest != observed_missingness_sha256_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 missingness changed in same-person pass"
            );
        ++observed_descriptor_passes_;
        observed_variant_probe_coverage_ += probe_width;
        PerformanceCategoryScopeV1 context_scale_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::row_scale)],
            n_ * k_ * q_ * probe_width,
            n_ * k_ * q_ * probe_width * sizeof(double) * 3U
        );
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            for (std::size_t context = 0; context < q_; ++context) {
                const double* phi = context_basis_.data() + n_ * context;
                for (std::size_t local_probe = 0;
                     local_probe < probe_width; ++local_probe) {
                    const double* source = same_base_scratch_.data() + n_ *
                        (annotation * probe_width + local_probe);
                    double* target = same_contextual_scratch_.data() + n_ *
                        ((annotation * q_ + context) * probe_width + local_probe);
                    for (std::size_t row = 0; row < n_; ++row)
                        target[row] = phi[row] * source[row];
                }
            }
        }
        context_scale_scope.finish();
        SemanticCoordinateV1 projection_coordinate{};
        projection_coordinate.resident_begin = 0;
        projection_coordinate.resident_end = 1;
        projection_coordinate.probe_begin = probe_begin;
        projection_coordinate.probe_end = probe_end;
        projection_coordinate.annotation_begin = 0;
        projection_coordinate.annotation_end = k_;
        projection_coordinate.context_begin = 0;
        projection_coordinate.context_end = q_;
        project_panel(
            same_contextual_scratch_.data(), k_ * q_ * probe_width,
            projection_coordinate,
            ContextualDenseSemanticV1::same_person_projection_tn,
            ContextualDenseSemanticV1::same_person_projection_nn,
            same_projected_scratch_
        );
        PerformanceCategoryScopeV1 component_reduction_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::reduction)],
            n_ * components_ * probe_width,
            n_ * components_ * probe_width * sizeof(double) * 4U
        );
        for (std::size_t component = 0; component < components_; ++component) {
            const std::size_t annotation = static_cast<std::size_t>(
                component_annotation_[component]
            );
            const std::size_t pair = static_cast<std::size_t>(
                component_pair_[component]
            );
            const std::size_t first = static_cast<std::size_t>(pair_q_[pair]);
            const std::size_t second = static_cast<std::size_t>(pair_r_[pair]);
            const double eta = static_cast<double>(pair_eta_[pair]);
            const double mass = annotation_masses_[annotation];
            for (std::size_t local_probe = 0;
                 local_probe < probe_width; ++local_probe) {
                const double* first_values = same_projected_scratch_.data() +
                    n_ * ((annotation * q_ + first) * probe_width + local_probe);
                const double* second_values = same_projected_scratch_.data() +
                    n_ * ((annotation * q_ + second) * probe_width + local_probe);
                double* target = same_g_scratch_.data() +
                    n_ * (local_probe + probe_width * component);
                for (std::size_t row = 0; row < n_; ++row) {
                    const double value = eta * first_values[row] *
                        second_values[row] / mass;
                    target[row] = value;
                    same_probe_sums_[row + n_ * component] += value;
                }
            }
        }
        component_reduction_scope.finish();
        SemanticCoordinateV1 gram_coordinate{};
        gram_coordinate.operation =
            ContextualDenseSemanticV1::same_person_gram_tn;
        gram_coordinate.resident_begin = 0;
        gram_coordinate.resident_end = 1;
        gram_coordinate.probe_begin = probe_begin;
        gram_coordinate.probe_end = probe_end;
        gram_coordinate.action_begin = 0;
        gram_coordinate.action_end = components_;
        gram_coordinate.phase = SemanticPhaseV1::same_person;
        gram_coordinate.role = SemanticRoleV1::tile;
        gram_coordinate.canonical_begin = probe_begin;
        gram_coordinate.canonical_end = probe_end;
        protected_tn(
            gram_coordinate,
            checked_blas_dim(components_, "same-person Gram rows"),
            checked_blas_dim(components_, "same-person Gram columns"),
            checked_blas_dim(n_ * probe_width,
                             "same-person Gram reduction"),
            same_g_scratch_.data(),
            checked_blas_dim(n_ * probe_width,
                             "same-person component stride"),
            same_g_scratch_.data(),
            checked_blas_dim(n_ * probe_width,
                             "same-person component stride"),
            gram_update_scratch_.data(),
            checked_blas_dim(components_, "same-person Gram stride")
        );
        PerformanceCategoryScopeV1 cross_reduction_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::reduction)],
            components_ * components_,
            components_ * components_ * sizeof(double) * 2U
        );
        for (std::size_t index = 0; index < components_ * components_; ++index)
            same_probe_cross_[index] += gram_update_scratch_[index];
    }
    SemanticCoordinateV1 global_coordinate{};
    global_coordinate.operation =
        ContextualDenseSemanticV1::same_person_gram_tn;
    global_coordinate.resident_begin = 0;
    global_coordinate.resident_end = 1;
    global_coordinate.probe_begin = 0;
    global_coordinate.probe_end = bd_;
    global_coordinate.action_begin = 0;
    global_coordinate.action_end = components_;
    global_coordinate.phase = SemanticPhaseV1::same_person;
    global_coordinate.role = SemanticRoleV1::global_merge;
    global_coordinate.canonical_begin = 0;
    global_coordinate.canonical_end = bd_;
    protected_tn(
        global_coordinate,
        checked_blas_dim(components_, "same-person global rows"),
        checked_blas_dim(components_, "same-person global columns"),
        checked_blas_dim(n_, "same-person global reduction"),
        same_probe_sums_.data(), checked_blas_dim(n_, "probe sum stride"),
        same_probe_sums_.data(), checked_blas_dim(n_, "probe sum stride"),
        gram_update_scratch_.data(),
        checked_blas_dim(components_, "same-person global stride")
    );
    const double denominator = static_cast<double>(bd_) *
        static_cast<double>(bd_ - 1U);
    same_person_pre_symmetry_max_abs_ = 0.0;
    PerformanceCategoryScopeV1 final_reduction_scope(
        performance_categories_[static_cast<std::size_t>(
            PerformanceCategoryV1::reduction)],
        components_ * components_,
        components_ * components_ * sizeof(double) * 4U
    );
    for (std::size_t left = 0; left < components_; ++left) {
        for (std::size_t right = 0; right < components_; ++right) {
            const std::size_t column_major = left + components_ * right;
            same_person_[left * components_ + right] =
                (gram_update_scratch_[column_major] -
                 same_probe_cross_[column_major]) / denominator;
        }
    }
    for (std::size_t left = 0; left < components_; ++left) {
        for (std::size_t right = left + 1; right < components_; ++right) {
            const double first = same_person_[left * components_ + right];
            const double second = same_person_[right * components_ + left];
            same_person_pre_symmetry_max_abs_ = std::max(
                same_person_pre_symmetry_max_abs_, std::abs(first - second)
            );
            const double average = 0.5 * (first + second);
            same_person_[left * components_ + right] = average;
            same_person_[right * components_ + left] = average;
        }
    }
    if (observed_variant_probe_coverage_ != bd_ ||
        !std::all_of(same_person_.begin(), same_person_.end(),
                     [](double value) { return std::isfinite(value); }))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 same-person coverage or finiteness failure"
        );
    state_ = "same_person_complete";
    group_output_fingerprint_ = fingerprint_outputs();
    checkpoint("post_same_person");
    if (fingerprint_outputs() != group_output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 same-person output mutation detected"
        );
}

void ContextualReferenceExecutorV1::residual_moments_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::residual_moments)]
    );
    phase_ = "residual_moments";
    state_ = "residual_moments_running";
    record_event(EventClassV1::phase_transition, SemanticCoordinateV1{},
                 "residual_moments_running", 0);

    PerformanceCategoryScopeV1 residual_reduction_scope(
        performance_categories_[static_cast<std::size_t>(
            PerformanceCategoryV1::reduction)],
        n_ * (l_ + h_ + h_ * h_), 0
    );
    std::fill(leverage_scratch_.begin(), leverage_scratch_.end(), 0.0);
    for (std::size_t row = 0; row < n_; ++row) {
        for (std::size_t basis = 0; basis < rank_; ++basis) {
            const double value = fixed_basis_[row + n_ * basis];
            leverage_scratch_[row] += value * value;
        }
    }

    const double residual_rank = static_cast<double>(n_ - rank_);
    const double epsilon = std::numeric_limits<double>::epsilon();
    double maximum_normalized_phenotype = 0.0;
    for (std::size_t trait = 0; trait < l_; ++trait) {
        double raw_sum_squares = 0.0;
        for (std::size_t row = 0; row < n_; ++row) {
            const double value = phenotype_batch_[row + n_ * trait];
            normalized_phenotypes_[row + n_ * trait] = value;
            raw_sum_squares += value * value;
        }
        for (std::size_t basis = 0; basis < rank_; ++basis) {
            double coefficient = 0.0;
            for (std::size_t row = 0; row < n_; ++row)
                coefficient += fixed_basis_[row + n_ * basis] *
                    phenotype_batch_[row + n_ * trait];
            phenotype_coefficient_scratch_[basis + rank_ * trait] =
                coefficient;
        }
        for (std::size_t row = 0; row < n_; ++row) {
            double update = 0.0;
            for (std::size_t basis = 0; basis < rank_; ++basis)
                update += fixed_basis_[row + n_ * basis] *
                    phenotype_coefficient_scratch_[basis + rank_ * trait];
            normalized_phenotypes_[row + n_ * trait] -= update;
        }
        double projected_sum_squares = 0.0;
        for (std::size_t row = 0; row < n_; ++row) {
            const double value = normalized_phenotypes_[row + n_ * trait];
            projected_sum_squares += value * value;
        }
        const double minimum = epsilon * raw_sum_squares;
        if (!std::isfinite(projected_sum_squares) ||
            projected_sum_squares <= minimum)
            throw std::runtime_error(
                "ContextualTraitExecutorV1 phenotype has zero or invalid residual variance"
            );
        const double scale = std::sqrt(
            residual_rank / projected_sum_squares
        );
        double normalized_sum_squares = 0.0;
        for (std::size_t row = 0; row < n_; ++row) {
            double& value = normalized_phenotypes_[row + n_ * trait];
            value *= scale;
            normalized_sum_squares += value * value;
            maximum_normalized_phenotype = std::max(
                maximum_normalized_phenotype, std::abs(value)
            );
        }
        phenotype_normalized_norm_error_max_abs_ = std::max(
            phenotype_normalized_norm_error_max_abs_,
            std::abs(normalized_sum_squares - residual_rank)
        );
        for (std::size_t basis = 0; basis < rank_; ++basis) {
            double leakage = 0.0;
            for (std::size_t row = 0; row < n_; ++row)
                leakage += fixed_basis_[row + n_ * basis] *
                    normalized_phenotypes_[row + n_ * trait];
            phenotype_projection_leakage_max_abs_ = std::max(
                phenotype_projection_leakage_max_abs_, std::abs(leakage)
            );
        }
        ++observed_phenotype_projections_;
    }
    if (phenotype_projection_leakage_max_abs_ >
            2.0e-10 * (1.0 + maximum_normalized_phenotype * n_) ||
        phenotype_normalized_norm_error_max_abs_ >
            2.0e-10 * (1.0 + residual_rank))
        throw std::runtime_error(
            "ContextualTraitExecutorV1 phenotype normalization verification failed"
        );

    std::fill(residual_compressed_scratch_.begin(),
              residual_compressed_scratch_.end(), 0.0);
    for (std::size_t residual = 0; residual < h_; ++residual) {
        for (std::size_t left = 0; left < rank_; ++left) {
            for (std::size_t right = 0; right < rank_; ++right) {
                double value = 0.0;
                for (std::size_t row = 0; row < n_; ++row)
                    value += residual_basis_[row + n_ * residual] *
                        fixed_basis_[row + n_ * left] *
                        fixed_basis_[row + n_ * right];
                residual_compressed_scratch_[
                    (residual * rank_ + left) * rank_ + right
                ] = value;
            }
        }
        double trace = 0.0;
        for (std::size_t row = 0; row < n_; ++row)
            trace += residual_basis_[row + n_ * residual] *
                (1.0 - leverage_scratch_[row]);
        residual_traces_[residual] = trace;
        for (std::size_t trait = 0; trait < l_; ++trait) {
            double rhs = 0.0;
            for (std::size_t row = 0; row < n_; ++row) {
                const double phenotype =
                    normalized_phenotypes_[row + n_ * trait];
                rhs += residual_basis_[row + n_ * residual] *
                    phenotype * phenotype;
            }
            residual_rhs_[residual * l_ + trait] = rhs;
        }
    }
    for (std::size_t left = 0; left < h_; ++left) {
        for (std::size_t right = 0; right < h_; ++right) {
            double value = 0.0;
            for (std::size_t row = 0; row < n_; ++row) {
                const double product = residual_basis_[row + n_ * left] *
                    residual_basis_[row + n_ * right];
                value += product * (1.0 - 2.0 * leverage_scratch_[row]);
            }
            for (std::size_t first = 0; first < rank_; ++first) {
                for (std::size_t second = 0; second < rank_; ++second) {
                    value += residual_compressed_scratch_[
                        (left * rank_ + first) * rank_ + second
                    ] * residual_compressed_scratch_[
                        (right * rank_ + second) * rank_ + first
                    ];
                }
            }
            residual_gram_[left * h_ + right] = value;
        }
    }
    double maximum_residual_gram = 0.0;
    for (std::size_t left = 0; left < h_; ++left) {
        for (std::size_t right = left + 1; right < h_; ++right) {
            residual_gram_pre_symmetry_max_abs_ = std::max(
                residual_gram_pre_symmetry_max_abs_,
                std::abs(residual_gram_[left * h_ + right] -
                         residual_gram_[right * h_ + left])
            );
            const double value = 0.5 * (
                residual_gram_[left * h_ + right] +
                residual_gram_[right * h_ + left]
            );
            residual_gram_[left * h_ + right] = value;
            residual_gram_[right * h_ + left] = value;
        }
    }
    for (double value : residual_gram_)
        maximum_residual_gram = std::max(
            maximum_residual_gram, std::abs(value)
        );
    if (residual_gram_pre_symmetry_max_abs_ >
            2.0e-10 * (1.0 + maximum_residual_gram))
        throw std::runtime_error(
            "ContextualTraitExecutorV1 residual Gram symmetry verification failed"
        );

    residual_reduction_scope.finish();
    state_ = "residual_moments_complete";
    const auto residual_state_seal = [&]() {
        Sha256 digest;
        sha_update_string(digest, "contextual_residual_derived_state_v1");
        for (const std::string& value : {
                array_sha256_f64_matrix_c(normalized_phenotypes_, n_, l_),
                array_sha256_f64_matrix_c(
                    phenotype_coefficient_scratch_, rank_, l_),
                array_sha256_f64_c_order(leverage_scratch_, {n_}),
                array_sha256_f64_c_order(
                    residual_compressed_scratch_, {h_, rank_, rank_}),
                array_sha256_f64_c_order(residual_rhs_, {h_, l_}),
                array_sha256_f64_c_order(residual_traces_, {h_}),
                array_sha256_f64_c_order(residual_gram_, {h_, h_})}) {
            sha_update_u64_le(digest, value.size());
            sha_update(digest, value.data(), value.size());
        }
        return digest.finish();
    };
    residual_derived_state_sha256_ = residual_state_seal();
    const std::uint64_t residual_fingerprint = fingerprint_outputs();
    checkpoint("post_residual_moments");
    if (fingerprint_outputs() != residual_fingerprint ||
        residual_state_seal() != residual_derived_state_sha256_)
        throw std::runtime_error(
            "ContextualTraitExecutorV1 residual derived-state mutation detected"
        );
}

void ContextualReferenceExecutorV1::trait_phase() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::trait)]
    );
    phase_ = "trait";
    state_ = "trait_running";
    record_event(EventClassV1::phase_transition, SemanticCoordinateV1{},
                 "trait_running", 0);

    {
        PerformanceCategoryScopeV1 category_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::row_scale)],
            n_ * q_ * l_, n_ * q_ * l_ * sizeof(double) * 3U
        );
        for (std::size_t trait = 0; trait < l_; ++trait) {
            for (std::size_t context = 0; context < q_; ++context) {
                double* column = trait_score_right_scratch_.data() +
                    n_ * (trait * q_ + context);
                for (std::size_t row = 0; row < n_; ++row)
                    column[row] = context_basis_[row + n_ * context] *
                        normalized_phenotypes_[row + n_ * trait];
            }
        }
    }

    Sha256 missing_digest = begin_missingness_sha256(m_, n_);
    Sha256 bed_content_digest = begin_retained_bed_content_digest();
    for (std::size_t variant_begin = 0; variant_begin < m_;
         variant_begin += variant_block_) {
        const std::size_t variant_end = std::min(
            m_, variant_begin + variant_block_
        );
        const std::size_t width = decode_block(
            variant_begin, variant_end, missing_digest,
            bed_content_digest
        );
        SemanticCoordinateV1 score_coordinate{};
        score_coordinate.operation = ContextualDenseSemanticV1::trait_score_tn;
        score_coordinate.variant_begin = variant_begin;
        score_coordinate.variant_end = variant_end;
        score_coordinate.context_begin = 0;
        score_coordinate.context_end = q_;
        score_coordinate.action_begin = 0;
        score_coordinate.action_end = l_;
        protected_tn(
            score_coordinate,
            checked_blas_dim(width, "trait score block rows"),
            checked_blas_dim(q_ * l_, "trait packed score columns"),
            checked_blas_dim(n_, "trait score reduction"),
            decoded_scratch_.data(), checked_blas_dim(n_, "decoded stride"),
            trait_score_right_scratch_.data(),
            checked_blas_dim(n_, "trait score RHS stride"),
            trait_score_scratch_.data(),
            checked_blas_dim(width, "trait score output stride")
        );

        {
            PerformanceCategoryScopeV1 category_scope(
                performance_categories_[static_cast<std::size_t>(
                    PerformanceCategoryV1::row_scale)],
                n_ * q_ * width,
                n_ * q_ * width * sizeof(double) * 3U
            );
            for (std::size_t context = 0; context < q_; ++context) {
                for (std::size_t variant = 0; variant < width; ++variant) {
                    double* feature = trait_raw_feature_scratch_.data() +
                        n_ * (context * width + variant);
                    const double* genotype = decoded_scratch_.data() + n_ * variant;
                    for (std::size_t row = 0; row < n_; ++row)
                        feature[row] = context_basis_[row + n_ * context] *
                            genotype[row];
                }
            }
        }
        for (std::size_t context = 0; context < q_; ++context) {
            for (std::size_t local_begin = 0; local_begin < width;
                 local_begin += trait_feature_tile_) {
                const std::size_t columns = std::min(
                    trait_feature_tile_, width - local_begin
                );
                const std::size_t global_begin = variant_begin + local_begin;
                const std::size_t global_end = global_begin + columns;
                const std::size_t column_begin = context * width + local_begin;
                SemanticCoordinateV1 coordinate{};
                coordinate.variant_begin = global_begin;
                coordinate.variant_end = global_end;
                coordinate.context_begin = context;
                coordinate.context_end = context + 1U;
                coordinate.action_begin = context * m_ + global_begin;
                coordinate.action_end = context * m_ + global_end;
                coordinate.canonical_begin = context * m_ + global_begin;
                coordinate.canonical_end = context * m_ + global_end;
                project_panel(
                    trait_raw_feature_scratch_.data() + n_ * column_begin,
                    columns, coordinate,
                    ContextualDenseSemanticV1::trait_feature_projection_tn,
                    ContextualDenseSemanticV1::trait_feature_projection_nn,
                    trait_projection_tile_scratch_
                );
                {
                    PerformanceCategoryScopeV1 category_scope(
                        performance_categories_[static_cast<std::size_t>(
                            PerformanceCategoryV1::packing)],
                        n_ * columns, n_ * columns * sizeof(double)
                    );
                    std::copy_n(
                        trait_projection_tile_scratch_.data(), n_ * columns,
                        trait_projected_feature_scratch_.data() + n_ * column_begin
                    );
                }
            }
        }

        PerformanceCategoryScopeV1 block_reduction_scope(
            performance_categories_[static_cast<std::size_t>(
                PerformanceCategoryV1::reduction)],
            width * components_, 0
        );
        for (std::size_t variant = 0; variant < width; ++variant) {
            const std::size_t global_variant = variant_begin + variant;
            const std::size_t group = static_cast<std::size_t>(
                group_index_[global_variant]
            );
            for (std::size_t component = 0; component < components_;
                 ++component) {
                const std::size_t annotation = static_cast<std::size_t>(
                    component_annotation_[component]
                );
                const std::size_t pair = static_cast<std::size_t>(
                    component_pair_[component]
                );
                const std::size_t left = static_cast<std::size_t>(pair_q_[pair]);
                const std::size_t right = static_cast<std::size_t>(pair_r_[pair]);
                const double weight = static_cast<double>(pair_eta_[pair]) *
                    annotation_value(global_variant, annotation);
                if (weight == 0.0) continue;
                const std::size_t grouped_component =
                    group * components_ + component;
                for (std::size_t trait = 0; trait < l_; ++trait) {
                    const double rhs = weight * trait_score_scratch_[
                        variant + width * (trait * q_ + left)
                    ] * trait_score_scratch_[
                        variant + width * (trait * q_ + right)
                    ];
                    genetic_rhs_[component * l_ + trait] += rhs;
                    group_rhs_[grouped_component * l_ + trait] += rhs;
                }
                const double* left_feature =
                    trait_projected_feature_scratch_.data() +
                    n_ * (left * width + variant);
                const double* right_feature =
                    trait_projected_feature_scratch_.data() +
                    n_ * (right * width + variant);
                double trace = 0.0;
                for (std::size_t row = 0; row < n_; ++row)
                    trace += left_feature[row] * right_feature[row];
                trace *= weight;
                genetic_traces_[component] += trace;
                group_trace_[grouped_component] += trace;
                for (std::size_t residual = 0; residual < h_; ++residual) {
                    double value = 0.0;
                    for (std::size_t row = 0; row < n_; ++row)
                        value += left_feature[row] *
                            residual_basis_[row + n_ * residual] *
                            right_feature[row];
                    value *= weight;
                    genetic_residual_[component * h_ + residual] += value;
                    group_genetic_residual_[
                        grouped_component * h_ + residual
                    ] += value;
                }
            }
        }
    }
    ++observed_descriptor_passes_;
    finish_retained_bed_content_digest(bed_content_digest);
    const std::string digest = missing_digest.finish();
    if (digest != expected_missingness_sha256_)
        throw std::runtime_error(
            "ContextualTraitExecutorV1 missingness SHA256 mismatch"
        );
    observed_missingness_sha256_ = digest;

    PerformanceCategoryScopeV1 final_reduction_scope(
        performance_categories_[static_cast<std::size_t>(
            PerformanceCategoryV1::reduction)],
        components_ * (l_ + h_ + 1U), 0
    );
    for (std::size_t component = 0; component < components_; ++component) {
        const std::size_t annotation = static_cast<std::size_t>(
            component_annotation_[component]
        );
        const double mass = annotation_masses_[annotation];
        for (std::size_t trait = 0; trait < l_; ++trait) {
            double grouped = 0.0;
            for (std::size_t group = 0; group < groups_; ++group)
                grouped += group_rhs_[
                    (group * components_ + component) * l_ + trait
                ];
            const double& full = genetic_rhs_[component * l_ + trait];
            trait_group_reconstruction_max_abs_ = std::max(
                trait_group_reconstruction_max_abs_, std::abs(grouped - full)
            );
            if (std::abs(grouped - full) > 2.0e-10 * (1.0 + std::abs(full)))
                throw std::runtime_error(
                    "ContextualTraitExecutorV1 grouped RHS reconstruction failed"
                );
            genetic_rhs_[component * l_ + trait] /= mass;
        }
        double grouped_trace = 0.0;
        for (std::size_t group = 0; group < groups_; ++group)
            grouped_trace += group_trace_[group * components_ + component];
        trait_group_reconstruction_max_abs_ = std::max(
            trait_group_reconstruction_max_abs_,
            std::abs(grouped_trace - genetic_traces_[component])
        );
        if (std::abs(grouped_trace - genetic_traces_[component]) >
                2.0e-10 * (1.0 + std::abs(genetic_traces_[component])))
            throw std::runtime_error(
                "ContextualTraitExecutorV1 grouped trace reconstruction failed"
            );
        genetic_traces_[component] /= mass;
        for (std::size_t residual = 0; residual < h_; ++residual) {
            double grouped = 0.0;
            for (std::size_t group = 0; group < groups_; ++group)
                grouped += group_genetic_residual_[
                    (group * components_ + component) * h_ + residual
                ];
            const double& full =
                genetic_residual_[component * h_ + residual];
            trait_group_reconstruction_max_abs_ = std::max(
                trait_group_reconstruction_max_abs_, std::abs(grouped - full)
            );
            if (std::abs(grouped - full) > 2.0e-10 * (1.0 + std::abs(full)))
                throw std::runtime_error(
                    "ContextualTraitExecutorV1 grouped residual reconstruction failed"
                );
            genetic_residual_[component * h_ + residual] /= mass;
        }
    }
    final_reduction_scope.finish();
    for (const std::vector<double>* values : {
            &genetic_rhs_, &genetic_traces_, &genetic_residual_,
            &residual_rhs_, &residual_traces_, &residual_gram_,
            &group_rhs_, &group_trace_, &group_genetic_residual_}) {
        if (std::any_of(values->begin(), values->end(),
                        [](double value) { return !std::isfinite(value); }))
            throw std::runtime_error(
                "ContextualTraitExecutorV1 produced nonfinite trait statistics"
            );
    }

    state_ = "trait_complete";
    const auto trait_output_seal = [&]() {
        Sha256 output_digest;
        sha_update_string(output_digest, "contextual_post_trait_outputs_v1");
        for (const std::string& value : {
                array_sha256_f64_c_order(genetic_rhs_, {components_, l_}),
                array_sha256_f64_c_order(genetic_traces_, {components_}),
                array_sha256_f64_c_order(genetic_residual_, {components_, h_}),
                array_sha256_f64_c_order(residual_rhs_, {h_, l_}),
                array_sha256_f64_c_order(residual_traces_, {h_}),
                array_sha256_f64_c_order(residual_gram_, {h_, h_}),
                array_sha256_f64_c_order(
                    group_rhs_, {groups_, components_, l_}),
                array_sha256_f64_c_order(
                    group_trace_, {groups_, components_}),
                array_sha256_f64_c_order(
                    group_genetic_residual_, {groups_, components_, h_})}) {
            sha_update_u64_le(output_digest, value.size());
            sha_update(output_digest, value.data(), value.size());
        }
        return output_digest.finish();
    };
    trait_output_phase_sha256_ = trait_output_seal();
    const std::uint64_t trait_fingerprint = fingerprint_outputs();
    checkpoint("post_trait");
    if (fingerprint_outputs() != trait_fingerprint ||
        trait_output_seal() != trait_output_phase_sha256_)
        throw std::runtime_error(
            "ContextualTraitExecutorV1 trait output mutation detected"
        );
}

void ContextualReferenceExecutorV1::inject_owned_mutation() {
    if (test_mutation_target_ == "none") return;
    const auto bump_double = [](std::vector<double>& values) {
        if (!values.empty()) values[0] = std::nextafter(
            values[0], std::numeric_limits<double>::infinity()
        );
    };
    const auto bump_i64 = [](std::vector<std::int64_t>& values) {
        if (!values.empty()) ++values[0];
    };
    const auto bump_string = [](std::vector<std::string>& values) {
        if (!values.empty() && !values[0].empty()) values[0][0] ^= 1;
    };
    if (test_mutation_target_ == "fixed_basis") bump_double(fixed_basis_);
    else if (test_mutation_target_ == "context_basis") bump_double(context_basis_);
    else if (test_mutation_target_ == "annotations") bump_double(annotation_weights_);
    else if (test_mutation_target_ == "affine_mean") bump_double(affine_mean_);
    else if (test_mutation_target_ == "affine_inverse_scale")
        bump_double(affine_inverse_scale_);
    else if (test_mutation_target_ == "sample_probes") bump_double(explicit_probes_);
    else if (test_mutation_target_ == "probe_keys" && !philox_keys_.empty())
        ++philox_keys_[0];
    else if (test_mutation_target_ == "variant_probes")
        bump_double(explicit_variant_probes_);
    else if (test_mutation_target_ == "variant_probe_keys" &&
             !variant_philox_keys_.empty())
        ++variant_philox_keys_[0];
    else if (test_mutation_target_ == "group_execution_order")
        bump_i64(group_execution_order_);
    else if (test_mutation_target_ == "retained_samples") bump_i64(retained_samples_);
    else if (test_mutation_target_ == "retained_variants") bump_i64(retained_variants_);
    else if (test_mutation_target_ == "groups") bump_i64(group_index_);
    else if (test_mutation_target_ == "counted_allele_flags" && !counted_a1_.empty())
        counted_a1_[0] ^= 1U;
    else if (test_mutation_target_ == "variant_metadata") bump_string(expected_variant_ids_);
    else if (test_mutation_target_ == "expected_missing_counts")
        bump_i64(expected_missing_counts_);
    else if (test_mutation_target_ == "annotation_names") bump_string(annotation_names_);
    else if (test_mutation_target_ == "group_names") bump_string(group_names_);
    else if (test_mutation_target_ == "phenotype_batch")
        bump_double(phenotype_batch_);
    else if (test_mutation_target_ == "residual_basis")
        bump_double(residual_basis_);
    else if (test_mutation_target_ == "trait_names") bump_string(trait_names_);
    else if (test_mutation_target_ == "residual_names")
        bump_string(residual_names_);
    else if (test_mutation_target_ == "pair_component_maps") bump_i64(pair_q_);
    else if (test_mutation_target_ == "retained_bed_record") {
        if (bed_base_ == nullptr || retained_variants_.empty() ||
            ::mprotect(bed_base_, bed_size_, PROT_READ | PROT_WRITE) != 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 cannot arm retained BED mutation test"
            );
        const std::size_t logical = retained_variants_.size() / 2U;
        const std::size_t physical = static_cast<std::size_t>(
            retained_variants_[logical]
        );
        bed_base_[3U + physical * bytes_per_snp_] ^= 0x03U;
        if (::mprotect(bed_base_, bed_size_, PROT_READ) != 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 cannot restore retained BED protection"
            );
    }
    else if (test_mutation_target_ == "output_buffers") {
        if (test_checkpoint_ == "post_residual_moments")
            bump_double(residual_rhs_);
        else if (test_checkpoint_ == "post_trait") bump_double(genetic_rhs_);
        else if (test_checkpoint_ == "post_source") bump_double(source_scores_);
        else if (test_checkpoint_ == "post_action") bump_double(raw_actions_);
        else if (test_checkpoint_ == "post_group")
            bump_double(group_gram_numerator_);
        else if (test_checkpoint_ == "post_same_person")
            bump_double(same_person_);
        else bump_double(gram_);
    }
    else
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 requested mutation target is unavailable"
        );
}

void ContextualReferenceExecutorV1::checkpoint(const char* name) {
    using namespace stage2_detail;
    check_files_unchanged();
    check_inputs_unchanged();
    SemanticCoordinateV1 coordinate{};
    record_event(EventClassV1::mutation_checkpoint, coordinate, name, 0);
    if (test_checkpoint_ == name) {
        std::unique_lock<std::mutex> lock(checkpoint_mutex_);
        reached_checkpoint_ = name;
        checkpoint_released_ = false;
        inject_owned_mutation();
        checkpoint_condition_.notify_all();
        checkpoint_condition_.wait(lock, [&] { return checkpoint_released_; });
    }
    check_files_unchanged();
    check_inputs_unchanged();
}

void ContextualReferenceExecutorV1::validate_event_ledger_for_publication() {
    using namespace stage2_detail;
    std::uint64_t protected_events = 0;
    std::uint64_t verification_events = 0;
    for (std::size_t index = 0; index < events_.size(); ++index) {
        const EventV1& event = events_[index];
        if (event.sequence != index + 1U || event.process_id <= 0 ||
            event.thread_id <= 0)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 event ledger sequence/ownership mismatch"
            );
        if (event.event_class == EventClassV1::terminal_failure)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 terminal event before publication"
            );
        const bool protected_coordinate = static_cast<std::size_t>(
            event.coordinate.operation
        ) < kContextualDenseSemanticCountV1;
        const bool attempt_event = protected_coordinate && (
            event.event_class == EventClassV1::protected_call ||
            event.event_class == EventClassV1::retry ||
            event.event_class == EventClassV1::repair ||
            event.event_class == EventClassV1::trusted_fallback ||
            event.event_class == EventClassV1::semantic_verification);
        if (attempt_event &&
            (event.rows <= 0 || event.columns <= 0 || event.reduction <= 0 ||
             event.left_stride <= 0 || event.right_stride <= 0 ||
             event.output_stride < event.rows ||
             event.operand_fingerprint == 0 ||
             event.witness_fingerprint == 0 ||
             event.runtime_before_fingerprint == 0 ||
             event.runtime_after_fingerprint == 0 ||
             !event.prefix_canary_verified ||
             !event.suffix_canary_verified ||
             !event.finiteness_verified ||
             !event.serialized_entry_verified ||
             !event.deterministic_non_vendor_backend))
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 protected event evidence incomplete"
            );
        if (event.event_class == EventClassV1::protected_call)
            ++protected_events;
        if (protected_coordinate &&
            event.event_class == EventClassV1::semantic_verification) {
            ++verification_events;
            if (event.accepted_output_fingerprint == 0)
                throw std::runtime_error(
                    "ContextualReferenceExecutorV1 accepted output identity missing"
                );
        }
    }
    if (protected_events != estimate_.total_protected_calls ||
        verification_events != estimate_.total_protected_calls ||
        event_sequence_ != events_.size())
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 exact event completeness mismatch"
        );
    if (events_.size() + 2U > telemetry_capacity_ ||
        events_.size() + 2U > estimate_.required_telemetry_capacity)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 publication/terminal telemetry slots not admitted"
        );
    event_ledger_validated_ = true;
}

void ContextualReferenceExecutorV1::finalize_and_release() {
    using namespace stage2_detail;
    PerformancePhaseScopeV1 performance_scope(
        performance_phases_[static_cast<std::size_t>(
            PerformancePhaseV1::finalization)]
    );
    phase_ = "finalization";
    state_ = "finalizing";
    output_fingerprint_ = fingerprint_outputs();
    checkpoint("pre_finalization");
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 scientific output mutation detected"
        );
    if (observed_calls_ != estimate_.calls)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 semantic protected-call ledger mismatch"
        );
    const std::uint64_t expected_passes = trait_mode_
        ? estimate_.trait_descriptor_passes
        : stage2_detail::checked_add(
            stage2_detail::checked_add(
                estimate_.source_descriptor_passes,
                estimate_.action_descriptor_passes, "descriptor passes"),
            stage2_detail::checked_add(
                estimate_.group_descriptor_passes,
                estimate_.same_person_descriptor_passes,
                "descriptor passes"), "descriptor passes"
        );
    const std::uint64_t expected_blocks = trait_mode_
        ? estimate_.trait_decoded_blocks
        : stage2_detail::checked_add(
            stage2_detail::checked_add(
                estimate_.source_decoded_blocks,
                estimate_.action_decoded_blocks, "decoded blocks"),
            stage2_detail::checked_add(
                estimate_.group_decoded_blocks,
                estimate_.same_person_decoded_blocks,
                "decoded blocks"), "decoded blocks"
        );
    const std::uint64_t expected_visits = stage2_detail::checked_mul(
        expected_passes, m_, "variant record visits"
    );
    if (observed_descriptor_passes_ != expected_passes ||
        observed_decoded_blocks_ != expected_blocks ||
        observed_variant_visits_ != expected_visits)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 descriptor accounting mismatch"
        );
    if (trait_mode_ && observed_phenotype_projections_ != l_)
        throw std::runtime_error(
            "ContextualTraitExecutorV1 phenotype projection accounting mismatch"
        );
    if (fault_mode_ != "none" && !fault_consumed_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 semantic fault selector matched no protected call"
        );
    if (enable_differential_snapshot_ && stage3_enabled_) {
        if (differential_same_probe_sums_.size() != n_ * components_ ||
            differential_same_probe_cross_.size() !=
                components_ * components_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 differential same-person reserve mismatch"
            );
        for (std::size_t row = 0; row < n_; ++row)
            for (std::size_t component = 0; component < components_;
                 ++component)
                differential_same_probe_sums_[row * components_ + component] =
                    same_probe_sums_[row + n_ * component];
        for (std::size_t row = 0; row < components_; ++row)
            for (std::size_t column = 0; column < components_; ++column)
                differential_same_probe_cross_[row * components_ + column] =
                    same_probe_cross_[row + components_ * column];
    }
    auto release_double = [](std::vector<double>& values) {
        std::vector<double>().swap(values);
    };
    auto release_u8 = [](std::vector<std::uint8_t>& values) {
        std::vector<std::uint8_t>().swap(values);
    };
    auto release_int = [](std::vector<int>& values) {
        std::vector<int>().swap(values);
    };
    for (std::vector<double>* values : {
            &primary_buffer_, &retry_buffer_, &witness_buffer_,
            &fallback_buffer_, &decoded_scratch_, &probe_resident_scratch_,
            &projected_probe_scratch_, &projection_coeff_scratch_,
            &projection_update_scratch_, &source_right_scratch_,
            &target_panel_scratch_, &target_right_scratch_,
            &target_update_scratch_, &action_panel_scratch_,
            &action_projected_scratch_, &gram_panel_scratch_,
            &gram_update_scratch_, &strict_genotype_scratch_,
            &group_target_scratch_, &group_action_scratch_,
            &group_cross_scratch_, &direct_action_scratch_,
            &direct_scaled_action_scratch_,
            &direct_scaled_genotype_scratch_, &direct_output_scratch_,
            &same_right_scratch_, &same_update_scratch_,
            &same_base_scratch_, &same_contextual_scratch_,
            &same_projected_scratch_, &same_g_scratch_,
            &same_probe_sums_, &same_probe_cross_,
            &variant_probe_scratch_, &direct_directional_action_,
            &direct_directional_genotype_, &normalized_phenotypes_,
            &phenotype_coefficient_scratch_, &leverage_scratch_,
            &residual_compressed_scratch_, &trait_score_right_scratch_,
            &trait_score_scratch_, &trait_raw_feature_scratch_,
            &trait_projected_feature_scratch_,
            &trait_projection_tile_scratch_})
        release_double(*values);
    release_u8(missing_scratch_);
    release_int(observed_scratch_);
    scratch_released_ = true;
    state_ = "scratch_released";
    record_event(EventClassV1::scratch_release, SemanticCoordinateV1{},
                 "execution_arena_released", 0);
    checkpoint("pre_publication");
    check_hardened_file_content_unchanged();
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 scientific output mutation detected"
        );
    validate_event_ledger_for_publication();
    state_ = trait_mode_ ? "trait_statistics_complete_ready" :
        (stage3_enabled_ ? "reference_statistics_complete_ready" :
                           "stage2_incomplete_ready");
    phase_ = "publication";
    record_event(EventClassV1::publication, SemanticCoordinateV1{},
                 trait_mode_ ? "compact_trait_statistics_ready" :
                 (stage3_enabled_
                    ? "compact_complete_statistics_ready"
                    : "compact_incomplete_result_ready"), 0);
}

void ContextualReferenceExecutorV1::execute() {
    if (fingerprint_outputs() != initial_output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 initial output buffer mutation detected"
        );
    checkpoint("post_seal");
    if (fingerprint_outputs() != initial_output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 output mutation at post_seal"
        );
    checkpoint("post_admission");
    if (fingerprint_outputs() != initial_output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 output mutation at post_admission"
        );
    if (trait_mode_) {
        residual_moments_phase();
        trait_phase();
    } else {
        source_phase();
        action_phase();
        gram_phase();
        if (stage3_enabled_) {
            group_phase();
            same_person_phase();
        }
    }
    finalize_and_release();
}

std::string ContextualReferenceExecutorV1::wait_for_test_checkpoint(
    std::uint64_t timeout_ms
) {
    std::unique_lock<std::mutex> lock(checkpoint_mutex_);
    if (!checkpoint_condition_.wait_for(
            lock, std::chrono::milliseconds(timeout_ms),
            [&] { return !reached_checkpoint_.empty(); }))
        return "";
    return reached_checkpoint_;
}

void ContextualReferenceExecutorV1::release_test_checkpoint() {
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);
    checkpoint_released_ = true;
    checkpoint_condition_.notify_all();
}

template <typename T>
nb::object stage2_readonly_owned_vec(const std::vector<T>& values) {
    const char* dtype = std::is_same_v<T, double> ? "float64" :
        (std::is_same_v<T, std::int64_t> ? "int64" : "uint8");
    nb::object result = nb::module_::import_("numpy").attr("empty")(
        nb::make_tuple(values.size()), nb::arg("dtype") = dtype,
        nb::arg("order") = "C"
    );
    auto view = nb::cast<nb::ndarray<T, nb::numpy, nb::ndim<1>,
                                     nb::c_contig, nb::device::cpu>>(result);
    if (!values.empty()) std::copy(values.begin(), values.end(), view.data());
    result.attr("setflags")(false);
    return result;
}

nb::object stage2_readonly_owned_mat(
    const std::vector<double>& values, std::size_t rows, std::size_t columns
) {
    if (values.size() != rows * columns)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 publication matrix shape mismatch"
        );
    nb::object result = nb::module_::import_("numpy").attr("empty")(
        nb::make_tuple(rows, columns), nb::arg("dtype") = "float64",
        nb::arg("order") = "C"
    );
    auto view = nb::cast<nb::ndarray<double, nb::numpy, nb::ndim<2>,
                                     nb::c_contig, nb::device::cpu>>(result);
    if (!values.empty()) std::copy(values.begin(), values.end(), view.data());
    result.attr("setflags")(false);
    return result;
}

nb::object stage2_readonly_owned_arr3(
    const std::vector<double>& values,
    std::size_t first, std::size_t second, std::size_t third
) {
    if (values.size() != stage2_detail::elements_for(
            {first, second, third}, "publication tensor elements"))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 publication tensor shape mismatch"
        );
    nb::object result = nb::module_::import_("numpy").attr("empty")(
        nb::make_tuple(first, second, third),
        nb::arg("dtype") = "float64", nb::arg("order") = "C"
    );
    auto view = nb::cast<nb::ndarray<double, nb::numpy, nb::ndim<3>,
                                     nb::c_contig, nb::device::cpu>>(result);
    if (!values.empty()) std::copy(values.begin(), values.end(), view.data());
    result.attr("setflags")(false);
    return result;
}

nb::dict ContextualReferenceExecutorV1::file_content_identity_dict() const {
    const auto descriptor_state = [](const FileState& state) {
        nb::dict item;
        item["device"] = static_cast<std::uint64_t>(state.device);
        item["inode"] = static_cast<std::uint64_t>(state.inode);
        item["size"] = static_cast<std::int64_t>(state.size);
        item["link_count"] = static_cast<std::uint64_t>(state.links);
        item["mtime_seconds"] = static_cast<std::int64_t>(state.mtime.tv_sec);
        item["mtime_nanoseconds"] =
            static_cast<std::int64_t>(state.mtime.tv_nsec);
        item["ctime_seconds"] = static_cast<std::int64_t>(state.ctime.tv_sec);
        item["ctime_nanoseconds"] =
            static_cast<std::int64_t>(state.ctime.tv_nsec);
        return item;
    };
    nb::dict result;
    result["schema"] = "contextual_file_content_identity_v2";
    result["policy"] =
        "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2";
    result["duplicated_descriptors"] = true;
    result["bim_full_sha256"] = bim_file_sha256_;
    result["fam_full_sha256"] = fam_file_sha256_;
    result["bed_header_sha256"] = bed_header_sha256_;
    result["retained_bed_logical_record_stream_sha256"] =
        retained_bed_content_sha256_;
    result["retained_record_count"] = m_;
    result["bytes_per_bed_record"] = bytes_per_snp_;
    result["retained_record_sha256_count"] = std::count_if(
        retained_bed_record_sha256_.begin(),
        retained_bed_record_sha256_.end(),
        [](const std::string& value) { return !value.empty(); }
    );
    result["bed_stream_accumulation"] =
        "authoritative_decode_logical_order_no_extra_bed_traversal_v1";
    result["bim_fam_hash_source"] =
        "full_duplicated_descriptor_bytes_v1";
    result["bed_descriptor_state"] = descriptor_state(bed_state_);
    result["bim_descriptor_state"] = descriptor_state(bim_state_);
    result["fam_descriptor_state"] = descriptor_state(fam_state_);
    result["boundary_sample_evidence"] =
        "first_last_4KiB_per_descriptor_v1";
    result["full_bed_file_sha256_claimed"] = false;
    result["absolute_snapshot_or_lease_claimed"] = false;
    result["toctou_closed"] = false;
    result["toctou_nonclaim"] =
        "no_absolute_snapshot_or_lease_mutation_after_last_verified_read_remains_out_of_scope_v1";
    return result;
}

nb::dict ContextualReferenceExecutorV1::info() const {
    if (trait_mode_) {
        nb::dict result;
        result["contextual_native_api_version"] = 1;
        result["contextual_backend_version"] =
            stage2_detail::kContextualBackendVersionV1;
        result["contextual_backend"] = stage2_detail::kTraitBackendV1;
        result["contextual_execution_backend"] =
            stage2_detail::kExecutionBackendV1;
        result["contextual_build_id"] = GWLDCORE_SOURCE_COMMIT;
        result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
        result["legacy_module_api_version"] = 9;
        result["lifecycle"] = state_;
        result["n"] = n_;
        result["m"] = m_;
        result["q"] = q_;
        result["k"] = k_;
        result["components"] = components_;
        result["traits"] = l_;
        result["residual_components"] = h_;
        result["annotation_mode"] = annotation_mode_;
        result["phenotype_normalization"] = phenotype_normalization_;
        result["genotype_scale_policy"] = stage2_detail::kScalePolicyV1;
        result["numa_policy"] = numa_policy_;
        result["output_numa_node"] = output_numa_node_;
        result["complete_trait_artifact"] = false;
        result["complete_trait_statistics"] = true;
        result["file_identity_policy"] =
            "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2";
        result["file_content_identity"] = file_content_identity_dict();
        return result;
    }
    nb::dict result;
    result["contextual_native_api_version"] = 1;
    result["contextual_backend_version"] =
        stage2_detail::kContextualBackendVersionV1;
    result["contextual_backend"] = stage2_detail::kBackendV1;
    result["contextual_execution_backend"] = stage2_detail::kExecutionBackendV1;
    result["contextual_build_id"] = GWLDCORE_SOURCE_COMMIT;
    result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
    result["legacy_module_api_version"] = 9;
    result["lifecycle"] = state_;
    result["n"] = n_;
    result["m"] = m_;
    result["q"] = q_;
    result["k"] = k_;
    result["components"] = components_;
    result["sample_probes"] = bt_;
    result["probe_policy"] = probe_policy_;
    result["probe_identity_sha256"] = probe_identity_sha256_;
    result["annotation_mode"] = annotation_mode_;
    result["genotype_scale_policy"] = stage2_detail::kScalePolicyV1;
    result["numa_policy"] = numa_policy_;
    result["output_numa_node"] = output_numa_node_;
    result["complete_reference_artifact"] = false;
    if (stage3_enabled_) {
        result["complete_reference_statistics"] = true;
        result["variant_probes"] = bd_;
        result["variant_probe_policy"] = variant_probe_policy_;
        result["variant_probe_identity_sha256"] =
            variant_probe_identity_sha256_;
        result["selected_grouped_algorithm"] = grouped_algorithm_;
        result["group_execution_order_sha256"] =
            group_execution_order_sha256_;
    }
    result["file_identity_policy"] =
        "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2";
    result["file_content_identity"] = file_content_identity_dict();
    return result;
}

nb::dict ContextualReferenceExecutorV1::preflight() const {
    if (trait_mode_) {
        nb::dict result;
        result["permanent_bytes"] = estimate_.permanent_bytes;
        result["residual_phase_bytes"] = estimate_.residual_phase_bytes;
        result["trait_phase_bytes"] = estimate_.trait_phase_bytes;
        result["compact_output_bytes"] = estimate_.compact_output_bytes;
        result["integrity_reserve_bytes"] = estimate_.integrity_reserve_bytes;
        result["telemetry_bytes"] = estimate_.telemetry_bytes;
        result["required_workspace_bytes"] = estimate_.required_workspace_bytes;
        result["required_telemetry_capacity"] =
            estimate_.required_telemetry_capacity;
        result["maximum_protected_output_elements"] =
            estimate_.maximum_protected_output_elements;
        result["trait_descriptor_passes"] =
            estimate_.trait_descriptor_passes;
        result["trait_decoded_blocks"] = estimate_.trait_decoded_blocks;
        result["total_descriptor_passes"] =
            estimate_.trait_descriptor_passes;
        result["total_decoded_blocks"] = estimate_.trait_decoded_blocks;
        result["total_variant_record_visits"] =
            estimate_.trait_descriptor_passes * m_;
        result["total_protected_calls"] = estimate_.total_protected_calls;
        nb::dict calls;
        for (std::size_t index = 0;
             index < kContextualDenseSemanticCountV1; ++index)
            calls[kContextualDenseSemanticNamesV1[index]] =
                estimate_.calls[index];
        result["semantic_call_ledger"] = calls;
        nb::dict selected_tiles;
        selected_tiles["variant_block"] = variant_block_;
        selected_tiles["trait_feature_tile"] = trait_feature_tile_;
        result["selected_tiles"] = selected_tiles;

        nb::dict phase_ledger;
        nb::dict residual;
        residual["phenotype_projections"] = l_;
        phase_ledger["residual_moments"] = residual;
        nb::dict trait;
        trait["descriptor_passes"] = estimate_.trait_descriptor_passes;
        trait["decoded_blocks"] = estimate_.trait_decoded_blocks;
        trait["variant_record_visits"] =
            estimate_.trait_descriptor_passes * m_;
        trait["logical_bed_record_bytes_touched"] =
            estimate_.trait_descriptor_passes * m_ * bytes_per_snp_;
        trait["access_order"] = "retained_logical_sequential_v1";
        trait["logical_duplicate_variant_decodes"] = 0;
        trait["phenotypes_in_packed_rhs"] = l_;
        phase_ledger["trait"] = trait;
        result["phase_ledger"] = phase_ledger;

        nb::dict memory;
        nb::dict lifetimes;
        const auto add_memory = [&](const char* name, std::uint64_t bytes,
                                    const char* live_phase,
                                    std::uint64_t concurrent_copies) {
            memory[name] = bytes;
            nb::dict item;
            item["byte_count"] = bytes;
            item["live_phase"] = live_phase;
            item["concurrent_copies"] = concurrent_copies;
            lifetimes[name] = item;
        };
        add_memory("permanent_bytes", estimate_.permanent_bytes,
                   "admission_through_publication", 1);
        add_memory("residual_phase_bytes", estimate_.residual_phase_bytes,
                   "residual_moments_and_trait_preallocated_arena", 1);
        add_memory("trait_phase_bytes", estimate_.trait_phase_bytes,
                   "preallocated_trait_arena", 1);
        add_memory("compact_output_bytes", estimate_.compact_output_bytes,
                   "publication", 1);
        add_memory("integrity_reserve_bytes",
                   estimate_.integrity_reserve_bytes,
                   "all_protected_phases", 4);
        add_memory("telemetry_bytes", estimate_.telemetry_bytes,
                   "admission_through_publication", 1);
        result["memory_ledger"] = memory;
        result["memory_lifetimes"] = lifetimes;
        result["memory_accounting_model"] =
            "tracked_vector_payload_bytes_v1_excludes_allocator_metadata_and_small_strings";
        result["os_physical_read_bytes_measured"] = false;
        result["os_page_faults_measured"] = false;
        result["physical_io_evidence"] =
            "logical_mmap_record_touches_only_v1";
        return result;
    }
    nb::dict result;
    result["permanent_bytes"] = estimate_.permanent_bytes;
    result["source_phase_bytes"] = estimate_.source_phase_bytes;
    result["action_phase_bytes"] = estimate_.action_phase_bytes;
    if (stage3_enabled_) {
        result["group_phase_bytes"] = estimate_.group_phase_bytes;
        result["same_person_phase_bytes"] = estimate_.same_person_phase_bytes;
    }
    result["compact_output_bytes"] = estimate_.compact_output_bytes;
    result["integrity_reserve_bytes"] = estimate_.integrity_reserve_bytes;
    result["telemetry_bytes"] = estimate_.telemetry_bytes;
    result["required_workspace_bytes"] = estimate_.required_workspace_bytes;
    result["required_telemetry_capacity"] =
        estimate_.required_telemetry_capacity;
    result["maximum_protected_output_elements"] =
        estimate_.maximum_protected_output_elements;
    result["resident_batches"] = estimate_.resident_batches;
    result["source_descriptor_passes"] = estimate_.source_descriptor_passes;
    result["action_descriptor_passes"] = estimate_.action_descriptor_passes;
    result["source_decoded_blocks"] = estimate_.source_decoded_blocks;
    result["action_decoded_blocks"] = estimate_.action_decoded_blocks;
    if (stage3_enabled_) {
        result["group_descriptor_passes"] = estimate_.group_descriptor_passes;
        result["group_decoded_blocks"] = estimate_.group_decoded_blocks;
        result["same_person_descriptor_passes"] =
            estimate_.same_person_descriptor_passes;
        result["same_person_decoded_blocks"] =
            estimate_.same_person_decoded_blocks;
    }
    const std::uint64_t total_descriptor_passes =
        estimate_.source_descriptor_passes +
        estimate_.action_descriptor_passes +
        estimate_.group_descriptor_passes +
        estimate_.same_person_descriptor_passes;
    const std::uint64_t total_decoded_blocks =
        estimate_.source_decoded_blocks + estimate_.action_decoded_blocks +
        estimate_.group_decoded_blocks + estimate_.same_person_decoded_blocks;
    result["total_descriptor_passes"] = total_descriptor_passes;
    result["total_decoded_blocks"] = total_decoded_blocks;
    result["total_variant_record_visits"] =
        total_descriptor_passes * m_;
    result["total_protected_calls"] = estimate_.total_protected_calls;
    nb::dict calls;
    for (std::size_t index = 0; index < kContextualDenseSemanticCountV1; ++index)
        calls[kContextualDenseSemanticNamesV1[index]] = estimate_.calls[index];
    result["semantic_call_ledger"] = calls;
    nb::dict selected_tiles;
    selected_tiles["variant_block"] = variant_block_;
    selected_tiles["sample_probe_resident"] = sample_probe_resident_;
    selected_tiles["sample_probe_tile"] = sample_probe_tile_;
    selected_tiles["action_tile"] = action_tile_;
    selected_tiles["annotation_tile"] = annotation_tile_;
    selected_tiles["context_tile"] = context_tile_;
    if (stage3_enabled_) {
        selected_tiles["variant_probe_tile"] = variant_probe_tile_;
        selected_tiles["group_tile"] = group_tile_;
    }
    result["selected_tiles"] = selected_tiles;
    nb::dict phase_ledger;
    nb::dict source;
    source["descriptor_passes"] = estimate_.source_descriptor_passes;
    source["decoded_blocks"] = estimate_.source_decoded_blocks;
    source["variant_record_visits"] = estimate_.source_descriptor_passes * m_;
    phase_ledger["source"] = source;
    nb::dict action;
    action["descriptor_passes"] = estimate_.action_descriptor_passes;
    action["decoded_blocks"] = estimate_.action_decoded_blocks;
    action["variant_record_visits"] = estimate_.action_descriptor_passes * m_;
    phase_ledger["action"] = action;
    if (stage3_enabled_) {
        source["logical_bed_record_bytes_touched"] =
            estimate_.source_descriptor_passes * m_ * bytes_per_snp_;
        source["access_order"] = "retained_logical_sequential_v1";
        source["logical_duplicate_variant_decodes"] =
            (estimate_.source_descriptor_passes - 1U) * m_;
        action["logical_bed_record_bytes_touched"] =
            estimate_.action_descriptor_passes * m_ * bytes_per_snp_;
        action["access_order"] = "retained_logical_sequential_v1";
        action["logical_duplicate_variant_decodes"] =
            (estimate_.action_descriptor_passes - 1U) * m_;
        nb::dict grouped;
        grouped["descriptor_passes"] = estimate_.group_descriptor_passes;
        grouped["decoded_blocks"] = estimate_.group_decoded_blocks;
        grouped["variant_record_visits"] =
            estimate_.group_descriptor_passes * m_;
        grouped["execution_order"] = "sealed_group_contiguous_permutation_v1";
        grouped["access_order"] = enable_grouped_differential_ ||
                grouped_algorithm_ == "direct_grouped_tn_v1"
            ? "sealed_group_indexed_plus_retained_logical_sequential_v1"
            : "sealed_group_indexed_v1";
        grouped["logical_bed_record_bytes_touched"] =
            estimate_.group_descriptor_passes * m_ * bytes_per_snp_;
        grouped["logical_duplicate_variant_decodes"] =
            (estimate_.group_descriptor_passes - 1U) * m_;
        grouped["group_restricted_duplicate_variant_decodes"] = 0;
        grouped["group_restricted_variant_coverage"] = m_;
        grouped["selected_algorithm"] = grouped_algorithm_;
        phase_ledger["grouped"] = grouped;
        nb::dict same;
        same["descriptor_passes"] =
            estimate_.same_person_descriptor_passes;
        same["decoded_blocks"] = estimate_.same_person_decoded_blocks;
        same["variant_record_visits"] =
            estimate_.same_person_descriptor_passes * m_;
        same["global_probe_merge_before_finalization"] = true;
        same["access_order"] = "retained_logical_sequential_v1";
        same["logical_bed_record_bytes_touched"] =
            estimate_.same_person_descriptor_passes * m_ * bytes_per_snp_;
        same["logical_duplicate_variant_decodes"] =
            (estimate_.same_person_descriptor_passes - 1U) * m_;
        phase_ledger["same_person"] = same;
        result["os_physical_read_bytes_measured"] = false;
        result["os_page_faults_measured"] = false;
        result["physical_io_evidence"] =
            "logical_mmap_record_touches_only_v1";
    }
    result["phase_ledger"] = phase_ledger;
    nb::dict memory;
    nb::dict memory_lifetimes;
    const auto add_memory = [&](const char* name, std::uint64_t bytes,
                                const char* live_phase,
                                std::uint64_t concurrent_copies) {
        memory[name] = bytes;
        nb::dict item;
        item["byte_count"] = bytes;
        item["live_phase"] = live_phase;
        item["concurrent_copies"] = concurrent_copies;
        memory_lifetimes[name] = item;
    };
    add_memory("permanent_bytes", estimate_.permanent_bytes,
               "admission_through_publication", 1);
    add_memory("source_phase_bytes", estimate_.source_phase_bytes,
               "source_and_action_preallocated_arena", 1);
    add_memory("action_phase_bytes", estimate_.action_phase_bytes,
               "source_and_action_preallocated_arena", 1);
    if (stage3_enabled_) {
        add_memory("group_phase_bytes", estimate_.group_phase_bytes,
                   "preallocated_group_arena", 1);
        add_memory("same_person_phase_bytes",
                   estimate_.same_person_phase_bytes,
                   "preallocated_same_person_arena", 1);
    }
    add_memory("compact_output_bytes", estimate_.compact_output_bytes,
               "publication", 1);
    add_memory("integrity_reserve_bytes", estimate_.integrity_reserve_bytes,
               "all_protected_phases", 4);
    add_memory("telemetry_bytes", estimate_.telemetry_bytes,
               "admission_through_publication", 1);
    result["memory_ledger"] = memory;
    result["memory_lifetimes"] = memory_lifetimes;
    result["memory_accounting_model"] =
        "tracked_vector_payload_bytes_v1_excludes_allocator_metadata_and_small_strings";
    const std::uint64_t dense_columns = stage2_detail::checked_mul(
        stage2_detail::checked_mul(
            stage2_detail::checked_mul(m_, k_, "dense target columns"), q_,
            "dense target columns"), bt_, "dense target columns");
    const std::uint64_t useful_columns = strict_annotation_ids_.empty()
        ? dense_columns : stage2_detail::checked_mul(
            stage2_detail::checked_mul(m_, q_, "useful target columns"), bt_,
            "useful target columns");
    result["target_dense_columns"] = dense_columns;
    result["target_useful_columns"] = useful_columns;
    result["strict_zero_weight_columns_eliminated"] =
        dense_columns - useful_columns;
    return result;
}

nb::dict ContextualReferenceExecutorV1::publish_trait_compact() const {
    check_files_unchanged();
    check_hardened_file_content_unchanged();
    check_inputs_unchanged();
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualTraitExecutorV1 scientific output mutation detected at publication entry"
        );

    std::vector<double> group_masses(groups_ * k_, 0.0);
    for (std::size_t group = 0; group < groups_; ++group)
        for (std::size_t annotation = 0; annotation < k_; ++annotation)
            group_masses[group * k_ + annotation] =
                group_annotation_masses_[group + groups_ * annotation];

    nb::dict result;
    result["complete_trait_artifact"] = false;
    result["complete_trait_statistics"] = true;
    result["internal_result_kind"] =
        "stage4_complete_trait_statistics_v1";
    result["genetic_rhs"] = stage2_readonly_owned_mat(
        genetic_rhs_, components_, l_
    );
    result["genetic_traces"] = stage2_readonly_owned_vec(genetic_traces_);
    result["genetic_residual"] = stage2_readonly_owned_mat(
        genetic_residual_, components_, h_
    );
    result["residual_rhs"] = stage2_readonly_owned_mat(
        residual_rhs_, h_, l_
    );
    result["residual_traces"] = stage2_readonly_owned_vec(residual_traces_);
    result["residual_gram"] = stage2_readonly_owned_mat(
        residual_gram_, h_, h_
    );
    result["group_rhs_unnormalized_num"] = stage2_readonly_owned_arr3(
        group_rhs_, groups_, components_, l_
    );
    result["group_trace_unnormalized_num"] = stage2_readonly_owned_mat(
        group_trace_, groups_, components_
    );
    result["group_genetic_residual_num"] = stage2_readonly_owned_arr3(
        group_genetic_residual_, groups_, components_, h_
    );
    result["annotation_masses"] = stage2_readonly_owned_vec(
        annotation_masses_
    );
    result["group_annotation_masses"] = stage2_readonly_owned_mat(
        group_masses, groups_, k_
    );
    result["group_variant_counts"] = stage2_readonly_owned_vec(
        group_variant_counts_
    );
    result["study_n"] = n_;
    result["n_variants"] = m_;
    result["residual_rank"] = n_ - rank_;
    result["q"] = q_;
    result["trait_count"] = l_;
    result["residual_component_count"] = h_;
    result["numeric_policy"] = "fp64_v1";
    result["deletion_semantics"] = stage2_detail::kDeletionPolicyV1;
    result["grouped_encoding"] = stage2_detail::kGroupedEncodingV1;
    result["group_numerator_unit"] =
        "raw_source_annotation_mass_v1";
    result["grouped_values_are_unnormalized_numerators"] = true;
    result["residual_only_deletion"] = "reuse_full_unchanged";
    result["phenotype_normalization"] = phenotype_normalization_;
    result["phenotypes_projected_once"] = true;
    result["feature_mode"] = "P_diag_phi_G_v1";
    result["pair_q"] = stage2_readonly_owned_vec(pair_q_);
    result["pair_r"] = stage2_readonly_owned_vec(pair_r_);
    result["pair_eta"] = stage2_readonly_owned_vec(pair_eta_);
    result["component_annotation"] = stage2_readonly_owned_vec(
        component_annotation_
    );
    result["component_pair"] = stage2_readonly_owned_vec(component_pair_);
    nb::dict scientific_array_sha256;
    scientific_array_sha256["genetic_rhs"] =
        stage2_detail::array_sha256_f64_c_order(
            genetic_rhs_, {components_, l_}
        );
    scientific_array_sha256["genetic_traces"] =
        stage2_detail::array_sha256_f64_c_order(
            genetic_traces_, {components_}
        );
    scientific_array_sha256["genetic_residual"] =
        stage2_detail::array_sha256_f64_c_order(
            genetic_residual_, {components_, h_}
        );
    scientific_array_sha256["residual_rhs"] =
        stage2_detail::array_sha256_f64_c_order(residual_rhs_, {h_, l_});
    scientific_array_sha256["residual_traces"] =
        stage2_detail::array_sha256_f64_c_order(residual_traces_, {h_});
    scientific_array_sha256["residual_gram"] =
        stage2_detail::array_sha256_f64_c_order(residual_gram_, {h_, h_});
    scientific_array_sha256["group_rhs_unnormalized_num"] =
        stage2_detail::array_sha256_f64_c_order(
            group_rhs_, {groups_, components_, l_}
        );
    scientific_array_sha256["group_trace_unnormalized_num"] =
        stage2_detail::array_sha256_f64_c_order(
            group_trace_, {groups_, components_}
        );
    scientific_array_sha256["group_genetic_residual_num"] =
        stage2_detail::array_sha256_f64_c_order(
            group_genetic_residual_, {groups_, components_, h_}
        );
    scientific_array_sha256["annotation_masses"] =
        stage2_detail::array_sha256_f64_c_order(annotation_masses_, {k_});
    scientific_array_sha256["group_annotation_masses"] =
        stage2_detail::array_sha256_f64_c_order(
            group_masses, {groups_, k_}
        );
    scientific_array_sha256["group_variant_counts"] =
        stage2_detail::array_sha256_i64_c_order(
            group_variant_counts_, {groups_}
        );
    scientific_array_sha256["pair_q"] =
        stage2_detail::array_sha256_i64_c_order(pair_q_, {pairs_});
    scientific_array_sha256["pair_r"] =
        stage2_detail::array_sha256_i64_c_order(pair_r_, {pairs_});
    scientific_array_sha256["pair_eta"] =
        stage2_detail::array_sha256_i64_c_order(pair_eta_, {pairs_});
    scientific_array_sha256["component_annotation"] =
        stage2_detail::array_sha256_i64_c_order(
            component_annotation_, {components_}
        );
    scientific_array_sha256["component_pair"] =
        stage2_detail::array_sha256_i64_c_order(
            component_pair_, {components_}
        );
    result["scientific_array_sha256"] = scientific_array_sha256;
    nb::dict phase_evidence_sha256;
    phase_evidence_sha256["residual_derived_state"] =
        residual_derived_state_sha256_;
    phase_evidence_sha256["post_trait_outputs"] =
        trait_output_phase_sha256_;
    result["phase_evidence_sha256"] = phase_evidence_sha256;
    result["annotation_names"] = annotation_names_;
    result["group_names"] = group_names_;
    result["trait_names"] = trait_names_;
    result["residual_names"] = residual_names_;
    result["annotation_mode"] = annotation_mode_;
    result["genotype_scale_policy"] = stage2_detail::kScalePolicyV1;
    result["allele_orientation"] = allele_orientation_;
    result["allele_coding"] = stage2_detail::kAlleleCodingV1;
    result["centering_source"] = centering_source_;
    result["centering_formula"] = stage2_detail::kCenteringFormulaV1;
    result["scaling_formula"] = stage2_detail::kScalingFormulaV1;
    result["missing_imputation"] = stage2_detail::kMissingImputationV1;
    result["ploidy_policy"] = stage2_detail::kPloidyPolicyV1;
    result["retained_variant_order_sha256"] =
        observed_retained_order_sha256_;
    result["affine_mean_sha256"] = observed_mean_sha256_;
    result["affine_inverse_scale_sha256"] = observed_inverse_sha256_;
    result["scale_plan_sha256"] = observed_scale_plan_sha256_;
    result["retained_sample_map_sha256"] = retained_sample_map_sha256_;
    result["variant_order_allele_sha256"] = variant_order_allele_sha256_;
    result["fixed_basis_sha256"] = fixed_basis_sha256_;
    result["evaluated_phi_sha256"] = evaluated_phi_sha256_;
    result["annotation_map_sha256"] = annotation_map_sha256_;
    result["group_map_sha256"] = group_map_sha256_;
    result["phenotype_batch_sha256"] = phenotype_batch_sha256_;
    result["residual_basis_sha256"] = residual_basis_sha256_;
    result["missingness_sha256"] = observed_missingness_sha256_;
    result["sealed_plan_sha256"] = sealed_plan_sha256_;
    result["execution_plan_sha256"] = execution_plan_sha256_;
    result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
    result["contextual_native_api_version"] = 1;
    result["contextual_backend_version"] =
        stage2_detail::kContextualBackendVersionV1;
    result["contextual_backend"] = stage2_detail::kTraitBackendV1;
    result["contextual_execution_backend"] =
        stage2_detail::kExecutionBackendV1;
    result["contextual_build_id"] = GWLDCORE_SOURCE_COMMIT;
    result["build_provenance"] = stage2_detail::build_provenance_dict();
    result["file_identity_policy"] =
        "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2";
    result["file_content_identity"] = file_content_identity_dict();

    nb::dict ownership;
    ownership["owns_data"] = true;
    ownership["read_only"] = true;
    result["output_ownership"] = ownership;
    nb::dict numa;
    numa["numa_applicable"] = false;
    numa["numa_verified"] = false;
    numa["policy"] = numa_policy_;
    numa["output_numa_node"] = output_numa_node_;
    numa["reason"] =
        "unbound standard allocator; no placement or output-node claim";
    result["numa"] = numa;

    nb::dict protected_counts;
    for (std::size_t index = 0;
         index < kContextualDenseSemanticCountV1; ++index)
        protected_counts[kContextualDenseSemanticNamesV1[index]] =
            observed_calls_[index];
    nb::dict diagnostics;
    diagnostics["maximum_projection_leakage"] = maximum_projection_leakage_;
    diagnostics["phenotype_projection_leakage_max_abs"] =
        phenotype_projection_leakage_max_abs_;
    diagnostics["phenotype_normalized_norm_error_max_abs"] =
        phenotype_normalized_norm_error_max_abs_;
    diagnostics["residual_gram_pre_symmetry_max_abs"] =
        residual_gram_pre_symmetry_max_abs_;
    diagnostics["trait_group_reconstruction_max_abs"] =
        trait_group_reconstruction_max_abs_;
    diagnostics["missing_genotype_calls"] = std::accumulate(
        expected_missing_counts_.begin(), expected_missing_counts_.end(),
        std::int64_t{0}
    );
    diagnostics["observed_descriptor_passes"] = observed_descriptor_passes_;
    diagnostics["observed_decoded_blocks"] = observed_decoded_blocks_;
    diagnostics["observed_variant_record_visits"] = observed_variant_visits_;
    diagnostics["observed_phenotype_projections"] =
        observed_phenotype_projections_;
    diagnostics["protected_call_counts"] = protected_counts;
    diagnostics["descriptor_accounting_verified"] = true;
    diagnostics["semantic_call_ledger_exact"] = true;
    diagnostics["files_unchanged_at_all_checkpoints"] = true;
    diagnostics["bed_content_evidence"] =
        "boundary_sample_only_no_extra_full_BED_traversal";
    diagnostics["inputs_unchanged_at_all_checkpoints"] = true;
    diagnostics["scratch_released"] = scratch_released_;
    diagnostics["scratch_released_before_publication"] = scratch_released_;
    diagnostics["all_large_buffers_preallocated_before_decode"] = true;
    diagnostics["runtime_large_allocations_after_decode"] = 0;
    diagnostics["tracked_high_water_bytes"] =
        estimate_.required_workspace_bytes;
    diagnostics["tracked_high_water_within_admission"] =
        estimate_.required_workspace_bytes <= workspace_cap_bytes_;
    diagnostics["strict_disjoint_optimized_path"] =
        !strict_annotation_ids_.empty();
    diagnostics["trait_group_reconstruction_verified"] = true;
    diagnostics["phenotype_projection_count_verified"] = true;
    diagnostics["phenotype_normalization_verified"] = true;
    diagnostics["operand_fingerprints_verified"] = true;
    diagnostics["runtime_thread_affinity_fingerprints_verified"] = true;
    diagnostics["independent_scalar_witness_verified"] = true;
    diagnostics["independent_scalar_fallback_available"] = true;
    diagnostics["final_output_identity_fnv64"] =
        std::to_string(output_fingerprint_);
    result["diagnostics"] = diagnostics;

    nb::dict telemetry;
    telemetry["capacity"] = telemetry_capacity_;
    telemetry["required_capacity"] = estimate_.required_telemetry_capacity;
    telemetry["observed_events"] = events_.size();
    telemetry["injection_count"] = injection_count_;
    telemetry["repair_count"] = repair_count_;
    telemetry["retry_count"] = retry_count_;
    telemetry["fallback_count"] = fallback_count_;
    telemetry["complete_without_drop"] = true;
    telemetry["fault_operation"] = fault_operation_;
    telemetry["fault_mode"] = fault_mode_;
    telemetry["fault_semantic_anchor"] = fault_semantic_anchor_;
    telemetry["protected_call_counts"] = protected_counts;
    nb::list event_list;
    for (const auto& event : events_)
        event_list.append(stage2_detail::event_dict(event));
    telemetry["events"] = event_list;
    result["telemetry"] = telemetry;
    result["admission"] = preflight();
    result["state"] = state_;
    result["lifecycle"] = state_;
    check_files_unchanged();
    check_hardened_file_content_unchanged();
    check_inputs_unchanged();
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualTraitExecutorV1 scientific output mutation detected at publication exit"
        );
    return result;
}

nb::dict ContextualReferenceExecutorV1::publish_compact() const {
    if (trait_mode_) return publish_trait_compact();
    check_files_unchanged();
    check_hardened_file_content_unchanged();
    check_inputs_unchanged();
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 scientific output mutation detected at publication entry"
        );
    nb::dict result;
    result["complete_reference_artifact"] = false;
    result["file_identity_policy"] =
        "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2";
    result["file_content_identity"] = file_content_identity_dict();
    result["internal_result_kind"] = stage3_enabled_
        ? stage2_detail::kStage3ResultKindV1
        : "stage2_source_action_full_gram_v1";
    if (stage3_enabled_)
        result["complete_reference_statistics"] = true;
    result["gram"] = stage2_readonly_owned_mat(
        gram_, components_, components_
    );
    result["raw_gram_numerator"] = stage2_readonly_owned_mat(
        raw_gram_numerator_, components_, components_
    );
    result["annotation_masses"] = stage2_readonly_owned_vec(
        annotation_masses_
    );
    nb::dict scientific_array_sha256;
    scientific_array_sha256["gram"] =
        stage2_detail::array_sha256_f64_c_order(
            gram_, {components_, components_}
        );
    scientific_array_sha256["raw_gram_numerator"] =
        stage2_detail::array_sha256_f64_c_order(
            raw_gram_numerator_, {components_, components_}
        );
    scientific_array_sha256["annotation_masses"] =
        stage2_detail::array_sha256_f64_c_order(annotation_masses_, {k_});
    scientific_array_sha256["pair_q"] =
        stage2_detail::array_sha256_i64_c_order(pair_q_, {pairs_});
    scientific_array_sha256["pair_r"] =
        stage2_detail::array_sha256_i64_c_order(pair_r_, {pairs_});
    scientific_array_sha256["pair_eta"] =
        stage2_detail::array_sha256_i64_c_order(pair_eta_, {pairs_});
    scientific_array_sha256["component_annotation"] =
        stage2_detail::array_sha256_i64_c_order(
            component_annotation_, {components_}
        );
    scientific_array_sha256["component_pair"] =
        stage2_detail::array_sha256_i64_c_order(
            component_pair_, {components_}
        );
    if (stage3_enabled_) {
        result["same_person"] = stage2_readonly_owned_mat(
            same_person_, components_, components_
        );
        result["group_gram_unnormalized_num"] =
            stage2_readonly_owned_arr3(
                group_gram_numerator_, groups_, components_, components_
            );
        std::vector<double> group_masses_row_major(
            groups_ * k_, 0.0
        );
        for (std::size_t group = 0; group < groups_; ++group)
            for (std::size_t annotation = 0; annotation < k_; ++annotation)
                group_masses_row_major[group * k_ + annotation] =
                    group_annotation_masses_[group + groups_ * annotation];
        result["group_annotation_masses"] = stage2_readonly_owned_mat(
            group_masses_row_major, groups_, k_
        );
        result["group_variant_counts"] = stage2_readonly_owned_vec(
            group_variant_counts_
        );
        scientific_array_sha256["same_person"] =
            stage2_detail::array_sha256_f64_c_order(
                same_person_, {components_, components_}
            );
        scientific_array_sha256["group_gram_unnormalized_num"] =
            stage2_detail::array_sha256_f64_c_order(
                group_gram_numerator_, {groups_, components_, components_}
            );
        scientific_array_sha256["group_annotation_masses"] =
            stage2_detail::array_sha256_f64_c_order(
                group_masses_row_major, {groups_, k_}
            );
        scientific_array_sha256["group_variant_counts"] =
            stage2_detail::array_sha256_i64_c_order(
                group_variant_counts_, {groups_}
            );
        result["reference_n"] = n_;
        result["n_variants"] = m_;
        result["residual_rank"] = n_ - rank_;
        result["q"] = q_;
        result["numeric_policy"] = "fp64_v1";
        result["deletion_semantics"] = stage2_detail::kDeletionPolicyV1;
        result["same_person_deletion"] = "reuse_full_unchanged";
        result["grouped_encoding"] = stage2_detail::kGroupedEncodingV1;
        result["group_numerator_unit"] =
            "mass_squared_fixed_probe_raw_action_v1";
        result["group_gram_raw_unit"] =
            "(1/B_T)_sum_raw_cross_products_annotation_mass_unnormalized_v1";
        result["grouped_values_are_unnormalized_numerators"] = true;
        result["same_person_signed_preserved"] = true;
        result["selected_grouped_algorithm"] = grouped_algorithm_;
        result["grouped_attribution_algorithm"] = grouped_algorithm_;
        result["direct_grouped_scaling"] = direct_grouped_scaling_;
        result["grouped_differential_enabled"] =
            enable_grouped_differential_;
        result["group_execution_order"] =
            "stable_group_contiguous_logical_index_permutation_v1";
        result["group_execution_order_sha256"] =
            group_execution_order_sha256_;
        result["group_execution_permutation_sha256"] =
            group_execution_order_sha256_;
        result["sealed_plan_sha256"] = sealed_plan_sha256_;
        result["sample_probe_policy"] = probe_policy_;
        result["sample_probe_identity_sha256"] = probe_identity_sha256_;
        result["sample_probe_count"] = bt_;
        result["variant_probe_policy"] = variant_probe_policy_;
        result["variant_probe_identity_sha256"] =
            variant_probe_identity_sha256_;
        result["variant_probe_count"] = bd_;
        result["retained_sample_map_sha256"] =
            retained_sample_map_sha256_;
        result["variant_order_allele_sha256"] =
            variant_order_allele_sha256_;
        result["fixed_basis_sha256"] = fixed_basis_sha256_;
        result["evaluated_phi_sha256"] = evaluated_phi_sha256_;
        result["annotation_map_sha256"] = annotation_map_sha256_;
        result["group_map_sha256"] = group_map_sha256_;
        result["same_person_sha256"] =
            stage2_detail::array_sha256_f64_c_order(
                same_person_, {components_, components_}
            );
        result["group_gram_unnormalized_num_sha256"] =
            stage2_detail::array_sha256_f64_c_order(
                group_gram_numerator_,
                {groups_, components_, components_}
            );
    }
    result["scientific_array_sha256"] = scientific_array_sha256;
    nb::dict phase_evidence_sha256;
    phase_evidence_sha256["post_gram"] = gram_phase_sha256_;
    result["phase_evidence_sha256"] = phase_evidence_sha256;
    result["pair_q"] = stage2_readonly_owned_vec(pair_q_);
    result["pair_r"] = stage2_readonly_owned_vec(pair_r_);
    result["pair_eta"] = stage2_readonly_owned_vec(pair_eta_);
    result["component_annotation"] = stage2_readonly_owned_vec(
        component_annotation_
    );
    result["component_pair"] = stage2_readonly_owned_vec(component_pair_);
    result["annotation_names"] = annotation_names_;
    result["group_names"] = group_names_;
    result["annotation_mode"] = annotation_mode_;
    result["probe_policy"] = probe_policy_;
    result["probe_identity_sha256"] = probe_identity_sha256_;
    result["genotype_scale_policy"] = stage2_detail::kScalePolicyV1;
    result["allele_orientation"] = allele_orientation_;
    result["allele_coding"] = stage2_detail::kAlleleCodingV1;
    result["centering_formula"] = stage2_detail::kCenteringFormulaV1;
    result["scaling_formula"] = stage2_detail::kScalingFormulaV1;
    result["missing_imputation"] = stage2_detail::kMissingImputationV1;
    result["ploidy_policy"] = stage2_detail::kPloidyPolicyV1;
    result["centering_source"] = centering_source_;
    result["retained_variant_order_sha256"] =
        observed_retained_order_sha256_;
    result["affine_mean_sha256"] = observed_mean_sha256_;
    result["affine_inverse_scale_sha256"] = observed_inverse_sha256_;
    result["missingness_sha256"] = observed_missingness_sha256_;
    result["scale_plan_sha256"] = observed_scale_plan_sha256_;
    result["contextual_native_api_version"] = 1;
    result["contextual_backend_version"] =
        stage2_detail::kContextualBackendVersionV1;
    result["contextual_backend"] = stage2_detail::kBackendV1;
    result["contextual_execution_backend"] = stage2_detail::kExecutionBackendV1;
    result["contextual_build_id"] = GWLDCORE_SOURCE_COMMIT;
    result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
    result["execution_plan_sha256"] = execution_plan_sha256_;
    result["build_provenance"] = stage2_detail::build_provenance_dict();

    nb::dict ownership;
    ownership["owns_data"] = true;
    ownership["read_only"] = true;
    result["output_ownership"] = ownership;
    nb::dict numa;
    numa["numa_applicable"] = false;
    numa["numa_verified"] = false;
    numa["policy"] = numa_policy_;
    numa["output_numa_node"] = output_numa_node_;
    numa["reason"] =
        "unbound standard allocator; no placement or output-node claim";
    result["numa"] = numa;

    nb::dict protected_counts;
    for (std::size_t index = 0; index < kContextualDenseSemanticCountV1; ++index)
        protected_counts[kContextualDenseSemanticNamesV1[index]] =
            observed_calls_[index];
    nb::dict diagnostics;
    diagnostics["maximum_projection_leakage"] = maximum_projection_leakage_;
    diagnostics["gram_pre_symmetry_max_abs"] = gram_pre_symmetry_max_abs_;
    diagnostics["missing_genotype_calls"] = std::accumulate(
        expected_missing_counts_.begin(), expected_missing_counts_.end(),
        std::int64_t{0}
    );
    diagnostics["observed_descriptor_passes"] = observed_descriptor_passes_;
    diagnostics["observed_decoded_blocks"] = observed_decoded_blocks_;
    diagnostics["observed_variant_record_visits"] = observed_variant_visits_;
    diagnostics["protected_call_counts"] = protected_counts;
    diagnostics["descriptor_accounting_verified"] = true;
    diagnostics["semantic_call_ledger_exact"] = true;
    diagnostics["files_unchanged_at_all_checkpoints"] = true;
    diagnostics["bed_content_evidence"] =
        "boundary_sample_only_no_extra_full_BED_traversal";
    diagnostics["inputs_unchanged_at_all_checkpoints"] = true;
    diagnostics["scratch_released"] = scratch_released_;
    diagnostics["scratch_released_before_publication"] = scratch_released_;
    diagnostics["all_large_buffers_preallocated_before_decode"] = true;
    diagnostics["runtime_large_allocations_after_decode"] = 0;
    diagnostics["tracked_high_water_bytes"] = estimate_.required_workspace_bytes;
    diagnostics["tracked_high_water_within_admission"] =
        estimate_.required_workspace_bytes <= workspace_cap_bytes_;
    diagnostics["strict_disjoint_optimized_path"] =
        !strict_annotation_ids_.empty();
    if (stage3_enabled_) {
        diagnostics["same_person_pre_symmetry_max_abs"] =
            same_person_pre_symmetry_max_abs_;
        diagnostics["group_reconstruction_max_abs"] =
            group_reconstruction_max_abs_;
        diagnostics["direct_action_restricted_max_abs"] =
            direct_action_restricted_max_abs_;
        diagnostics["direct_genotype_restricted_max_abs"] =
            direct_genotype_restricted_max_abs_;
        diagnostics["direct_placement_max_abs"] =
            direct_placement_max_abs_;
        diagnostics["group_reconstruction_verified"] = true;
        diagnostics["same_person_global_merge_verified"] = true;
        diagnostics["same_person_cross_tile_pairs_included"] = true;
        diagnostics["variant_probe_coverage"] =
            observed_variant_probe_coverage_;
        diagnostics["group_execution_variant_visits"] =
            observed_group_execution_visits_;
        diagnostics["group_execution_permutation_coverage_verified"] = true;
        diagnostics["group_execution_permutation_unique_verified"] = true;
        diagnostics["group_execution_contiguous_verified"] = true;
        diagnostics["group_tile_batches"] =
            (groups_ + group_tile_ - 1U) / group_tile_;
        diagnostics["logical_order_missing_identity_verified"] = true;
        diagnostics["same_person_signed_output_preserved"] = true;
    }
    const std::uint64_t dense_columns = stage2_detail::checked_mul(
        stage2_detail::checked_mul(
            stage2_detail::checked_mul(m_, k_, "dense target columns"), q_,
            "dense target columns"), bt_, "dense target columns");
    const std::uint64_t useful_columns = strict_annotation_ids_.empty()
        ? dense_columns : stage2_detail::checked_mul(
            stage2_detail::checked_mul(m_, q_, "useful target columns"), bt_,
            "useful target columns");
    diagnostics["target_dense_columns"] = dense_columns;
    diagnostics["target_useful_columns"] = useful_columns;
    diagnostics["strict_zero_weight_columns_eliminated"] =
        dense_columns - useful_columns;
    diagnostics["operand_fingerprints_verified"] = true;
    diagnostics["runtime_thread_affinity_fingerprints_verified"] = true;
    diagnostics["independent_scalar_witness_verified"] = true;
    diagnostics["independent_scalar_fallback_available"] = true;
    diagnostics["source_identity_fnv64"] = std::to_string(source_fingerprint_);
    diagnostics["source_action_identity_fnv64"] = std::to_string(action_fingerprint_);
    diagnostics["final_output_identity_fnv64"] = std::to_string(output_fingerprint_);
    result["diagnostics"] = diagnostics;

    nb::dict telemetry;
    telemetry["capacity"] = telemetry_capacity_;
    telemetry["required_capacity"] = estimate_.required_telemetry_capacity;
    telemetry["observed_events"] = events_.size();
    telemetry["injection_count"] = injection_count_;
    telemetry["repair_count"] = repair_count_;
    telemetry["retry_count"] = retry_count_;
    telemetry["fallback_count"] = fallback_count_;
    telemetry["complete_without_drop"] = true;
    telemetry["fault_operation"] = fault_operation_;
    telemetry["fault_mode"] = fault_mode_;
    telemetry["fault_semantic_anchor"] = fault_semantic_anchor_;
    telemetry["protected_call_counts"] = protected_counts;
    nb::list event_list;
    for (const auto& event : events_)
        event_list.append(stage2_detail::event_dict(event));
    telemetry["events"] = event_list;
    result["telemetry"] = telemetry;
    result["admission"] = preflight();
    result["state"] = state_;
    result["lifecycle"] = state_;
    check_files_unchanged();
    check_hardened_file_content_unchanged();
    check_inputs_unchanged();
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 scientific output mutation detected at publication exit"
        );
    return result;
}

nb::dict ContextualReferenceExecutorV1::run() {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (state_ != "admitted")
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 is one-shot and lifecycle is not admitted"
        );
    stage2_detail::PerformancePhaseScopeV1 total_performance_scope(
        performance_phases_[static_cast<std::size_t>(
            stage2_detail::PerformancePhaseV1::run_total)]
    );
    try {
        {
            nb::gil_scoped_release release;
            execute();
        }
        nb::dict result;
        {
            stage2_detail::PerformancePhaseScopeV1 publication_scope(
                performance_phases_[static_cast<std::size_t>(
                    stage2_detail::PerformancePhaseV1::publication)]
            );
            result = publish_compact();
        }
        total_performance_scope.finish();
        performance_complete_ = true;
        return result;
    } catch (const std::exception& error) {
        failure_message_ = error.what();
        failure_type_ = "std_exception";
        failure_phase_ = phase_;
        state_ = "failed";
        try {
            record_event(stage2_detail::EventClassV1::terminal_failure,
                         last_coordinate_, "terminal_exception", 0);
        } catch (...) {
            // Admission reserves the terminal slot. Never replace the
            // scientific failure if internal telemetry itself is damaged.
        }
        phase_ = "failed";
        {
            std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex_);
            checkpoint_released_ = true;
        }
        checkpoint_condition_.notify_all();
        throw;
    } catch (...) {
        failure_message_ = "unknown native exception";
        failure_type_ = "unknown_exception";
        failure_phase_ = phase_;
        state_ = "failed";
        try {
            record_event(stage2_detail::EventClassV1::terminal_failure,
                         last_coordinate_, "terminal_exception", 0);
        } catch (...) {
        }
        phase_ = "failed";
        {
            std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex_);
            checkpoint_released_ = true;
        }
        checkpoint_condition_.notify_all();
        throw;
    }
}

nb::dict ContextualReferenceExecutorV1::performance_report() const {
    using namespace stage2_detail;
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (!performance_complete_ ||
        (state_ != "stage2_incomplete_ready" &&
         state_ != "reference_statistics_complete_ready" &&
         state_ != "trait_statistics_complete_ready"))
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 performance_report requires successful run"
        );
    check_files_unchanged();
    check_hardened_file_content_unchanged();
    check_inputs_unchanged();
    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 scientific output mutation detected before performance report"
        );

    nb::dict report;
    report["schema"] = "contextual_native_performance_report_v1";
    report["schema_version"] = 1;
    report["metadata_only"] = true;
    report["contextual_native_api_version"] = 1;
    report["contextual_backend_version"] = kContextualBackendVersionV1;
    report["contextual_backend"] = trait_mode_
        ? kTraitBackendV1 : kBackendV1;
    report["contextual_execution_backend"] = kExecutionBackendV1;
    report["contextual_build_id"] = GWLDCORE_SOURCE_COMMIT;
    report["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
    report["execution_plan_sha256"] = execution_plan_sha256_;
    report["lifecycle"] = state_;
    report["execution_mode"] = execution_mode_;

    nb::dict units;
    units["time"] = "nanoseconds";
    units["memory"] = "bytes";
    units["work"] = "logical_items";
    units["flop_count"] = "floating_point_operations";
    units["throughput"] = "gflop_per_second";
    units["count"] = "integer";
    report["units"] = units;

    nb::dict dimensions;
    dimensions["samples"] = n_;
    dimensions["variants"] = m_;
    dimensions["fixed_effect_rank"] = rank_;
    dimensions["contexts"] = q_;
    dimensions["annotations"] = k_;
    dimensions["context_pairs"] = pairs_;
    dimensions["components"] = components_;
    dimensions["groups"] = groups_;
    dimensions["sample_probes"] = bt_;
    dimensions["variant_probes"] = bd_;
    dimensions["traits"] = l_;
    dimensions["residual_components"] = h_;
    report["dimensions"] = dimensions;

    nb::dict selected_plan;
    selected_plan["variant_block"] = variant_block_;
    selected_plan["sample_probe_resident"] = sample_probe_resident_;
    selected_plan["sample_probe_tile"] = sample_probe_tile_;
    selected_plan["action_tile"] = action_tile_;
    selected_plan["annotation_tile"] = annotation_tile_;
    selected_plan["context_tile"] = context_tile_;
    selected_plan["variant_probe_tile"] = variant_probe_tile_;
    selected_plan["group_tile"] = group_tile_;
    selected_plan["trait_feature_tile"] = trait_feature_tile_;
    selected_plan["grouped_algorithm"] = grouped_algorithm_;
    selected_plan["direct_grouped_scaling"] = direct_grouped_scaling_;
    selected_plan["strict_disjoint_indexed_routing"] =
        !strict_annotation_ids_.empty();
    report["selected_plan"] = selected_plan;

    nb::dict runtime_policy;
    runtime_policy["decode_threads"] = decode_threads_;
    runtime_policy["blas_threads"] = blas_threads_;
    runtime_policy["numa_policy"] = numa_policy_;
    runtime_policy["output_numa_node"] = output_numa_node_;
    runtime_policy["protected_dispatch_serialized"] = true;
    runtime_policy["deterministic_non_vendor_backend"] = true;
    report["runtime_policy"] = runtime_policy;

    nb::dict admission_bytes;
    admission_bytes["permanent"] = estimate_.permanent_bytes;
    admission_bytes["source_phase"] = estimate_.source_phase_bytes;
    admission_bytes["action_phase"] = estimate_.action_phase_bytes;
    admission_bytes["group_phase"] = estimate_.group_phase_bytes;
    admission_bytes["same_person_phase"] =
        estimate_.same_person_phase_bytes;
    admission_bytes["residual_phase"] = estimate_.residual_phase_bytes;
    admission_bytes["trait_phase"] = estimate_.trait_phase_bytes;
    admission_bytes["compact_output"] = estimate_.compact_output_bytes;
    admission_bytes["integrity_reserve"] = estimate_.integrity_reserve_bytes;
    admission_bytes["telemetry"] = estimate_.telemetry_bytes;
    admission_bytes["required_workspace"] =
        estimate_.required_workspace_bytes;
    admission_bytes["workspace_cap"] = workspace_cap_bytes_;
    report["admission_bytes"] = admission_bytes;

    nb::list phase_order;
    nb::dict phases;
    std::uint64_t active_phase_wall_ns = 0;
    std::uint64_t active_phase_process_cpu_ns = 0;
    for (std::size_t index = 0; index < kPerformancePhaseNamesV1.size();
         ++index) {
        const char* name = kPerformancePhaseNamesV1[index];
        phase_order.append(name);
        const PerformancePhaseRecordV1& phase = performance_phases_[index];
        nb::dict item;
        item["calls"] = phase.calls;
        item["wall_ns"] = phase.wall_ns;
        item["process_cpu_ns"] = phase.process_cpu_ns;
        item["rss_begin_bytes"] = phase.rss_begin_bytes;
        item["rss_end_bytes"] = phase.rss_end_bytes;
        item["rss_high_water_begin_bytes"] =
            phase.rss_high_water_begin_bytes;
        item["rss_high_water_end_bytes"] =
            phase.rss_high_water_end_bytes;
        item["rss_high_water_delta_bytes"] = nonnegative_delta_u64(
            phase.rss_high_water_end_bytes,
            phase.rss_high_water_begin_bytes
        );
        item["process_cpu_available"] = phase.process_cpu_available;
        item["proc_status_available"] = phase.proc_status_available;
        item["rusage_available"] = phase.rusage_available;
        item["proc_io_available"] = phase.proc_io_available;
        item["minor_page_faults"] = phase.minor_page_faults;
        item["major_page_faults"] = phase.major_page_faults;
        item["input_block_operations"] = phase.input_block_operations;
        item["output_block_operations"] = phase.output_block_operations;
        item["io_read_characters"] = phase.io_read_characters;
        item["io_write_characters"] = phase.io_write_characters;
        item["io_read_syscalls"] = phase.io_read_syscalls;
        item["io_write_syscalls"] = phase.io_write_syscalls;
        item["io_read_bytes"] = phase.io_read_bytes;
        item["io_write_bytes"] = phase.io_write_bytes;
        phases[name] = item;
        if (index != static_cast<std::size_t>(PerformancePhaseV1::run_total)) {
            active_phase_wall_ns = saturating_add_u64(
                active_phase_wall_ns, phase.wall_ns
            );
            active_phase_process_cpu_ns = saturating_add_u64(
                active_phase_process_cpu_ns, phase.process_cpu_ns
            );
        }
    }
    report["phase_order"] = phase_order;
    report["phases"] = phases;

    nb::list category_order;
    nb::dict categories;
    std::uint64_t category_wall_ns = 0;
    std::uint64_t category_process_cpu_ns = 0;
    for (std::size_t index = 0; index < kPerformanceCategoryNamesV1.size();
         ++index) {
        const char* name = kPerformanceCategoryNamesV1[index];
        category_order.append(name);
        const PerformanceCategoryRecordV1& category =
            performance_categories_[index];
        nb::dict item;
        item["calls"] = category.calls;
        item["wall_ns"] = category.wall_ns;
        item["process_cpu_ns"] = category.process_cpu_ns;
        item["work_items"] = category.work_items;
        item["logical_bytes"] = category.logical_bytes;
        categories[name] = item;
        category_wall_ns = saturating_add_u64(
            category_wall_ns, category.wall_ns
        );
        category_process_cpu_ns = saturating_add_u64(
            category_process_cpu_ns, category.process_cpu_ns
        );
    }
    report["category_order"] = category_order;
    report["categories"] = categories;

    using ProtectedShapeKeyV1 = std::tuple<
        bool, int, int, int, int, int, int
    >;
    std::array<
        std::map<ProtectedShapeKeyV1, std::uint64_t>,
        kContextualDenseSemanticCountV1
    > shape_histograms;
    for (const EventV1& event : events_) {
        const std::size_t operation = static_cast<std::size_t>(
            event.coordinate.operation
        );
        if (event.event_class != EventClassV1::protected_call ||
            operation >= kContextualDenseSemanticCountV1)
            continue;
        const ProtectedShapeKeyV1 key = std::make_tuple(
            event.transpose_left, event.rows, event.columns, event.reduction,
            event.left_stride, event.right_stride, event.output_stride
        );
        shape_histograms[operation][key] = saturating_add_u64(
            shape_histograms[operation][key], 1
        );
    }

    nb::list operation_order;
    nb::dict operations;
    std::uint64_t protected_calls = 0;
    double protected_flop_count = 0.0;
    std::uint64_t primary_wall_ns = 0;
    std::uint64_t witness_wall_ns = 0;
    std::uint64_t recovery_wall_ns = 0;
    std::uint64_t protected_total_wall_ns = 0;
    std::uint64_t integrity_noncompute_wall_ns = 0;
    for (std::size_t index = 0; index < kContextualDenseSemanticCountV1;
         ++index) {
        const char* name = kContextualDenseSemanticNamesV1[index];
        operation_order.append(name);
        const PerformanceOperationRecordV1& operation =
            performance_operations_[index];
        const std::uint64_t measured_compute_ns = saturating_add_u64(
            saturating_add_u64(
                operation.primary_wall_ns, operation.witness_wall_ns
            ), operation.recovery_wall_ns
        );
        const std::uint64_t noncompute_ns = nonnegative_delta_u64(
            operation.protected_total_wall_ns, measured_compute_ns
        );
        const std::uint64_t integrity_overhead_ns = saturating_add_u64(
            saturating_add_u64(
                operation.witness_wall_ns, operation.recovery_wall_ns
            ), noncompute_ns
        );
        nb::dict item;
        item["calls"] = operation.calls;
        item["logical_flop_count"] = operation.logical_flop_count;
        item["output_elements"] = operation.output_elements;
        item["transpose_left"] = operation.transpose_left;
        item["minimum_rows"] = operation.minimum_rows;
        item["maximum_rows"] = operation.maximum_rows;
        item["minimum_columns"] = operation.minimum_columns;
        item["maximum_columns"] = operation.maximum_columns;
        item["minimum_reduction"] = operation.minimum_reduction;
        item["maximum_reduction"] = operation.maximum_reduction;
        nb::list shape_histogram;
        for (const auto& shape : shape_histograms[index]) {
            bool transpose_left = false;
            int rows = 0, columns = 0, reduction = 0;
            int left_stride = 0, right_stride = 0, output_stride = 0;
            std::tie(
                transpose_left, rows, columns, reduction,
                left_stride, right_stride, output_stride
            ) = shape.first;
            nb::dict shape_item;
            shape_item["transpose_left"] = transpose_left;
            shape_item["rows"] = rows;
            shape_item["columns"] = columns;
            shape_item["reduction"] = reduction;
            shape_item["left_stride"] = left_stride;
            shape_item["right_stride"] = right_stride;
            shape_item["output_stride"] = output_stride;
            shape_item["calls"] = shape.second;
            shape_histogram.append(shape_item);
        }
        item["shape_histogram"] = shape_histogram;
        item["primary_wall_ns"] = operation.primary_wall_ns;
        item["witness_wall_ns"] = operation.witness_wall_ns;
        item["retry_calls"] = operation.retry_calls;
        item["fallback_calls"] = operation.fallback_calls;
        item["recovery_wall_ns"] = operation.recovery_wall_ns;
        item["protected_total_wall_ns"] =
            operation.protected_total_wall_ns;
        item["integrity_noncompute_wall_ns"] = noncompute_ns;
        item["integrity_overhead_wall_ns"] = integrity_overhead_ns;
        item["primary_effective_gflops"] = operation.primary_wall_ns == 0
            ? 0.0 : operation.logical_flop_count /
                static_cast<double>(operation.primary_wall_ns);
        item["integrity_overhead_fraction"] =
            operation.protected_total_wall_ns == 0 ? 0.0 :
            static_cast<double>(integrity_overhead_ns) /
                static_cast<double>(operation.protected_total_wall_ns);
        operations[name] = item;
        protected_calls = saturating_add_u64(
            protected_calls, operation.calls
        );
        protected_flop_count += operation.logical_flop_count;
        primary_wall_ns = saturating_add_u64(
            primary_wall_ns, operation.primary_wall_ns
        );
        witness_wall_ns = saturating_add_u64(
            witness_wall_ns, operation.witness_wall_ns
        );
        recovery_wall_ns = saturating_add_u64(
            recovery_wall_ns, operation.recovery_wall_ns
        );
        protected_total_wall_ns = saturating_add_u64(
            protected_total_wall_ns, operation.protected_total_wall_ns
        );
        integrity_noncompute_wall_ns = saturating_add_u64(
            integrity_noncompute_wall_ns, noncompute_ns
        );
    }
    report["operation_order"] = operation_order;
    report["operations"] = operations;

    const std::uint64_t expected_descriptor_passes =
        estimate_.source_descriptor_passes +
        estimate_.action_descriptor_passes +
        estimate_.group_descriptor_passes +
        estimate_.same_person_descriptor_passes +
        estimate_.trait_descriptor_passes;
    const std::uint64_t expected_decoded_blocks =
        estimate_.source_decoded_blocks +
        estimate_.action_decoded_blocks +
        estimate_.group_decoded_blocks +
        estimate_.same_person_decoded_blocks +
        estimate_.trait_decoded_blocks;
    const std::uint64_t expected_variant_visits = expected_descriptor_passes * m_;
    const std::uint64_t expected_group_execution_visits =
        stage3_enabled_ &&
            (grouped_algorithm_ == "group_restricted_action_v1" ||
             enable_grouped_differential_)
        ? m_ : 0;
    const std::uint64_t observed_duplicate_decodes =
        observed_variant_visits_ >= m_ ? observed_variant_visits_ - m_ : 0;
    nb::dict accounting;
    accounting["expected_descriptor_passes"] = expected_descriptor_passes;
    accounting["observed_descriptor_passes"] = observed_descriptor_passes_;
    accounting["expected_decoded_blocks"] = expected_decoded_blocks;
    accounting["observed_decoded_blocks"] = observed_decoded_blocks_;
    accounting["expected_variant_record_visits"] = expected_variant_visits;
    accounting["observed_variant_record_visits"] = observed_variant_visits_;
    accounting["unique_variant_records"] = m_;
    accounting["duplicate_decodes"] = observed_duplicate_decodes;
    accounting["bed_record_bytes_per_variant"] = bytes_per_snp_;
    accounting["expected_physical_record_bytes_visited"] =
        expected_variant_visits * bytes_per_snp_;
    accounting["observed_physical_record_bytes_visited"] =
        observed_variant_visits_ * bytes_per_snp_;
    accounting["expected_group_execution_variant_visits"] =
        expected_group_execution_visits;
    accounting["observed_group_execution_variant_visits"] =
        observed_group_execution_visits_;
    accounting["expected_protected_calls"] = estimate_.total_protected_calls;
    accounting["observed_protected_calls"] = protected_calls;
    accounting["descriptor_accounting_exact"] =
        expected_descriptor_passes == observed_descriptor_passes_ &&
        expected_decoded_blocks == observed_decoded_blocks_ &&
        expected_variant_visits == observed_variant_visits_;
    accounting["protected_call_accounting_exact"] =
        estimate_.total_protected_calls == protected_calls;
    report["accounting"] = accounting;

    const PerformancePhaseRecordV1& run_total = performance_phases_[
        static_cast<std::size_t>(PerformancePhaseV1::run_total)
    ];
    const std::uint64_t integrity_overhead_wall_ns = saturating_add_u64(
        saturating_add_u64(witness_wall_ns, recovery_wall_ns),
        integrity_noncompute_wall_ns
    );
    nb::dict totals;
    totals["run_wall_ns"] = run_total.wall_ns;
    totals["run_process_cpu_ns"] = run_total.process_cpu_ns;
    totals["active_phase_wall_ns"] = active_phase_wall_ns;
    totals["active_phase_process_cpu_ns"] = active_phase_process_cpu_ns;
    totals["category_wall_ns"] = category_wall_ns;
    totals["category_process_cpu_ns"] = category_process_cpu_ns;
    totals["protected_calls"] = protected_calls;
    totals["protected_logical_flop_count"] = protected_flop_count;
    totals["primary_wall_ns"] = primary_wall_ns;
    totals["witness_wall_ns"] = witness_wall_ns;
    totals["recovery_wall_ns"] = recovery_wall_ns;
    totals["protected_total_wall_ns"] = protected_total_wall_ns;
    totals["integrity_noncompute_wall_ns"] = integrity_noncompute_wall_ns;
    totals["integrity_overhead_wall_ns"] = integrity_overhead_wall_ns;
    totals["integrity_overhead_fraction"] =
        protected_total_wall_ns == 0 ? 0.0 :
        static_cast<double>(integrity_overhead_wall_ns) /
            static_cast<double>(protected_total_wall_ns);
    totals["process_peak_rss_bytes"] =
        run_total.rss_high_water_end_bytes;
    report["totals"] = totals;

    nb::dict resource_capabilities;
    resource_capabilities["process_cpu_clock"] =
        run_total.process_cpu_available;
    resource_capabilities["linux_proc_status"] =
        run_total.proc_status_available;
    resource_capabilities["posix_getrusage"] =
        run_total.rusage_available;
    resource_capabilities["linux_proc_io"] = run_total.proc_io_available;
    report["resource_capabilities"] = resource_capabilities;

    nb::dict invariants;
    invariants["phase_wall_within_run"] =
        active_phase_wall_ns <= run_total.wall_ns;
    invariants["phase_process_cpu_within_run"] =
        active_phase_process_cpu_ns <= run_total.process_cpu_ns;
    invariants["category_wall_within_active_phases"] =
        category_wall_ns <= active_phase_wall_ns;
    invariants["category_process_cpu_within_active_phases"] =
        category_process_cpu_ns <= active_phase_process_cpu_ns;
    invariants["descriptor_accounting_exact"] =
        expected_descriptor_passes == observed_descriptor_passes_ &&
        expected_decoded_blocks == observed_decoded_blocks_ &&
        expected_variant_visits == observed_variant_visits_;
    invariants["protected_calls_match_semantic_ledger"] =
        estimate_.total_protected_calls == protected_calls &&
        performance_operations_.size() == observed_calls_.size() &&
        std::equal(
            performance_operations_.begin(), performance_operations_.end(),
            observed_calls_.begin(),
            [](const PerformanceOperationRecordV1& performance,
               std::uint64_t observed) {
                return performance.calls == observed;
            }
        );
    invariants["scientific_state_unchanged_verified"] = true;
    invariants["report_contains_scientific_ndarray"] = false;
    invariants["instrumentation_changes_execution_plan"] = false;
    report["invariants"] = invariants;
    report["categories_are_nonoverlapping"] = true;
    report["categories_cover_entire_run"] = false;

    if (fingerprint_outputs() != output_fingerprint_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 scientific output mutation detected during performance report"
        );
    return report;
}

nb::dict ContextualReferenceExecutorV1::failure_report() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (state_ != "failed")
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 failure_report requires failed lifecycle"
        );
    nb::dict report;
    report["metadata_only"] = true;
    report["contextual_native_api_version"] = 1;
    report["contextual_backend_version"] =
        stage2_detail::kContextualBackendVersionV1;
    report["contextual_backend"] = trait_mode_
        ? stage2_detail::kTraitBackendV1 : stage2_detail::kBackendV1;
    report["contextual_execution_backend"] =
        stage2_detail::kExecutionBackendV1;
    report["state"] = state_;
    report["lifecycle"] = state_;
    report["failure_type"] = failure_type_;
    report["failure_message"] = failure_message_;
    report["last_phase"] = failure_phase_;
    report["last_semantic_anchor"] =
        stage2_detail::semantic_anchor(last_coordinate_);
    report["capacity"] = telemetry_capacity_;
    report["required_capacity"] = estimate_.required_telemetry_capacity;
    report["observed_events"] = events_.size();
    using stage2_detail::EventClassV1;
    std::array<std::uint64_t, kContextualDenseSemanticCountV1>
        protected_by_operation{};
    std::uint64_t terminal_count = 0;
    std::uint64_t injection_events = 0;
    std::uint64_t detection_events = 0;
    std::uint64_t retry_events = 0;
    std::uint64_t repair_events = 0;
    std::uint64_t fallback_events = 0;
    bool contiguous = event_sequence_ == events_.size();
    bool ownership_complete = true;
    bool evidence_complete = true;
    bool resolution_complete = true;
    for (std::size_t index = 0; index < events_.size(); ++index) {
        const auto& event = events_[index];
        contiguous = contiguous && event.sequence == index + 1U;
        ownership_complete = ownership_complete &&
            event.process_id > 0 && event.thread_id > 0;
        const std::size_t operation = static_cast<std::size_t>(
            event.coordinate.operation
        );
        const bool protected_coordinate =
            operation < kContextualDenseSemanticCountV1;
        const bool verified_attempt = protected_coordinate && (
            event.event_class == EventClassV1::protected_call ||
            event.event_class == EventClassV1::retry ||
            event.event_class == EventClassV1::repair ||
            event.event_class == EventClassV1::trusted_fallback ||
            event.event_class == EventClassV1::semantic_verification);
        if (verified_attempt)
            evidence_complete = evidence_complete &&
                event.rows > 0 && event.columns > 0 && event.reduction > 0 &&
                event.left_stride > 0 && event.right_stride > 0 &&
                event.output_stride >= event.rows &&
                event.operand_fingerprint != 0 &&
                event.witness_fingerprint != 0 &&
                event.accepted_output_fingerprint != 0 &&
                event.runtime_before_fingerprint != 0 &&
                event.runtime_after_fingerprint != 0 &&
                event.prefix_canary_verified &&
                event.suffix_canary_verified &&
                event.finiteness_verified &&
                event.serialized_entry_verified &&
                event.deterministic_non_vendor_backend;
        const bool diagnostic_attempt = protected_coordinate && (
            event.event_class == EventClassV1::fault_injection ||
            event.event_class == EventClassV1::fault_detection);
        if (diagnostic_attempt)
            evidence_complete = evidence_complete &&
                event.rows > 0 && event.columns > 0 && event.reduction > 0 &&
                event.left_stride > 0 && event.right_stride > 0 &&
                event.output_stride >= event.rows &&
                event.operand_fingerprint != 0 &&
                event.witness_fingerprint != 0 &&
                event.accepted_output_fingerprint != 0 &&
                event.runtime_before_fingerprint != 0 &&
                event.runtime_after_fingerprint != 0 &&
                event.serialized_entry_verified &&
                event.deterministic_non_vendor_backend;
        if (event.event_class == EventClassV1::protected_call)
            ++protected_by_operation[operation];
        else if (event.event_class == EventClassV1::fault_injection)
            ++injection_events;
        else if (event.event_class == EventClassV1::fault_detection)
            ++detection_events;
        else if (event.event_class == EventClassV1::retry)
            ++retry_events;
        else if (event.event_class == EventClassV1::repair)
            ++repair_events;
        else if (event.event_class == EventClassV1::trusted_fallback)
            ++fallback_events;
        else if (event.event_class == EventClassV1::terminal_failure)
            ++terminal_count;
        if (event.event_class == EventClassV1::fault_detection)
            resolution_complete = resolution_complete &&
                event.resolution != nullptr && event.resolution[0] != '\0';
    }
    const bool terminal_last = !events_.empty() &&
        events_.back().event_class == EventClassV1::terminal_failure;
    const bool terminal_unique = terminal_count == 1U;
    const bool capacity_complete = events_.size() <= telemetry_capacity_ &&
        events_.size() <= estimate_.required_telemetry_capacity;
    const bool counts_consistent =
        protected_by_operation == observed_calls_ &&
        injection_events == injection_count_ &&
        detection_events >= injection_events &&
        retry_events == retry_count_ && repair_events == repair_count_ &&
        fallback_events == fallback_count_;
    report["event_sequence_contiguous"] = contiguous;
    report["event_ownership_complete"] = ownership_complete;
    report["event_evidence_complete"] = evidence_complete;
    report["event_resolution_complete"] = resolution_complete;
    report["event_counts_consistent"] = counts_consistent;
    report["event_capacity_complete"] = capacity_complete;
    report["terminal_event_count"] = terminal_count;
    report["terminal_event_unique"] = terminal_unique;
    report["terminal_event_last"] = terminal_last;
    report["fault_injection_event_count"] = injection_events;
    report["fault_detection_count"] = detection_events;
    report["event_ledger_complete_without_drop"] = contiguous &&
        ownership_complete && evidence_complete && resolution_complete &&
        counts_consistent && capacity_complete && terminal_unique &&
        terminal_last;
    report["terminal_event_recorded"] = terminal_unique && terminal_last;
    report["injection_count"] = injection_count_;
    report["repair_count"] = repair_count_;
    report["retry_count"] = retry_count_;
    report["fallback_count"] = fallback_count_;
    nb::list event_list;
    for (const auto& event : events_)
        event_list.append(stage2_detail::event_dict(event));
    report["events"] = event_list;
    return report;
}

nb::dict ContextualReferenceExecutorV1::differential_snapshot() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (!enable_differential_snapshot_)
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 differential snapshot is disabled"
        );
    if (state_ != "stage2_incomplete_ready" &&
        state_ != "reference_statistics_complete_ready")
        throw std::runtime_error(
            "ContextualReferenceExecutorV1 differential snapshot requires a complete successful run"
        );
    std::vector<double> source(m_ * q_ * bt_);
    for (std::size_t variant = 0; variant < m_; ++variant)
        for (std::size_t context = 0; context < q_; ++context)
            for (std::size_t probe = 0; probe < bt_; ++probe)
                source[(variant * q_ + context) * bt_ + probe] =
                    source_scores_[source_index(variant, context, probe)];
    std::vector<double> raw(n_ * components_ * bt_);
    std::vector<double> normalized(n_ * components_ * bt_);
    for (std::size_t row = 0; row < n_; ++row) {
        for (std::size_t component = 0; component < components_; ++component) {
            const std::size_t annotation = static_cast<std::size_t>(
                component_annotation_[component]
            );
            for (std::size_t probe = 0; probe < bt_; ++probe) {
                const std::size_t destination =
                    (row * components_ + component) * bt_ + probe;
                const double value = raw_actions_[action_index(
                    row, component, probe
                )];
                raw[destination] = value;
                normalized[destination] = value / annotation_masses_[annotation];
            }
        }
    }
    nb::dict result;
    result["nonproduction_differential_snapshot"] = true;
    result["source_scores"] = stage2_readonly_owned_arr3(
        source, m_, q_, bt_
    );
    result["raw_actions"] = stage2_readonly_owned_arr3(
        raw, n_, components_, bt_
    );
    result["normalized_actions"] = stage2_readonly_owned_arr3(
        normalized, n_, components_, bt_
    );
    if (stage3_enabled_ && enable_grouped_differential_) {
        result["group_gram_numerator_restricted"] =
            stage2_readonly_owned_arr3(
                group_restricted_differential_, groups_, components_,
                components_
            );
        result["group_gram_numerator_direct_action_scaled"] =
            stage2_readonly_owned_arr3(
                group_direct_action_, groups_, components_, components_
            );
        result["group_gram_numerator_direct_genotype_scaled"] =
            stage2_readonly_owned_arr3(
                group_direct_genotype_, groups_, components_, components_
            );
    }
    if (stage3_enabled_) {
        result["same_probe_sums"] = stage2_readonly_owned_mat(
            differential_same_probe_sums_, n_, components_
        );
        result["same_probe_cross"] = stage2_readonly_owned_mat(
            differential_same_probe_cross_, components_, components_
        );
        result["same_probe_sums_layout"] = "sample_component_c_v1";
        result["same_probe_cross_layout"] = "component_component_c_v1";
        result["same_probe_sums_sha256"] =
            stage2_detail::array_sha256_f64_c_order(
                differential_same_probe_sums_, {n_, components_}
            );
        result["same_probe_cross_sha256"] =
            stage2_detail::array_sha256_f64_c_order(
                differential_same_probe_cross_, {components_, components_}
            );
        result["same_probe_sample_count"] = n_;
        result["same_probe_component_count"] = components_;
        result["same_probe_variant_probe_count"] = bd_;
        result["same_probe_variant_probe_identity_sha256"] =
            variant_probe_identity_sha256_;
        result["same_probe_retained_sample_map_sha256"] =
            retained_sample_map_sha256_;
        result["same_probe_component_annotation_sha256"] =
            stage2_detail::array_sha256_i64_c_order(
                component_annotation_, {components_}
            );
        result["same_probe_component_pair_sha256"] =
            stage2_detail::array_sha256_i64_c_order(
                component_pair_, {components_}
            );

        std::vector<std::string> probe_leaves;
        probe_leaves.reserve(bd_);
        if (!explicit_variant_probes_.empty()) {
            for (std::size_t probe = 0; probe < bd_; ++probe)
                probe_leaves.push_back(
                    stage2_detail::array_sha256_f64_span(
                        explicit_variant_probes_.data() + m_ * probe, m_
                    )
                );
        } else {
            for (std::size_t probe = 0; probe < bd_; ++probe)
                probe_leaves.push_back(
                    stage2_detail::array_sha256_u64_span(
                        variant_philox_keys_.data() + 2U * probe, 2U
                    )
                );
        }
        if (probe_leaves.size() != bd_)
            throw std::runtime_error(
                "ContextualReferenceExecutorV1 variant-probe leaf ledger mismatch"
            );

        nb::dict science_identity_inputs;
        science_identity_inputs["annotation_map_sha256"] =
            annotation_map_sha256_;
        science_identity_inputs["evaluated_phi_sha256"] =
            evaluated_phi_sha256_;
        science_identity_inputs["fixed_basis_sha256"] = fixed_basis_sha256_;
        science_identity_inputs["group_map_sha256"] = group_map_sha256_;
        science_identity_inputs["missingness_sha256"] =
            observed_missingness_sha256_;
        science_identity_inputs["retained_sample_map_sha256"] =
            retained_sample_map_sha256_;
        science_identity_inputs["retained_variant_order_sha256"] =
            observed_retained_order_sha256_;
        science_identity_inputs["scale_plan_sha256"] =
            observed_scale_plan_sha256_;
        science_identity_inputs["variant_order_allele_sha256"] =
            variant_order_allele_sha256_;
        science_identity_inputs["annotation_names"] = annotation_names_;
        science_identity_inputs["group_names"] = group_names_;
        science_identity_inputs["pair_q"] =
            stage2_readonly_owned_vec(pair_q_);
        science_identity_inputs["pair_r"] =
            stage2_readonly_owned_vec(pair_r_);
        science_identity_inputs["pair_eta"] =
            stage2_readonly_owned_vec(pair_eta_);
        science_identity_inputs["component_annotation"] =
            stage2_readonly_owned_vec(component_annotation_);
        science_identity_inputs["component_pair"] =
            stage2_readonly_owned_vec(component_pair_);

        nb::dict native_identity;
        native_identity["schema"] =
            "contextual_native_variant_probe_full_range_identity_v1";
        native_identity["source_commit"] = GWLDCORE_SOURCE_COMMIT;
        native_identity["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
        native_identity["build_id"] = GWLDCORE_SOURCE_COMMIT;
        native_identity["build_provenance"] =
            stage2_detail::build_provenance_dict();
        native_identity["native_api_version"] = 1;
        native_identity["native_backend_version"] =
            stage2_detail::kContextualBackendVersionV1;
        native_identity["native_backend"] = stage2_detail::kBackendV1;
        native_identity["native_execution_backend"] =
            stage2_detail::kExecutionBackendV1;
        native_identity["numeric_policy"] = "fp64_v1";
        native_identity["feature_mode"] = "P_diag_phi_G_v1";
        native_identity["execution_plan_sha256"] = execution_plan_sha256_;
        native_identity["sealed_plan_sha256"] = sealed_plan_sha256_;
        native_identity["variant_probe_policy"] = variant_probe_policy_;
        native_identity["native_variant_probe_identity_sha256"] =
            variant_probe_identity_sha256_;
        native_identity["global_probe_begin"] = 0;
        native_identity["global_probe_end"] = bd_;
        native_identity["global_probe_count"] = bd_;
        native_identity["ordered_probe_sha256s"] = probe_leaves;
        native_identity["global_probe_plan_sha256"] =
            stage2_detail::variant_probe_plan_sha256(probe_leaves);
        native_identity["sample_count"] = n_;
        native_identity["component_count"] = components_;
        native_identity["probe_sums_layout"] =
            "sample_component_c_v1";
        native_identity["probe_cross_layout"] =
            "component_component_c_v1";
        native_identity["full_executor_range_only"] = true;
        native_identity["partitioned_native_finalizer_claimed"] = false;
        native_identity["science_identity_inputs"] = science_identity_inputs;
        result["same_probe_native_identity"] = native_identity;
    }
    return result;
}

nb::list ContextualReferenceExecutorV1::semantic_anchors() const {
    nb::list result;
    const std::size_t begin = trait_mode_ ? 14U : 0U;
    const std::size_t count = trait_mode_ ? 17U :
        (stage3_enabled_ ? 14U : 7U);
    for (std::size_t index = begin; index < count; ++index) {
        if (estimate_.calls[index] == 0) continue;
        const auto operation = static_cast<ContextualDenseSemanticV1>(index);
        std::uint64_t variant = 0;
        std::uint64_t group = 0;
        if (operation == ContextualDenseSemanticV1::group_target_nn ||
            operation == ContextualDenseSemanticV1::group_cross_gram_tn) {
            variant = static_cast<std::uint64_t>(group_execution_order_[0]);
            group = static_cast<std::uint64_t>(group_index_[variant]);
        } else if (operation == ContextualDenseSemanticV1::direct_grouped_tn) {
            group = static_cast<std::uint64_t>(group_index_[0]);
        }
        std::uint64_t annotation = 0;
        if (!strict_annotation_ids_.empty()) {
            if (operation == ContextualDenseSemanticV1::full_target_nn)
                annotation = static_cast<std::uint64_t>(
                    strict_annotation_ids_[0]
                );
            else if (operation == ContextualDenseSemanticV1::group_target_nn)
                annotation = static_cast<std::uint64_t>(
                    strict_annotation_ids_[variant]
                );
        }
        const auto append = [&](stage2_detail::SemanticPhaseV1 phase,
                                stage2_detail::SemanticRoleV1 role,
                                stage2_detail::SemanticPlacementV1 placement,
                                bool include_identity) {
            nb::dict item;
            item["operation"] = kContextualDenseSemanticNamesV1[index];
            item["semantic_phase"] = stage2_detail::semantic_phase_name(phase);
            item["semantic_role"] = stage2_detail::semantic_role_name(role);
            item["semantic_placement"] =
                stage2_detail::semantic_placement_name(placement);
            item["canonical_coordinate"] = variant;
            item["semantic_anchor"] = stage2_detail::semantic_point_anchor(
                operation, 0, 0, variant, annotation, 0, 0, group,
                phase, role, placement, 0, include_identity
            );
            result.append(item);
        };
        const auto phase = stage2_detail::semantic_phase_for_operation(operation);
        if (operation == ContextualDenseSemanticV1::same_person_gram_tn) {
            append(phase, stage2_detail::SemanticRoleV1::tile,
                   stage2_detail::SemanticPlacementV1::none, true);
            append(phase, stage2_detail::SemanticRoleV1::global_merge,
                   stage2_detail::SemanticPlacementV1::none, true);
        } else if (operation == ContextualDenseSemanticV1::direct_grouped_tn) {
            if (enable_grouped_differential_ ||
                direct_grouped_scaling_ != "genotype_scaled_v1")
                append(phase, stage2_detail::SemanticRoleV1::ordinary,
                       stage2_detail::SemanticPlacementV1::action_scaled, true);
            if (enable_grouped_differential_ ||
                direct_grouped_scaling_ == "genotype_scaled_v1")
                append(phase, stage2_detail::SemanticRoleV1::ordinary,
                       stage2_detail::SemanticPlacementV1::genotype_scaled, true);
        } else {
            append(phase, stage2_detail::SemanticRoleV1::ordinary,
                   stage2_detail::SemanticPlacementV1::none, false);
        }
    }
    return result;
}

void bind_contextual_reference_executor_v1(nb::module_& module) {
    module.def("contextual_streamed_semantic_operations_v1", []() {
        std::vector<std::string> operations;
        operations.reserve(7);
        for (std::size_t index = 0; index < 7; ++index)
            operations.emplace_back(kContextualDenseSemanticNamesV1[index]);
        return operations;
    });
    module.def("contextual_complete_reference_semantic_operations_v1", []() {
        std::vector<std::string> operations;
        operations.reserve(14);
        for (std::size_t index = 0; index < 14; ++index)
            operations.emplace_back(kContextualDenseSemanticNamesV1[index]);
        return operations;
    });
    module.def("contextual_trait_semantic_operations_v1", []() {
        std::vector<std::string> operations;
        operations.reserve(3);
        for (std::size_t index = 14; index < 17; ++index)
            operations.emplace_back(kContextualDenseSemanticNamesV1[index]);
        return operations;
    });
    nb::class_<ContextualReferenceExecutorV1>(
        module, "ContextualReferenceExecutorV1"
    )
        .def(
            nb::init<
                int, int, int,
                nb_vec1_ro<std::int64_t>, nb_vec1_ro<std::int64_t>,
                std::vector<std::string>, std::vector<std::string>,
                std::vector<std::string>, nb_vec1_ro<std::uint8_t>,
                nb_vec1_ro<double>, nb_vec1_ro<double>,
                nb_vec1_ro<std::int64_t>, nb_mat2f_ro<double>,
                nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                nb_vec1_ro<std::int64_t>, nb::object, nb::object,
                std::vector<std::string>, std::vector<std::string>,
                std::string, std::string, std::string, std::string,
                std::string, std::string, std::string,
                std::uint64_t, std::uint64_t, std::uint64_t,
                std::uint64_t, std::uint64_t, std::uint64_t,
                std::uint64_t, int, int, std::uint64_t, std::uint64_t,
                bool, std::string, std::string, std::string,
                std::string, std::string, std::string, int,
                nb::object, nb::object, std::uint64_t, std::uint64_t,
                std::uint64_t, std::string, std::string, bool
            >(),
            nb::arg("bed_descriptor"), nb::arg("bim_descriptor"),
            nb::arg("fam_descriptor"),
            nb::arg("retained_sample_indices"),
            nb::arg("retained_variant_indices"),
            nb::arg("expected_variant_ids"), nb::arg("counted_alleles"),
            nb::arg("other_alleles"), nb::arg("counted_allele_is_a1"),
            nb::arg("affine_mean"), nb::arg("affine_inverse_scale"),
            nb::arg("expected_missing_counts"), nb::arg("fixed_basis"),
            nb::arg("context_basis"), nb::arg("annotation_weights"),
            nb::arg("group_index"), nb::arg("sample_probes").none(),
            nb::arg("philox_keys").none(), nb::arg("annotation_names"),
            nb::arg("group_names"), nb::kw_only(),
            nb::arg("annotation_mode") = "generic_nonnegative_weights_v1",
            nb::arg("retained_variant_order_sha256") = "",
            nb::arg("affine_mean_sha256") = "",
            nb::arg("affine_inverse_scale_sha256") = "",
            nb::arg("missingness_sha256") = "",
            nb::arg("scale_plan_sha256") = "",
            nb::arg("centering_source") = "provided_v1",
            nb::arg("sample_probe_count") = 0,
            nb::arg("variant_block") = 0,
            nb::arg("sample_probe_resident") = 0,
            nb::arg("sample_probe_tile") = 0,
            nb::arg("action_tile") = 0,
            nb::arg("annotation_tile") = 0,
            nb::arg("context_tile") = 0,
            nb::arg("decode_threads") = 1,
            nb::arg("blas_threads") = 1,
            nb::arg("workspace_cap_bytes") =
                std::numeric_limits<std::uint64_t>::max(),
            nb::arg("telemetry_capacity") =
                std::numeric_limits<std::uint64_t>::max(),
            nb::arg("enable_differential_snapshot") = false,
            nb::arg("fault_operation") = "",
            nb::arg("fault_mode") = "none",
            nb::arg("fault_semantic_anchor") = "",
            nb::arg("test_checkpoint") = "",
            nb::arg("test_mutation_target") = "none",
            nb::arg("numa_policy") = "unbound_first_touch_v1",
            nb::arg("output_numa_node") = -1,
            nb::arg("variant_probes").none() = nb::none(),
            nb::arg("variant_philox_keys").none() = nb::none(),
            nb::arg("variant_probe_count") = 0,
            nb::arg("variant_probe_tile") = 0,
            nb::arg("group_tile") = 1,
            nb::arg("grouped_algorithm") = "group_restricted_action_v1",
            nb::arg("direct_grouped_scaling") = "action_scaled_v1",
            nb::arg("enable_grouped_differential") = false,
            "Seal a descriptor-owned one-shot Stage 2/3 contextual reference executor."
        )
        .def("info", &ContextualReferenceExecutorV1::info)
        .def("preflight", &ContextualReferenceExecutorV1::preflight)
        .def("run", &ContextualReferenceExecutorV1::run)
        .def("performance_report",
             &ContextualReferenceExecutorV1::performance_report)
        .def("failure_report", &ContextualReferenceExecutorV1::failure_report)
        .def("differential_snapshot",
             &ContextualReferenceExecutorV1::differential_snapshot)
        .def("semantic_anchors", &ContextualReferenceExecutorV1::semantic_anchors)
        .def("wait_for_test_checkpoint",
             &ContextualReferenceExecutorV1::wait_for_test_checkpoint,
             nb::arg("timeout_ms") = 5000)
        .def("release_test_checkpoint",
             &ContextualReferenceExecutorV1::release_test_checkpoint);

    module.def(
        "ContextualTraitExecutorV1",
        [](
            int bed_descriptor, int bim_descriptor, int fam_descriptor,
            nb_vec1_ro<std::int64_t> retained_sample_indices,
            nb_vec1_ro<std::int64_t> retained_variant_indices,
            std::vector<std::string> expected_variant_ids,
            std::vector<std::string> counted_alleles,
            std::vector<std::string> other_alleles,
            nb_vec1_ro<std::uint8_t> counted_allele_is_a1,
            nb_vec1_ro<double> affine_mean,
            nb_vec1_ro<double> affine_inverse_scale,
            nb_vec1_ro<std::int64_t> expected_missing_counts,
            nb_mat2f_ro<double> fixed_basis,
            nb_mat2f_ro<double> context_basis,
            nb_mat2f_ro<double> annotation_weights,
            nb_vec1_ro<std::int64_t> group_index,
            nb::object phenotype_batch, nb::object residual_basis,
            std::vector<std::string> annotation_names,
            std::vector<std::string> group_names,
            std::vector<std::string> trait_names,
            std::vector<std::string> residual_names,
            std::string annotation_mode,
            std::string retained_variant_order_sha256,
            std::string affine_mean_sha256,
            std::string affine_inverse_scale_sha256,
            std::string missingness_sha256,
            std::string scale_plan_sha256,
            std::string centering_source,
            std::uint64_t variant_block,
            std::uint64_t trait_feature_tile,
            int decode_threads, int blas_threads,
            std::uint64_t workspace_cap_bytes,
            std::uint64_t telemetry_capacity,
            std::string phenotype_normalization,
            std::string fault_operation,
            std::string fault_mode,
            std::string fault_semantic_anchor,
            std::string test_checkpoint,
            std::string test_mutation_target,
            std::string numa_policy,
            int output_numa_node
        ) -> ContextualReferenceExecutorV1* {
            return new ContextualReferenceExecutorV1(
                bed_descriptor, bim_descriptor, fam_descriptor,
                retained_sample_indices, retained_variant_indices,
                std::move(expected_variant_ids), std::move(counted_alleles),
                std::move(other_alleles), counted_allele_is_a1,
                affine_mean, affine_inverse_scale, expected_missing_counts,
                fixed_basis, context_basis, annotation_weights, group_index,
                nb::none(), nb::none(), std::move(annotation_names),
                std::move(group_names), std::move(annotation_mode),
                std::move(retained_variant_order_sha256),
                std::move(affine_mean_sha256),
                std::move(affine_inverse_scale_sha256),
                std::move(missingness_sha256),
                std::move(scale_plan_sha256), std::move(centering_source),
                0, variant_block, 0, 0, 0, 0, 0,
                decode_threads, blas_threads, workspace_cap_bytes,
                telemetry_capacity, false, std::move(fault_operation),
                std::move(fault_mode), std::move(fault_semantic_anchor),
                std::move(test_checkpoint), std::move(test_mutation_target),
                std::move(numa_policy), output_numa_node,
                nb::none(), nb::none(), 0, 0, 1,
                "group_restricted_action_v1", "action_scaled_v1", false,
                std::move(phenotype_batch), std::move(residual_basis),
                std::move(trait_names), std::move(residual_names),
                "trait_v1", std::move(phenotype_normalization),
                trait_feature_tile
            );
        },
        nb::rv_policy::take_ownership,
        nb::arg("bed_descriptor"), nb::arg("bim_descriptor"),
        nb::arg("fam_descriptor"),
        nb::arg("retained_sample_indices"),
        nb::arg("retained_variant_indices"),
        nb::arg("expected_variant_ids"), nb::arg("counted_alleles"),
        nb::arg("other_alleles"), nb::arg("counted_allele_is_a1"),
        nb::arg("affine_mean"), nb::arg("affine_inverse_scale"),
        nb::arg("expected_missing_counts"), nb::arg("fixed_basis"),
        nb::arg("context_basis"), nb::arg("annotation_weights"),
        nb::arg("group_index"), nb::arg("phenotype_batch"),
        nb::arg("residual_basis"), nb::arg("annotation_names"),
        nb::arg("group_names"), nb::arg("trait_names"),
        nb::arg("residual_names"), nb::kw_only(),
        nb::arg("annotation_mode") = "generic_nonnegative_weights_v1",
        nb::arg("retained_variant_order_sha256") = "",
        nb::arg("affine_mean_sha256") = "",
        nb::arg("affine_inverse_scale_sha256") = "",
        nb::arg("missingness_sha256") = "",
        nb::arg("scale_plan_sha256") = "",
        nb::arg("centering_source") = "provided_v1",
        nb::arg("variant_block") = 0,
        nb::arg("trait_feature_tile") = 0,
        nb::arg("decode_threads") = 1,
        nb::arg("blas_threads") = 1,
        nb::arg("workspace_cap_bytes") =
            std::numeric_limits<std::uint64_t>::max(),
        nb::arg("telemetry_capacity") =
            std::numeric_limits<std::uint64_t>::max(),
        nb::arg("phenotype_normalization") =
            "project_then_unit_residual_variance_v1",
        nb::arg("fault_operation") = "",
        nb::arg("fault_mode") = "none",
        nb::arg("fault_semantic_anchor") = "",
        nb::arg("test_checkpoint") = "",
        nb::arg("test_mutation_target") = "none",
        nb::arg("numa_policy") = "unbound_first_touch_v1",
        nb::arg("output_numa_node") = -1,
        "Seal a descriptor-owned one-shot Stage 4 contextual trait executor."
    );
}

}  // namespace summit::context_v1
