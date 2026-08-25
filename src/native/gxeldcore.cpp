#include "nb_utils.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/syscall.h>
  #include <sched.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "blas_compat.hpp"
#include "genotype.hpp"
#include "mailman.hpp"

#ifdef GWLDCORE_USE_BLIS
  #include <blis/blis.h>
#if !defined(BLIS_ENABLE_TLS)
  #error "The private upstream BLIS candidate must be built with TLS enabled"
#endif
#if !defined(BLIS_ENABLE_PTHREADS) || !defined(BLIS_ENABLE_PTHREADS_AS_DEFAULT)
  #error "The private upstream BLIS candidate must default to pthreads"
#endif
// Upstream BLIS 2.0 retains the original CBLAS_ORDER spelling while newer
// CBLAS headers expose the same enum as CBLAS_LAYOUT. Keep the implementation
// type-safe without changing the shared compatibility header.
using CBLAS_LAYOUT = CBLAS_ORDER;
#endif

#if defined(GWLDCORE_USE_OPENBLAS) || defined(GWLDCORE_USE_BLIS)
  #define GWLDCORE_USE_FIXED_VENDOR_BLAS
#endif

#ifdef GWLDCORE_USE_OPENBLAS
extern "C" char* openblas_get_config(void);
extern "C" char* openblas_get_corename(void);
extern "C" int openblas_get_num_threads(void);
extern "C" int openblas_get_parallel(void);
extern "C" void openblas_set_num_threads(int);
#endif

namespace {

#ifndef GWLDCORE_SOURCE_COMMIT
#define GWLDCORE_SOURCE_COMMIT "unknown"
#endif
#ifndef GWLDCORE_COMPILER_ID
#define GWLDCORE_COMPILER_ID "unknown"
#endif
#ifndef GWLDCORE_COMPILER_VERSION
#define GWLDCORE_COMPILER_VERSION "unknown"
#endif
#ifndef GWLDCORE_BUILD_TYPE
#define GWLDCORE_BUILD_TYPE "unknown"
#endif
#ifndef GWLDCORE_BLAS_VENDOR
#define GWLDCORE_BLAS_VENDOR "unknown"
#endif
#ifndef GWLDCORE_NATIVE_OPT
#define GWLDCORE_NATIVE_OPT 0
#endif
#ifndef GWLDCORE_OPENMP_ENABLED
#define GWLDCORE_OPENMP_ENABLED 0
#endif
#ifndef GWLDCORE_GEMM_INTEGRITY_ENABLED
#define GWLDCORE_GEMM_INTEGRITY_ENABLED 0
#endif
#ifndef GWLDCORE_PRIVATE_OPENBLAS_ENABLED
#define GWLDCORE_PRIVATE_OPENBLAS_ENABLED 0
#endif
#ifndef GWLDCORE_PRIVATE_OPENBLAS_SHA256
#define GWLDCORE_PRIVATE_OPENBLAS_SHA256 "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_ENABLED
#define GWLDCORE_PRIVATE_BLAS_ENABLED 0
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_BACKEND
#define GWLDCORE_PRIVATE_BLAS_BACKEND "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_SHA256
#define GWLDCORE_PRIVATE_BLAS_SHA256 "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_SOURCE_COMMIT
#define GWLDCORE_PRIVATE_BLAS_SOURCE_COMMIT "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256
#define GWLDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256 "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY
#define GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_HEADER_SHA256
#define GWLDCORE_PRIVATE_BLAS_HEADER_SHA256 "none"
#endif
#ifndef GWLDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256
#define GWLDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256 "none"
#endif

constexpr double kOrthonormalTolerance = 1.0e-10;
constexpr double kStandardizedEnvTolerance = 1.0e-8;
constexpr size_t kGemmTelemetryCapacity = 16384;
constexpr size_t kNumaPageSamplesPerOperand = 8;
constexpr size_t kGemmOperandCount = 3;
constexpr size_t kMaximumGemmNumaPageSamples =
    kNumaPageSamplesPerOperand * kGemmOperandCount;
constexpr int kNumaAddressSelectionSchemaVersion = 1;
constexpr const char* kNumaAddressSelectionPolicy =
    "evenly_spaced_fully_contained_page_bases";
constexpr const char* kNativeIntegritySnapshotNumaSchema =
    "summit.native_integrity_snapshot_numa.v1";
constexpr const char* kNativeIntegritySnapshotOperandRole =
    "native_integrity_snapshot_of_logical_b";
constexpr size_t kNativeIntegritySnapshotQueryChunkPages = 65536;
constexpr const char* kNativeGemmOutputNumaSchema =
    "summit.native_gemm_output_numa.v1";
constexpr const char* kNativeGemmOutputOperandRole =
    "protected_gemm_output";
constexpr size_t kNativeGemmOutputQueryChunkPages = 65536;
constexpr size_t kNativeGemmOutputEvidenceCapacity = 16384;
constexpr int kMultiEnvironmentMailmanMaximumProbes = 10;
constexpr int kMpolBind = 2;
constexpr int kMpolStaticNodes = 1 << 15;
constexpr int kStaticMembindPolicy = kMpolBind | kMpolStaticNodes;
constexpr unsigned long kMpolAddress = 1UL << 1;
constexpr unsigned long kMpolStrict = 1UL << 0;

struct OpenMPEntryState {
    bool in_parallel = false;
    int level = 0;
    int active_level = 0;
    int max_active_levels = 0;
    int max_threads = 1;
    int num_threads = 1;
    int thread_num = 0;
};

OpenMPEntryState capture_openmp_entry_state() noexcept {
    OpenMPEntryState state;
#ifdef _OPENMP
    state.in_parallel = omp_in_parallel() != 0;
    state.level = omp_get_level();
    state.active_level = omp_get_active_level();
    state.max_active_levels = omp_get_max_active_levels();
    state.max_threads = omp_get_max_threads();
    state.num_threads = omp_get_num_threads();
    state.thread_num = omp_get_thread_num();
#endif
    return state;
}

bool is_nested_vendor_entry(const OpenMPEntryState& state) noexcept {
    // omp_in_parallel() can be false for a serialized parallel region.  A
    // nonzero lexical level is still an outer OpenMP region and must not be
    // allowed to become the parent of a vendor BLAS team.
    return state.in_parallel || state.level != 0 || state.active_level != 0;
}

void require_vendor_entry_outside_openmp(const OpenMPEntryState& state) {
    if (!is_nested_vendor_entry(state)) return;
    throw std::runtime_error(
        "Refusing GxE vendor BLAS entry from inside an outer OpenMP region "
        "(omp_in_parallel=" + std::to_string(state.in_parallel ? 1 : 0) +
        ", omp_level=" + std::to_string(state.level) +
        ", omp_active_level=" + std::to_string(state.active_level) + ")"
    );
}

struct CpuAffinityEvidence {
    int current_cpu = -1;
    int cpu_count = -1;
    std::string cpu_list;
};

constexpr const char* kOpenMPPlacementContractSchema =
    "summit.openmp_placement_attestation.v1";
constexpr const char* kOpenMPEffectiveCapacityPolicy =
    "bound_places_else_sched_affinity_v1";

#if defined(__linux__) && defined(_OPENMP) && _OPENMP >= 201307
constexpr bool kOpenMPPlacementContractSupported = true;
#else
constexpr bool kOpenMPPlacementContractSupported = false;
#endif

CpuAffinityEvidence capture_cpu_affinity_evidence() {
    CpuAffinityEvidence evidence;
#if defined(__linux__)
    evidence.current_cpu = ::sched_getcpu();
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (::sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        return evidence;
    }
    evidence.cpu_count = CPU_COUNT(&affinity);
    for (int first = 0; first < CPU_SETSIZE;) {
        while (first < CPU_SETSIZE && !CPU_ISSET(first, &affinity)) ++first;
        if (first == CPU_SETSIZE) break;
        int last = first;
        while (last + 1 < CPU_SETSIZE && CPU_ISSET(last + 1, &affinity)) {
            ++last;
        }
        if (!evidence.cpu_list.empty()) evidence.cpu_list += ',';
        evidence.cpu_list += std::to_string(first);
        if (last != first) {
            evidence.cpu_list += '-';
            evidence.cpu_list += std::to_string(last);
        }
        first = last + 1;
    }
#endif
    return evidence;
}

const char* openmp_proc_bind_name(int binding) noexcept {
    // OpenMP 5.1 renamed the numeric value 2 from "master" to "primary".
    // Use the historic spelling so the schema is stable across toolchains.
    switch (binding) {
        case 0: return "false";
        case 1: return "true";
        case 2: return "master";
        case 3: return "close";
        case 4: return "spread";
        default: return "unknown";
    }
}

int effective_openmp_capacity() noexcept {
#ifdef _OPENMP
    const int thread_limit = std::max(1, omp_get_thread_limit());
#if _OPENMP >= 201307
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        // A bound initial thread normally exposes only its singleton place via
        // sched_getaffinity().  The team capacity is the number of usable
        // runtime places, not that calling-thread mask.
        const int places = omp_get_num_places();
        if (places <= 0) return 0;
        int nonempty_places = 0;
        for (int place = 0; place < places; ++place) {
            if (omp_get_place_num_procs(place) > 0) ++nonempty_places;
        }
        return std::min(thread_limit, nonempty_places);
    }
#endif
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (::sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int affinity_count = CPU_COUNT(&affinity);
        if (affinity_count > 0) return std::min(thread_limit, affinity_count);
    }
#endif
    return thread_limit;
#else
    return 1;
#endif
}

struct OpenMPPlacementEnvironmentValue {
    std::string name;
    bool present = false;
    std::string value;

    bool operator==(const OpenMPPlacementEnvironmentValue& other) const noexcept {
        return name == other.name && present == other.present
            && value == other.value;
    }
};

using OpenMPPlacementEnvironmentSnapshot =
    std::vector<OpenMPPlacementEnvironmentValue>;

constexpr std::array<const char*, 7> kOpenMPPlacementEnvironmentNames{{
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_MAX_ACTIVE_LEVELS",
    "GOMP_CPU_AFFINITY",
}};

OpenMPPlacementEnvironmentSnapshot capture_openmp_placement_environment() {
    OpenMPPlacementEnvironmentSnapshot snapshot;
    snapshot.reserve(kOpenMPPlacementEnvironmentNames.size());
    for (const char* name : kOpenMPPlacementEnvironmentNames) {
        const char* value = std::getenv(name);
        snapshot.push_back(OpenMPPlacementEnvironmentValue{
            name,
            value != nullptr,
            value == nullptr ? "" : value,
        });
    }
    return snapshot;
}

const OpenMPPlacementEnvironmentValue& openmp_placement_environment_value(
    const OpenMPPlacementEnvironmentSnapshot& snapshot,
    const char* name
) {
    const auto found = std::find_if(
        snapshot.begin(), snapshot.end(),
        [name](const OpenMPPlacementEnvironmentValue& value) {
            return value.name == name;
        }
    );
    if (found == snapshot.end()) {
        throw std::logic_error(
            std::string("Internal OpenMP placement contract omits ") + name
        );
    }
    return *found;
}

struct OpenMPPlacementWorkerEvidence {
    int thread_num = -1;
    int place_num = -1;
    std::vector<int> place_cpu_ids;
    std::vector<int> sched_affinity_cpu_ids;
    int sched_affinity_errno = 0;
    int current_cpu = -1;
    bool verified = false;
};

struct OpenMPPlacementEvidence {
    int requested_threads = 0;
    std::vector<int> expected_cpu_ids;
    bool omp_dynamic = false;
    int omp_thread_limit = 0;
    int omp_max_active_levels = 0;
    std::string omp_proc_bind;
    bool omp_binding_active = false;
    int omp_num_places = 0;
    int effective_capacity = 0;
    std::vector<std::vector<int>> place_cpu_ids;
    int team_size = 0;
    bool exact_singleton_places = false;
    bool exact_team_coverage = false;
    std::vector<OpenMPPlacementWorkerEvidence> workers;
    bool verified = false;
};

struct FixedOpenMPPlacementRuntime {
    std::mutex mutex;
    bool configured = false;
    int requested_threads = 0;
    std::vector<int> expected_cpu_ids;
    OpenMPPlacementEnvironmentSnapshot environment;
    OpenMPPlacementEvidence evidence;
};

FixedOpenMPPlacementRuntime& fixed_openmp_placement_runtime() {
    static FixedOpenMPPlacementRuntime runtime;
    return runtime;
}

void validate_configured_openmp_placement_for_entry_locked(
    const FixedOpenMPPlacementRuntime& runtime
) {
    if (!runtime.configured) return;
    if (capture_openmp_placement_environment() != runtime.environment) {
        throw std::runtime_error(
            "The OpenMP placement environment changed after configuration"
        );
    }
#if defined(__linux__) && defined(_OPENMP) && _OPENMP >= 201307
    if (omp_get_dynamic() != 0 ||
        omp_get_thread_limit() != runtime.evidence.omp_thread_limit ||
        omp_get_max_active_levels() !=
            runtime.evidence.omp_max_active_levels ||
        openmp_proc_bind_name(static_cast<int>(omp_get_proc_bind())) !=
            runtime.evidence.omp_proc_bind ||
        omp_get_num_places() != runtime.requested_threads ||
        effective_openmp_capacity() < runtime.requested_threads) {
        throw std::runtime_error(
            "The OpenMP runtime placement changed after configuration"
        );
    }
    for (int place = 0; place < runtime.requested_threads; ++place) {
        if (omp_get_place_num_procs(place) != 1) {
            throw std::runtime_error(
                "The OpenMP singleton places changed after configuration"
            );
        }
        int cpu = -1;
        omp_get_place_proc_ids(place, &cpu);
        if (cpu != runtime.expected_cpu_ids[static_cast<size_t>(place)]) {
            throw std::runtime_error(
                "The OpenMP singleton places changed after configuration"
            );
        }
    }

    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (::sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        throw std::runtime_error(
            "Could not verify the configured OpenMP calling-thread affinity"
        );
    }
    bool affinity_nonempty = false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &affinity)) continue;
        affinity_nonempty = true;
        if (std::find(
                runtime.expected_cpu_ids.begin(),
                runtime.expected_cpu_ids.end(), cpu
            ) == runtime.expected_cpu_ids.end()) {
            throw std::runtime_error(
                "The calling-thread affinity escaped the configured OpenMP CPU set"
            );
        }
    }
    const int cpu = ::sched_getcpu();
    if (!affinity_nonempty || cpu < 0 || !CPU_ISSET(cpu, &affinity) ||
        std::find(
            runtime.expected_cpu_ids.begin(),
            runtime.expected_cpu_ids.end(), cpu
        ) == runtime.expected_cpu_ids.end()) {
        throw std::runtime_error(
            "The calling-thread CPU is outside the configured OpenMP CPU set"
        );
    }
#else
    throw std::runtime_error(
        "The configured OpenMP placement cannot be verified on this platform"
    );
#endif
}

void validate_configured_openmp_placement_for_entry() {
    auto& runtime = fixed_openmp_placement_runtime();
    const std::lock_guard<std::mutex> lock(runtime.mutex);
    validate_configured_openmp_placement_for_entry_locked(runtime);
}

std::vector<int> configured_openmp_cpu_ids() {
    auto& runtime = fixed_openmp_placement_runtime();
    const std::lock_guard<std::mutex> lock(runtime.mutex);
    return runtime.configured ? runtime.expected_cpu_ids : std::vector<int>{};
}

#if defined(__linux__)
class ScopedBlisPthreadAffinity {
public:
    ScopedBlisPthreadAffinity() {
        const std::vector<int> cpu_ids = configured_openmp_cpu_ids();
        if (cpu_ids.empty()) return;
        CPU_ZERO(&original_);
        if (::sched_getaffinity(0, sizeof(original_), &original_) != 0) {
            throw std::runtime_error(
                "Could not capture the BLIS caller affinity before pthread entry"
            );
        }
        cpu_set_t selected;
        CPU_ZERO(&selected);
        for (const int cpu : cpu_ids) {
            if (cpu < 0 || cpu >= CPU_SETSIZE) {
                throw std::runtime_error(
                    "Configured BLIS pthread CPU is outside cpu_set_t"
                );
            }
            CPU_SET(cpu, &selected);
        }
        if (::sched_setaffinity(0, sizeof(selected), &selected) != 0) {
            throw std::runtime_error(
                "Could not expose the selected CPU set to BLIS pthread workers"
            );
        }
        changed_ = true;
    }

    ScopedBlisPthreadAffinity(const ScopedBlisPthreadAffinity&) = delete;
    ScopedBlisPthreadAffinity& operator=(
        const ScopedBlisPthreadAffinity&
    ) = delete;

    ~ScopedBlisPthreadAffinity() {
        if (changed_) {
            (void)::sched_setaffinity(0, sizeof(original_), &original_);
        }
    }

    void restore() {
        if (!changed_) return;
        if (::sched_setaffinity(0, sizeof(original_), &original_) != 0) {
            throw std::runtime_error(
                "Could not restore the calling-thread affinity after BLIS"
            );
        }
        changed_ = false;
    }

private:
    cpu_set_t original_{};
    bool changed_ = false;
};
#endif

nb::list cpu_ids_to_list(const std::vector<int>& ids) {
    nb::list result;
    for (const int id : ids) result.append(id);
    return result;
}

nb::dict openmp_placement_evidence_to_dict(
    const OpenMPPlacementEvidence& evidence
) {
    nb::dict result;
    result["schema"] = kOpenMPPlacementContractSchema;
    result["schema_version"] = 1;
    result["verified"] = evidence.verified;
    result["immutable"] = true;
    result["requested_threads"] = evidence.requested_threads;
    result["expected_cpu_ids"] = cpu_ids_to_list(evidence.expected_cpu_ids);
    result["omp_dynamic"] = evidence.omp_dynamic;
    result["omp_thread_limit"] = evidence.omp_thread_limit;
    result["omp_max_active_levels"] = evidence.omp_max_active_levels;
    result["omp_proc_bind"] = evidence.omp_proc_bind;
    result["omp_binding_active"] = evidence.omp_binding_active;
    result["omp_num_places"] = evidence.omp_num_places;
    result["effective_openmp_capacity"] = evidence.effective_capacity;
    nb::list places;
    for (const auto& ids : evidence.place_cpu_ids) {
        places.append(cpu_ids_to_list(ids));
    }
    result["place_cpu_ids"] = std::move(places);
    result["team_size"] = evidence.team_size;
    result["exact_singleton_places"] = evidence.exact_singleton_places;
    result["exact_team_coverage"] = evidence.exact_team_coverage;
    nb::list workers;
    for (const auto& worker : evidence.workers) {
        nb::dict item;
        item["thread_num"] = worker.thread_num;
        item["place_num"] = worker.place_num;
        item["place_cpu_ids"] = cpu_ids_to_list(worker.place_cpu_ids);
        item["sched_affinity_cpu_ids"] =
            cpu_ids_to_list(worker.sched_affinity_cpu_ids);
        item["current_cpu"] = worker.current_cpu;
        item["verified"] = worker.verified;
        workers.append(std::move(item));
    }
    result["workers"] = std::move(workers);
    result["vendor_calls"] = 0;
    return result;
}

OpenMPPlacementEvidence probe_openmp_placement(
    const std::vector<int>& expected_cpu_ids,
    int requested_threads
) {
    if (!kOpenMPPlacementContractSupported) {
        throw std::runtime_error(
            "The explicit OpenMP placement contract requires Linux and OpenMP 4.0"
        );
    }
#if defined(__linux__) && defined(_OPENMP) && _OPENMP >= 201307
    if (requested_threads <= 0) {
        throw std::runtime_error(
            "OpenMP placement threads must be positive"
        );
    }
    if (expected_cpu_ids.size() != static_cast<size_t>(requested_threads)) {
        throw std::runtime_error(
            "OpenMP placement CPU count must equal the requested thread count"
        );
    }
    std::vector<int> unique_ids = expected_cpu_ids;
    for (const int cpu : unique_ids) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::runtime_error(
                "OpenMP placement contains a CPU outside the sched_affinity range"
            );
        }
    }
    std::sort(unique_ids.begin(), unique_ids.end());
    if (std::adjacent_find(unique_ids.begin(), unique_ids.end()) !=
        unique_ids.end()) {
        throw std::runtime_error(
            "OpenMP placement CPU IDs must be unique"
        );
    }
    const OpenMPEntryState entry = capture_openmp_entry_state();
    if (is_nested_vendor_entry(entry)) {
        throw std::runtime_error(
            "OpenMP placement must be configured outside every OpenMP region"
        );
    }

    OpenMPPlacementEvidence evidence;
    evidence.requested_threads = requested_threads;
    evidence.expected_cpu_ids = expected_cpu_ids;
    evidence.omp_dynamic = omp_get_dynamic() != 0;
    evidence.omp_thread_limit = omp_get_thread_limit();
    evidence.omp_max_active_levels = omp_get_max_active_levels();
    const int proc_bind = static_cast<int>(omp_get_proc_bind());
    evidence.omp_proc_bind = openmp_proc_bind_name(proc_bind);
    evidence.omp_binding_active = proc_bind != 0;
    evidence.omp_num_places = omp_get_num_places();
    evidence.effective_capacity = effective_openmp_capacity();

    if (evidence.omp_dynamic) {
        throw std::runtime_error(
            "The explicit OpenMP placement contract requires OMP_DYNAMIC=FALSE"
        );
    }
    if (evidence.omp_thread_limit < requested_threads) {
        throw std::runtime_error(
            "The OpenMP thread limit is smaller than the placement team"
        );
    }
    if (evidence.omp_max_active_levels != 1) {
        throw std::runtime_error(
            "The explicit OpenMP placement contract requires exactly one "
            "maximum active level"
        );
    }
    if (!evidence.omp_binding_active) {
        throw std::runtime_error(
            "The explicit OpenMP placement contract requires active OMP_PROC_BIND"
        );
    }
    if (evidence.omp_num_places != requested_threads) {
        throw std::runtime_error(
            "The OpenMP place count differs from the placement team"
        );
    }
    if (evidence.effective_capacity < requested_threads) {
        throw std::runtime_error(
            "The effective OpenMP capacity is smaller than the placement team"
        );
    }

    evidence.place_cpu_ids.reserve(
        static_cast<size_t>(evidence.omp_num_places)
    );
    for (int place = 0; place < evidence.omp_num_places; ++place) {
        const int count = omp_get_place_num_procs(place);
        if (count <= 0) {
            throw std::runtime_error(
                "An OpenMP place contains no processors"
            );
        }
        std::vector<int> ids(static_cast<size_t>(count), -1);
        omp_get_place_proc_ids(place, ids.data());
        evidence.place_cpu_ids.push_back(std::move(ids));
    }
    evidence.exact_singleton_places = true;
    for (int place = 0; place < requested_threads; ++place) {
        const auto& ids = evidence.place_cpu_ids[static_cast<size_t>(place)];
        if (ids.size() != 1 || ids[0] != expected_cpu_ids[static_cast<size_t>(place)]) {
            evidence.exact_singleton_places = false;
            break;
        }
    }
    if (!evidence.exact_singleton_places) {
        throw std::runtime_error(
            "OpenMP places are not the expected ordered singleton CPU set"
        );
    }

    evidence.workers.resize(static_cast<size_t>(requested_threads));
    for (auto& worker : evidence.workers) {
        worker.sched_affinity_cpu_ids.reserve(CPU_SETSIZE);
    }
    std::atomic<int> observed_team_size{0};
    #pragma omp parallel num_threads(requested_threads)
    {
        const int thread_num = omp_get_thread_num();
        if (thread_num == 0) {
            observed_team_size.store(
                omp_get_num_threads(), std::memory_order_relaxed
            );
        }
        auto& worker = evidence.workers[static_cast<size_t>(thread_num)];
        worker.thread_num = thread_num;
        worker.place_num = omp_get_place_num();
        worker.current_cpu = ::sched_getcpu();
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        if (::sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
            worker.sched_affinity_errno = errno;
        } else {
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &affinity)) {
                    worker.sched_affinity_cpu_ids.push_back(cpu);
                }
            }
        }
    }
    evidence.team_size = observed_team_size.load(std::memory_order_relaxed);
    if (evidence.team_size != requested_threads) {
        throw std::runtime_error(
            "The OpenMP placement probe did not create the exact requested team"
        );
    }

    evidence.exact_team_coverage = true;
    for (int thread_num = 0; thread_num < requested_threads; ++thread_num) {
        auto& worker = evidence.workers[static_cast<size_t>(thread_num)];
        if (worker.place_num >= 0 && worker.place_num < evidence.omp_num_places) {
            worker.place_cpu_ids = evidence.place_cpu_ids[
                static_cast<size_t>(worker.place_num)
            ];
        }
        const int expected_cpu = expected_cpu_ids[static_cast<size_t>(thread_num)];
        worker.verified = worker.thread_num == thread_num
            && worker.place_num == thread_num
            && worker.place_cpu_ids == std::vector<int>{expected_cpu}
            && worker.sched_affinity_errno == 0
            && worker.sched_affinity_cpu_ids == std::vector<int>{expected_cpu}
            && worker.current_cpu == expected_cpu;
        if (!worker.verified) evidence.exact_team_coverage = false;
    }
    if (!evidence.exact_team_coverage) {
        throw std::runtime_error(
            "The OpenMP team does not exactly cover the expected pinned CPUs"
        );
    }
    evidence.verified = true;
    return evidence;
#else
    (void)expected_cpu_ids;
    (void)requested_threads;
    throw std::runtime_error(
        "The explicit OpenMP placement contract is unavailable"
    );
#endif
}

nb::dict configure_openmp_placement(
    const std::vector<int>& expected_cpu_ids,
    int requested_threads
) {
    const OpenMPPlacementEnvironmentSnapshot environment =
        capture_openmp_placement_environment();
    const auto& proc_bind = openmp_placement_environment_value(
        environment, "OMP_PROC_BIND"
    );
    const auto& places = openmp_placement_environment_value(
        environment, "OMP_PLACES"
    );
    const auto& gomp_affinity = openmp_placement_environment_value(
        environment, "GOMP_CPU_AFFINITY"
    );
    if (!proc_bind.present || proc_bind.value.empty() ||
        !places.present || places.value.empty()) {
        throw std::runtime_error(
            "The explicit OpenMP placement contract requires process-start "
            "OMP_PROC_BIND and OMP_PLACES"
        );
    }
    if (gomp_affinity.present) {
        throw std::runtime_error(
            "GOMP_CPU_AFFINITY is forbidden when OMP_PLACES is authoritative"
        );
    }

    auto& runtime = fixed_openmp_placement_runtime();
    const std::lock_guard<std::mutex> lock(runtime.mutex);
    if (runtime.configured &&
        (requested_threads != runtime.requested_threads ||
         expected_cpu_ids != runtime.expected_cpu_ids)) {
        throw std::runtime_error(
            "The immutable OpenMP placement was already configured with "
            "different CPU IDs or thread count"
        );
    }
    if (runtime.configured && environment != runtime.environment) {
        throw std::runtime_error(
            "The OpenMP placement environment changed after configuration"
        );
    }
    if (runtime.configured) {
        validate_configured_openmp_placement_for_entry_locked(runtime);
    }
    OpenMPPlacementEvidence evidence = probe_openmp_placement(
        expected_cpu_ids, requested_threads
    );
    if (capture_openmp_placement_environment() != environment) {
        throw std::runtime_error(
            "The OpenMP placement environment changed during configuration"
        );
    }
    if (!runtime.configured) {
        runtime.configured = true;
        runtime.requested_threads = requested_threads;
        runtime.expected_cpu_ids = expected_cpu_ids;
        runtime.environment = environment;
    }
    runtime.evidence = evidence;
    return openmp_placement_evidence_to_dict(runtime.evidence);
}

void append_openmp_placement_build_info(nb::dict& result) {
    result["openmp_effective_capacity_policy"] =
        kOpenMPEffectiveCapacityPolicy;
    result["openmp_placement_contract_supported"] =
        kOpenMPPlacementContractSupported;
    result["openmp_placement_contract_schema"] =
        kOpenMPPlacementContractSchema;
    result["openmp_placement_contract_immutable"] = true;
    result["openmp_placement_probe_vendor_calls"] = 0;
    auto& runtime = fixed_openmp_placement_runtime();
    const std::lock_guard<std::mutex> lock(runtime.mutex);
    result["openmp_placement_contract_configured"] = runtime.configured;
    if (runtime.configured) {
        result["openmp_placement_contract_evidence"] =
            openmp_placement_evidence_to_dict(runtime.evidence);
    } else {
        result["openmp_placement_contract_evidence"] = nb::none();
    }
}

int current_cpu() noexcept {
#if defined(__linux__)
    return ::sched_getcpu();
#else
    return -1;
#endif
}

double process_cpu_seconds_now() noexcept {
#if defined(CLOCK_PROCESS_CPUTIME_ID)
    struct timespec stamp{};
    if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stamp) == 0) {
        return static_cast<double>(stamp.tv_sec)
            + static_cast<double>(stamp.tv_nsec) * 1.0e-9;
    }
#endif
    const std::clock_t ticks = std::clock();
    return ticks == static_cast<std::clock_t>(-1)
        ? 0.0
        : static_cast<double>(ticks) / static_cast<double>(CLOCKS_PER_SEC);
}

enum class NumaPageQueryState {
    not_attempted,
    queried,
    partial,
    permission_denied,
    unsupported,
    syscall_error,
    page_query_failed,
    empty,
    no_full_pages,
    invalid_range,
};

const char* numa_page_query_state_name(NumaPageQueryState state) noexcept {
    switch (state) {
        case NumaPageQueryState::not_attempted: return "not_attempted";
        case NumaPageQueryState::queried: return "queried";
        case NumaPageQueryState::partial: return "partial";
        case NumaPageQueryState::permission_denied: return "permission_denied";
        case NumaPageQueryState::unsupported: return "unsupported";
        case NumaPageQueryState::syscall_error: return "syscall_error";
        case NumaPageQueryState::page_query_failed: return "page_query_failed";
        case NumaPageQueryState::empty: return "empty";
        case NumaPageQueryState::no_full_pages: return "no_full_pages";
        case NumaPageQueryState::invalid_range: return "invalid_range";
    }
    return "unknown";
}

struct OperandNumaPageSamples {
    NumaPageQueryState query_state = NumaPageQueryState::not_attempted;
    size_t page_count_in_storage_span = 0;
    size_t fully_contained_page_count = 0;
    size_t operand_byte_count = 0;
    size_t operand_start_page_offset = 0;
    size_t operand_end_exclusive_page_offset = 0;
    size_t sample_count = 0;
    size_t combined_sample_offset = 0;
    std::array<size_t, kNumaPageSamplesPerOperand> full_page_indices{};
    std::array<size_t, kNumaPageSamplesPerOperand> operand_byte_offsets{};
    std::array<int, kNumaPageSamplesPerOperand> page_status{};
};

struct GemmNumaPageSamples {
    long system_page_size = -1;
    long syscall_result = -1;
    int syscall_errno = 0;
    std::array<OperandNumaPageSamples, kGemmOperandCount> operands{};
};

bool gemm_storage_span_elements(
    size_t major_dimension,
    size_t leading_dimension,
    size_t minor_dimension,
    size_t& span_elements
) noexcept {
    if (major_dimension == 0 || minor_dimension == 0) {
        span_elements = 0;
        return true;
    }
    const size_t major_offset = major_dimension - 1;
    if (leading_dimension != 0
            && major_offset > (
                std::numeric_limits<size_t>::max() - minor_dimension
            ) / leading_dimension) {
        return false;
    }
    span_elements = major_offset * leading_dimension + minor_dimension;
    return true;
}

bool append_operand_numa_page_samples(
    const double* pointer,
    size_t span_elements,
    size_t page_size,
    std::array<void*, kMaximumGemmNumaPageSamples>& pages,
    size_t& combined_count,
    OperandNumaPageSamples& operand
) noexcept {
    if (pointer == nullptr || span_elements == 0) {
        operand.query_state = NumaPageQueryState::empty;
        return true;
    }
    if (span_elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
        operand.query_state = NumaPageQueryState::invalid_range;
        return false;
    }
    const size_t byte_count = span_elements * sizeof(double);
    const uintptr_t first_byte = reinterpret_cast<uintptr_t>(pointer);
    if (byte_count == 0
            || byte_count - 1 > std::numeric_limits<uintptr_t>::max() - first_byte) {
        operand.query_state = NumaPageQueryState::invalid_range;
        return false;
    }
    const uintptr_t last_byte = first_byte + byte_count - 1;
    const uintptr_t first_page = first_byte - first_byte % page_size;
    const uintptr_t last_page = last_byte - last_byte % page_size;
    const size_t storage_pages =
        static_cast<size_t>((last_page - first_page) / page_size) + 1;
    operand.page_count_in_storage_span = storage_pages;
    operand.operand_byte_count = byte_count;
    operand.operand_start_page_offset =
        static_cast<size_t>(first_byte % page_size);
    operand.operand_end_exclusive_page_offset = (
        operand.operand_start_page_offset + byte_count % page_size
    ) % page_size;

    // A boundary page may be shared with an adjacent allocation, so its single
    // page-level placement cannot be attributed exclusively to this operand.
    // Select only pages whose complete [base, base + page_size) range lies
    // inside [pointer, pointer + byte_count).
    const size_t bytes_to_first_full_page =
        operand.operand_start_page_offset == 0
            ? 0
            : page_size - operand.operand_start_page_offset;
    const size_t full_pages = bytes_to_first_full_page <= byte_count
        ? (byte_count - bytes_to_first_full_page) / page_size
        : 0;
    operand.fully_contained_page_count = full_pages;
    if (full_pages == 0) {
        operand.query_state = NumaPageQueryState::no_full_pages;
        return true;
    }

    const uintptr_t first_full_page = first_byte + bytes_to_first_full_page;
    const size_t samples = std::min(full_pages, kNumaPageSamplesPerOperand);
    operand.sample_count = samples;
    operand.combined_sample_offset = combined_count;
    operand.query_state = NumaPageQueryState::not_attempted;
    for (size_t sample = 0; sample < samples; ++sample) {
        const size_t page_index = samples == 1
            ? 0
            : sample * (full_pages - 1) / (samples - 1);
        const size_t byte_offset =
            bytes_to_first_full_page + page_index * page_size;
        operand.full_page_indices[sample] = page_index;
        operand.operand_byte_offsets[sample] = byte_offset;
        pages[combined_count++] = reinterpret_cast<void*>(
            first_full_page + page_index * page_size
        );
    }
    return true;
}

GemmNumaPageSamples sample_gemm_operand_numa_pages(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transpose_a,
    CBLAS_TRANSPOSE transpose_b,
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    const double* c,
    int ldc
) noexcept {
    GemmNumaPageSamples result;
#if defined(__linux__) && defined(SYS_move_pages)
    result.system_page_size = ::sysconf(_SC_PAGESIZE);
    if (result.system_page_size <= 0) {
        for (auto& operand : result.operands) {
            operand.query_state = NumaPageQueryState::unsupported;
        }
        return result;
    }
    const size_t page_size = static_cast<size_t>(result.system_page_size);
    std::array<void*, kMaximumGemmNumaPageSamples> pages{};
    std::array<int, kMaximumGemmNumaPageSamples> statuses{};
    size_t combined_count = 0;
    size_t a_span = 0;
    size_t b_span = 0;
    size_t c_span = 0;
    const size_t a_rows = static_cast<size_t>(std::max(
        0, transpose_a == CblasNoTrans ? m : k
    ));
    const size_t a_columns = static_cast<size_t>(std::max(
        0, transpose_a == CblasNoTrans ? k : m
    ));
    const size_t b_rows = static_cast<size_t>(std::max(
        0, transpose_b == CblasNoTrans ? k : n
    ));
    const size_t b_columns = static_cast<size_t>(std::max(
        0, transpose_b == CblasNoTrans ? n : k
    ));
    const size_t c_rows = static_cast<size_t>(std::max(0, m));
    const size_t c_columns = static_cast<size_t>(std::max(0, n));
    const bool column_major = layout == CblasColMajor;
    const bool a_span_valid = gemm_storage_span_elements(
        column_major ? a_columns : a_rows,
        static_cast<size_t>(std::max(0, lda)),
        column_major ? a_rows : a_columns,
        a_span
    );
    const bool b_span_valid = gemm_storage_span_elements(
        column_major ? b_columns : b_rows,
        static_cast<size_t>(std::max(0, ldb)),
        column_major ? b_rows : b_columns,
        b_span
    );
    const bool c_span_valid = gemm_storage_span_elements(
        column_major ? c_columns : c_rows,
        static_cast<size_t>(std::max(0, ldc)),
        column_major ? c_rows : c_columns,
        c_span
    );
    if (!a_span_valid) {
        result.operands[0].query_state = NumaPageQueryState::invalid_range;
    } else {
        append_operand_numa_page_samples(
            a, a_span, page_size, pages, combined_count, result.operands[0]
        );
    }
    if (!b_span_valid) {
        result.operands[1].query_state = NumaPageQueryState::invalid_range;
    } else {
        append_operand_numa_page_samples(
            b, b_span, page_size, pages, combined_count, result.operands[1]
        );
    }
    if (!c_span_valid) {
        result.operands[2].query_state = NumaPageQueryState::invalid_range;
    } else {
        append_operand_numa_page_samples(
            c, c_span, page_size, pages, combined_count, result.operands[2]
        );
    }
    if (combined_count == 0) return result;

    errno = 0;
    result.syscall_result = ::syscall(
        SYS_move_pages,
        0,
        static_cast<unsigned long>(combined_count),
        pages.data(),
        nullptr,
        statuses.data(),
        0
    );
    if (result.syscall_result < 0) {
        result.syscall_errno = errno;
        const NumaPageQueryState state =
            (errno == EPERM || errno == EACCES)
                ? NumaPageQueryState::permission_denied
                : errno == ENOSYS
                    ? NumaPageQueryState::unsupported
                    : NumaPageQueryState::syscall_error;
        for (auto& operand : result.operands) {
            if (operand.query_state == NumaPageQueryState::not_attempted) {
                operand.query_state = state;
            }
        }
        return result;
    }

    for (auto& operand : result.operands) {
        if (operand.query_state != NumaPageQueryState::not_attempted) continue;
        size_t successful = 0;
        for (size_t sample = 0; sample < operand.sample_count; ++sample) {
            const int status = statuses[operand.combined_sample_offset + sample];
            operand.page_status[sample] = status;
            successful += status >= 0 ? 1 : 0;
        }
        operand.query_state = successful == operand.sample_count
            ? NumaPageQueryState::queried
            : successful == 0
                ? NumaPageQueryState::page_query_failed
                : NumaPageQueryState::partial;
    }
#else
    for (auto& operand : result.operands) {
        operand.query_state = NumaPageQueryState::unsupported;
    }
#endif
    return result;
}

nb::dict operand_numa_page_samples_to_dict(
    const OperandNumaPageSamples& operand
) {
    nb::dict node_histogram;
    nb::dict page_error_histogram;
    nb::list ordered_samples;
    const bool page_results_available =
        operand.query_state == NumaPageQueryState::queried
        || operand.query_state == NumaPageQueryState::partial
        || operand.query_state == NumaPageQueryState::page_query_failed;
    size_t resolved_pages = 0;
    size_t page_error_count = 0;
    for (size_t sample = 0;
         page_results_available && sample < operand.sample_count;
         ++sample) {
        const int status = operand.page_status[sample];
        resolved_pages += status >= 0 ? 1 : 0;
        page_error_count += status < 0 ? 1 : 0;
        bool already_counted = false;
        for (size_t previous = 0; previous < sample; ++previous) {
            if (operand.page_status[previous] == status) {
                already_counted = true;
                break;
            }
        }
        if (already_counted) continue;
        int count = 1;
        for (size_t later = sample + 1; later < operand.sample_count; ++later) {
            count += operand.page_status[later] == status ? 1 : 0;
        }
        const long long status_magnitude = status >= 0
            ? static_cast<long long>(status)
            : -static_cast<long long>(status);
        const std::string key = std::to_string(status_magnitude);
        if (status >= 0) {
            node_histogram[key.c_str()] = count;
        } else {
            page_error_histogram[key.c_str()] = count;
        }
    }
    for (size_t sample = 0; sample < operand.sample_count; ++sample) {
        nb::dict evidence;
        evidence["sample_ordinal"] = sample;
        evidence["full_page_index"] = operand.full_page_indices[sample];
        evidence["byte_offset_from_operand_start"] =
            operand.operand_byte_offsets[sample];
        if (!page_results_available) {
            evidence["status_kind"] = "unavailable";
            evidence["raw_move_pages_status"] = nb::none();
            evidence["numa_node"] = nb::none();
            evidence["page_query_errno"] = nb::none();
        } else if (operand.page_status[sample] >= 0) {
            evidence["status_kind"] = "numa_node";
            evidence["raw_move_pages_status"] = operand.page_status[sample];
            evidence["numa_node"] = operand.page_status[sample];
            evidence["page_query_errno"] = nb::none();
        } else {
            evidence["status_kind"] = "page_query_error";
            evidence["raw_move_pages_status"] = operand.page_status[sample];
            evidence["numa_node"] = nb::none();
            evidence["page_query_errno"] =
                -static_cast<long long>(operand.page_status[sample]);
        }
        ordered_samples.append(std::move(evidence));
    }
    nb::dict result;
    result["query_status"] = numa_page_query_state_name(operand.query_state);
    result["storage_span_pages"] = operand.page_count_in_storage_span;
    result["fully_contained_pages"] = operand.fully_contained_page_count;
    result["operand_byte_count"] = operand.operand_byte_count;
    result["operand_start_address_page_offset"] =
        operand.operand_start_page_offset;
    result["operand_end_exclusive_address_page_offset"] =
        operand.operand_end_exclusive_page_offset;
    result["selected_sample_pages"] = operand.sample_count;
    result["resolved_sample_pages"] = resolved_pages;
    result["page_query_error_pages"] = page_error_count;
    result["node_histogram"] = std::move(node_histogram);
    result["page_error_errno_histogram"] = std::move(page_error_histogram);
    result["ordered_samples"] = std::move(ordered_samples);
    return result;
}

nb::dict gemm_numa_page_samples_to_dict(const GemmNumaPageSamples& samples) {
    nb::dict operands;
    operands["a"] = operand_numa_page_samples_to_dict(samples.operands[0]);
    operands["b"] = operand_numa_page_samples_to_dict(samples.operands[1]);
    operands["c"] = operand_numa_page_samples_to_dict(samples.operands[2]);
    nb::dict result;
    // Keep schema_version=1 because every original key and value type remains
    // available.  The refined address-selection contract is independently
    // versioned so schema-1 consumers remain compatible while new validators
    // can require boundary-safe samples and ordered evidence.
    result["schema_version"] = 1;
    result["sampling_method"] = "move_pages_query_no_migration";
    result["sampling_timing"] = "after_vendor_call_outside_timed_interval";
    result["sample_limit_per_operand"] = kNumaPageSamplesPerOperand;
    result["address_selection_schema_version"] =
        kNumaAddressSelectionSchemaVersion;
    result["address_selection_policy"] = kNumaAddressSelectionPolicy;
    result["selected_addresses_are_page_bases"] = true;
    result["partial_boundary_pages_included"] = false;
    result["first_and_last_fully_contained_pages_selected"] = true;
    result["virtual_addresses_exposed"] = false;
    result["operand_byte_range_semantics"] =
        "[start_address,end_exclusive_address)";
    result["address_evidence"] =
        "ordered_samples_with_operand_relative_byte_offsets_and_full_page_indices";
    result["system_page_size"] = samples.system_page_size;
    result["syscall_result"] = samples.syscall_result;
    result["syscall_errno"] = samples.syscall_errno;
    result["operands"] = std::move(operands);
    return result;
}

nb::dict test_operand_numa_page_selection(nb_vec1_ro<double> operand_values) {
#if defined(__linux__)
    const long observed_page_size = ::sysconf(_SC_PAGESIZE);
    if (observed_page_size <= 0) {
        throw std::runtime_error(
            "Could not determine the page size for the NUMA selection test"
        );
    }
    std::array<void*, kMaximumGemmNumaPageSamples> pages{};
    size_t combined_count = 0;
    OperandNumaPageSamples operand;
    append_operand_numa_page_samples(
        operand_values.data(), operand_values.shape(0),
        static_cast<size_t>(observed_page_size), pages, combined_count, operand
    );
    nb::dict result = operand_numa_page_samples_to_dict(operand);
    nb::list selected_page_base_relative_offsets;
    bool selected_page_bases_match_ordered_offsets =
        combined_count == operand.sample_count;
    const uintptr_t operand_start =
        reinterpret_cast<uintptr_t>(operand_values.data());
    for (size_t sample = 0; sample < combined_count; ++sample) {
        const uintptr_t selected_page = reinterpret_cast<uintptr_t>(pages[sample]);
        const size_t relative_offset = static_cast<size_t>(
            selected_page - operand_start
        );
        selected_page_base_relative_offsets.append(relative_offset);
        selected_page_bases_match_ordered_offsets =
            selected_page_bases_match_ordered_offsets
            && relative_offset == operand.operand_byte_offsets[sample];
    }
    result["system_page_size"] = observed_page_size;
    result["address_selection_schema_version"] =
        kNumaAddressSelectionSchemaVersion;
    result["address_selection_policy"] = kNumaAddressSelectionPolicy;
    result["partial_boundary_pages_included"] = false;
    result["selected_page_base_relative_offsets"] =
        std::move(selected_page_base_relative_offsets);
    result["selected_page_bases_match_ordered_offsets"] =
        selected_page_bases_match_ordered_offsets;
    return result;
#else
    (void)operand_values;
    throw std::runtime_error(
        "NUMA page selection observability is supported only on Linux"
    );
#endif
}

class Sha256 {
public:
    Sha256() noexcept
        : state_{
              0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
          } {}

    void update(const unsigned char* bytes, size_t byte_count) noexcept {
        bit_count_ += static_cast<uint64_t>(byte_count) * 8U;
        while (byte_count > 0) {
            const size_t copied = std::min(byte_count, buffer_.size() - used_);
            std::memcpy(buffer_.data() + used_, bytes, copied);
            used_ += copied;
            bytes += copied;
            byte_count -= copied;
            if (used_ == buffer_.size()) {
                transform(buffer_.data());
                used_ = 0;
            }
        }
    }

    std::string finish() {
        buffer_[used_++] = 0x80U;
        if (used_ > 56) {
            std::fill(buffer_.begin() + static_cast<ptrdiff_t>(used_),
                      buffer_.end(), 0U);
            transform(buffer_.data());
            used_ = 0;
        }
        std::fill(buffer_.begin() + static_cast<ptrdiff_t>(used_),
                  buffer_.begin() + 56, 0U);
        for (size_t index = 0; index < 8; ++index) {
            buffer_[63 - index] = static_cast<unsigned char>(
                bit_count_ >> (index * 8U)
            );
        }
        transform(buffer_.data());
        std::ostringstream encoded;
        encoded << std::hex << std::setfill('0');
        for (uint32_t word : state_) {
            encoded << std::setw(8) << word;
        }
        return encoded.str();
    }

private:
    static uint32_t rotate_right(uint32_t value, unsigned count) noexcept {
        return (value >> count) | (value << (32U - count));
    }

    void transform(const unsigned char* block) noexcept {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<uint32_t, 64> schedule{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t offset = index * 4;
            schedule[index] =
                static_cast<uint32_t>(block[offset]) << 24U
                | static_cast<uint32_t>(block[offset + 1]) << 16U
                | static_cast<uint32_t>(block[offset + 2]) << 8U
                | static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t index = 16; index < schedule.size(); ++index) {
            const uint32_t previous_15 = schedule[index - 15];
            const uint32_t previous_2 = schedule[index - 2];
            const uint32_t sigma_0 =
                rotate_right(previous_15, 7)
                ^ rotate_right(previous_15, 18)
                ^ (previous_15 >> 3U);
            const uint32_t sigma_1 =
                rotate_right(previous_2, 17)
                ^ rotate_right(previous_2, 19)
                ^ (previous_2 >> 10U);
            schedule[index] = schedule[index - 16] + sigma_0
                + schedule[index - 7] + sigma_1;
        }
        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t index = 0; index < schedule.size(); ++index) {
            const uint32_t sum_1 = rotate_right(e, 6)
                ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp_1 = h + sum_1 + choice
                + constants[index] + schedule[index];
            const uint32_t sum_0 = rotate_right(a, 2)
                ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp_2 = sum_0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp_1;
            d = c;
            c = b;
            b = a;
            a = temp_1 + temp_2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<unsigned char, 64> buffer_{};
    size_t used_ = 0;
    uint64_t bit_count_ = 0;
};

struct NativeIntegritySnapshotNumaEvidence {
    bool contract_required = false;
    bool integrity_check_shape_eligible = false;
    bool integrity_check_executed = false;
    bool snapshot_available = false;
    size_t logical_byte_count = 0;
    size_t mapping_bytes = 0;
    size_t page_size = 0;
    size_t mapping_page_count = 0;
    std::vector<int> selected_nodes;
    bool anonymous_private_mapping = false;
    bool page_aligned_mapping = false;
    bool bound_before_first_touch = false;
    bool live_owner_policy_verified = false;
    bool pre_touch_range_policy_verified = false;
    bool pre_vendor_complete_page_query = false;
    size_t queried_pages = 0;
    size_t resolved_pages = 0;
    size_t query_chunks = 0;
    std::map<int, size_t> node_histogram;
    std::string ordered_status_sha256;
    bool pre_vendor_strict_policy_verified = false;
    bool sealed_read_only_before_vendor = false;
    bool complete = false;
};

struct NativeGemmOutputNumaEvidenceData {
    bool applicable = false;
    uint64_t call_id = 0;
    bool contract_required = false;
    bool complete = false;
    std::string allocation_mode;
    size_t logical_rows = 0;
    size_t logical_columns = 0;
    std::string storage_layout;
    size_t logical_byte_count = 0;
    size_t capacity_byte_count = 0;
    size_t mapping_bytes = 0;
    size_t page_size = 0;
    size_t mapping_page_count = 0;
    std::vector<int> selected_nodes;
    bool anonymous_private_mapping = false;
    bool page_aligned_mapping = false;
    bool writable_output = false;
    bool bound_before_first_touch = false;
    bool pre_touch_live_owner_policy_verified = false;
    bool pre_touch_range_policy_verified = false;
    bool post_repair_live_owner_policy_verified = false;
    bool post_repair_range_policy_verified = false;
    bool post_repair_complete_page_query = false;
    size_t queried_pages = 0;
    size_t resolved_pages = 0;
    size_t query_chunks = 0;
    std::map<int, size_t> node_histogram;
    std::string ordered_status_sha256;
    bool post_repair_strict_policy_verified = false;
};

class SharedNativeGemmOutputNumaEvidence {
public:
    explicit SharedNativeGemmOutputNumaEvidence(
        NativeGemmOutputNumaEvidenceData initial
    ) : data_(std::move(initial)) {}

    NativeGemmOutputNumaEvidenceData snapshot() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return data_;
    }

    template <typename Function>
    void update(Function&& function) {
        const std::lock_guard<std::mutex> lock(mutex_);
        function(data_);
    }

private:
    mutable std::mutex mutex_;
    NativeGemmOutputNumaEvidenceData data_;
};

std::string trim_ascii_whitespace(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

int parse_nonnegative_decimal(const std::string& value, const char* label) {
    if (value.empty()) {
        throw std::runtime_error(std::string("Empty NUMA node in ") + label);
    }
    uint64_t parsed = 0;
    for (char character : value) {
        if (character < '0' || character > '9') {
            throw std::runtime_error(
                std::string("Invalid NUMA node list in ") + label
            );
        }
        parsed = parsed * 10U + static_cast<unsigned>(character - '0');
        if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                std::string("NUMA node exceeds the integer range in ") + label
            );
        }
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_numa_node_list(
    const std::string& raw_value,
    const char* label
) {
    const std::string value = trim_ascii_whitespace(raw_value);
    if (value.empty()) {
        throw std::runtime_error(std::string("Empty NUMA node list in ") + label);
    }
    std::vector<int> nodes;
    size_t component_start = 0;
    while (component_start <= value.size()) {
        const size_t comma = value.find(',', component_start);
        const size_t component_end = comma == std::string::npos
            ? value.size() : comma;
        const std::string component = trim_ascii_whitespace(
            value.substr(component_start, component_end - component_start)
        );
        if (component.empty()) {
            throw std::runtime_error(
                std::string("Empty NUMA node-list component in ") + label
            );
        }
        const size_t dash = component.find('-');
        const int first = parse_nonnegative_decimal(
            trim_ascii_whitespace(component.substr(0, dash)), label
        );
        const int last = dash == std::string::npos
            ? first
            : parse_nonnegative_decimal(
                  trim_ascii_whitespace(component.substr(dash + 1)), label
              );
        if (dash != std::string::npos
            && component.find('-', dash + 1) != std::string::npos) {
            throw std::runtime_error(
                std::string("Invalid NUMA node range in ") + label
            );
        }
        if (last < first || static_cast<uint64_t>(last - first) > 1048576U) {
            throw std::runtime_error(
                std::string("Invalid NUMA node range in ") + label
            );
        }
        for (int node = first;; ++node) {
            nodes.push_back(node);
            if (node == last) break;
        }
        if (comma == std::string::npos) break;
        component_start = comma + 1;
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    if (nodes.empty()) {
        throw std::runtime_error(std::string("Empty NUMA node list in ") + label);
    }
    return nodes;
}

std::string canonical_numa_node_list(const std::vector<int>& nodes) {
    std::ostringstream result;
    for (size_t index = 0; index < nodes.size(); ++index) {
        if (index != 0) result << ',';
        result << nodes[index];
    }
    return result.str();
}

std::string read_first_prefixed_line(
    const char* path,
    const char* prefix,
    const char* label
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("Cannot read ") + label);
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.compare(0, std::strlen(prefix), prefix) == 0) {
            return trim_ascii_whitespace(line.substr(std::strlen(prefix)));
        }
    }
    throw std::runtime_error(std::string("Missing ") + label);
}

std::vector<int> read_current_mems_allowed_nodes() {
    return parse_numa_node_list(
        read_first_prefixed_line(
            "/proc/self/status", "Mems_allowed_list:",
            "/proc/self/status Mems_allowed_list"
        ),
        "/proc/self/status Mems_allowed_list"
    );
}

std::vector<int> read_possible_numa_nodes() {
    return parse_numa_node_list(
        read_first_prefixed_line(
            "/sys/devices/system/node/possible", "",
            "/sys/devices/system/node/possible"
        ),
        "/sys/devices/system/node/possible"
    );
}

struct NativeNumaPolicy {
    int mode = -1;
    std::vector<int> nodes;
};

size_t numa_mask_word_count(unsigned long maxnode) {
    constexpr size_t bits_per_word = sizeof(unsigned long) * 8U;
    return std::max<size_t>(
        1, (static_cast<size_t>(maxnode) + bits_per_word - 1) / bits_per_word
    );
}

std::vector<unsigned long> make_numa_mask(
    const std::vector<int>& nodes,
    unsigned long maxnode
) {
    constexpr size_t bits_per_word = sizeof(unsigned long) * 8U;
    std::vector<unsigned long> mask(numa_mask_word_count(maxnode), 0UL);
    for (int node : nodes) {
        if (node < 0 || static_cast<unsigned long>(node) >= maxnode) {
            throw std::runtime_error(
                "Authenticated NUMA node exceeds the possible node mask"
            );
        }
        mask[static_cast<size_t>(node) / bits_per_word]
            |= 1UL << (static_cast<size_t>(node) % bits_per_word);
    }
    return mask;
}

NativeNumaPolicy query_numa_policy(
    unsigned long maxnode,
    const void* address = nullptr
) {
#if defined(__linux__) && defined(SYS_get_mempolicy)
    std::vector<unsigned long> mask(numa_mask_word_count(maxnode), 0UL);
    NativeNumaPolicy policy;
    errno = 0;
    const long result = ::syscall(
        SYS_get_mempolicy,
        &policy.mode,
        mask.data(),
        maxnode,
        const_cast<void*>(address),
        address == nullptr ? 0UL : kMpolAddress
    );
    if (result != 0) {
        const int error = errno == 0 ? EIO : errno;
        throw std::runtime_error(
            std::string("get_mempolicy failed for the native integrity snapshot: ")
            + std::strerror(error)
        );
    }
    constexpr size_t bits_per_word = sizeof(unsigned long) * 8U;
    for (unsigned long node = 0; node < maxnode; ++node) {
        if (mask[static_cast<size_t>(node) / bits_per_word]
            & (1UL << (static_cast<size_t>(node) % bits_per_word))) {
            policy.nodes.push_back(static_cast<int>(node));
        }
    }
    return policy;
#else
    (void)maxnode;
    (void)address;
    throw std::runtime_error(
        "Native integrity snapshot NUMA policy queries are unsupported"
    );
#endif
}

void require_exact_static_membind(
    const NativeNumaPolicy& observed,
    const std::vector<int>& expected,
    const char* label
) {
    if (observed.mode != kStaticMembindPolicy || observed.nodes != expected) {
        throw std::runtime_error(
            std::string(label)
            + " differs from the authenticated static NUMA membind"
        );
    }
}

struct AuthenticatedNativeNumaPolicy {
    std::vector<int> nodes;
    unsigned long maxnode = 0;
    std::vector<unsigned long> raw_mask;
};

struct NativeNumaContractRequest {
    bool required = false;
    std::vector<int> nodes;
};

NativeNumaContractRequest native_numa_contract_request() {
    const char* provenance = std::getenv("SUMMIT_NUMA_POLICY_PROVENANCE");
    const char* applied = std::getenv("SUMMIT_NUMA_POLICY_APPLIED");
    // Legacy gw_ldscore modes may publish APPLIED alone for policies that do
    // not carry the pre-import attestation.  Only PROVENANCE opts a process in
    // to this strict snapshot contract; once present, the pair is mandatory.
    if (provenance == nullptr) {
        return {};
    }
    if (applied == nullptr) {
        throw std::runtime_error(
            "Partial native integrity snapshot NUMA policy markers are forbidden"
        );
    }
    if (std::string(provenance) != "pre_numeric_import") {
        throw std::runtime_error(
            "Native integrity snapshot NUMA provenance is not pre_numeric_import"
        );
    }
    constexpr const char* prefix = "libnuma:membind:";
    const std::string applied_policy(applied);
    if (applied_policy.compare(0, std::strlen(prefix), prefix) != 0) {
        throw std::runtime_error(
            "Native integrity snapshot NUMA attestation is not a libnuma membind"
        );
    }
    const std::string encoded_nodes = applied_policy.substr(std::strlen(prefix));
    const std::vector<int> selected = parse_numa_node_list(
        encoded_nodes, "SUMMIT_NUMA_POLICY_APPLIED"
    );
    if (canonical_numa_node_list(selected) != encoded_nodes) {
        throw std::runtime_error(
            "Native integrity snapshot NUMA attestation is not canonical"
        );
    }
    NativeNumaContractRequest result;
    result.required = true;
    result.nodes = selected;
    return result;
}

void require_native_snapshot_request_match(
    const NativeNumaContractRequest& request,
    const NativeIntegritySnapshotNumaEvidence& evidence
) {
    if (evidence.contract_required != request.required) {
        throw std::runtime_error(
            "Native integrity snapshot contract changed before vendor entry"
        );
    }
    if (request.required && evidence.selected_nodes != request.nodes) {
        throw std::runtime_error(
            "Native integrity snapshot NUMA nodes changed before vendor entry"
        );
    }
}

void require_native_output_request_match(
    const NativeNumaContractRequest& expected
) {
    const NativeNumaContractRequest observed = native_numa_contract_request();
    if (observed.required != expected.required) {
        throw std::runtime_error(
            "Native GEMM output NUMA contract changed during the protected call"
        );
    }
    if (expected.required && observed.nodes != expected.nodes) {
        throw std::runtime_error(
            "Native GEMM output NUMA nodes changed during the protected call"
        );
    }
}

AuthenticatedNativeNumaPolicy authenticated_native_numa_policy(
    const NativeNumaContractRequest& request
) {
    if (!request.required || request.nodes.empty()) {
        throw std::runtime_error(
            "Cannot authenticate an absent native integrity snapshot NUMA contract"
        );
    }
    const std::vector<int>& selected = request.nodes;
    const std::vector<int> allowed = read_current_mems_allowed_nodes();
    for (int node : selected) {
        if (!std::binary_search(allowed.begin(), allowed.end(), node)) {
            throw std::runtime_error(
                "Authenticated native integrity snapshot node escaped Mems_allowed_list"
            );
        }
    }
    const std::vector<int> possible = read_possible_numa_nodes();
    const unsigned long maxnode =
        static_cast<unsigned long>(possible.back()) + 1UL;
    for (int node : selected) {
        if (!std::binary_search(possible.begin(), possible.end(), node)) {
            throw std::runtime_error(
                "Authenticated native integrity snapshot node is not possible"
            );
        }
    }
    require_exact_static_membind(
        query_numa_policy(maxnode), selected,
        "Live owner-thread NUMA policy"
    );
    AuthenticatedNativeNumaPolicy result;
    result.nodes = selected;
    result.maxnode = maxnode;
    result.raw_mask = make_numa_mask(selected, maxnode);
    return result;
}

void bind_native_snapshot_range(
    void* address,
    size_t byte_count,
    const AuthenticatedNativeNumaPolicy& policy,
    unsigned long flags,
    const char* label
) {
#if defined(__linux__) && defined(SYS_mbind)
    errno = 0;
    const long result = ::syscall(
        SYS_mbind,
        address,
        static_cast<unsigned long>(byte_count),
        kStaticMembindPolicy,
        policy.raw_mask.data(),
        policy.maxnode,
        flags
    );
    if (result != 0) {
        const int error = errno == 0 ? EIO : errno;
        throw std::runtime_error(
            std::string(label) + ": " + std::strerror(error)
        );
    }
#else
    (void)address;
    (void)byte_count;
    (void)policy;
    (void)flags;
    (void)label;
    throw std::runtime_error(
        "Native integrity snapshot NUMA range binding is unsupported"
    );
#endif
}

class NativeIntegritySnapshotMapping {
public:
    explicit NativeIntegritySnapshotMapping(size_t logical_byte_count) {
#if defined(__linux__)
        const long observed_page_size = ::sysconf(_SC_PAGESIZE);
        if (observed_page_size <= 0) {
            throw std::runtime_error(
                "Could not determine page size for the native integrity snapshot"
            );
        }
        if (logical_byte_count == 0) {
            throw std::runtime_error(
                "Native integrity snapshot byte count must be positive"
            );
        }
        const size_t page_size = static_cast<size_t>(observed_page_size);
        const size_t page_count = (logical_byte_count - 1) / page_size + 1;
        if (page_count > std::numeric_limits<size_t>::max() / page_size) {
            throw std::overflow_error(
                "Native integrity snapshot mapping size overflow"
            );
        }
        const size_t mapping_bytes = page_count * page_size;
        // Explicit benchmark runs publish both immutable early-NUMA markers.
        // Their presence upgrades this allocation to the fail-closed placement
        // contract below.  Legacy callers with both markers absent still get a
        // dedicated, sealed mapping, but never receive locality claims.
        const NativeNumaContractRequest request =
            native_numa_contract_request();
        if (request.required) {
            policy_ = authenticated_native_numa_policy(request);
        }
        mapping_ = ::mmap(
            nullptr,
            mapping_bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0
        );
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            const int error = errno == 0 ? ENOMEM : errno;
            throw std::runtime_error(
                std::string("Native integrity snapshot mmap failed: ")
                + std::strerror(error)
            );
        }
        mapping_bytes_ = mapping_bytes;
        try {
            if (reinterpret_cast<uintptr_t>(mapping_) % page_size != 0) {
                throw std::runtime_error(
                    "Native integrity snapshot mmap is not page aligned"
                );
            }
            if (request.required) {
                bind_native_snapshot_range(
                    mapping_, mapping_bytes_, policy_, 0UL,
                    "Native integrity snapshot pre-touch mbind failed"
                );
                require_exact_static_membind(
                    query_numa_policy(policy_.maxnode, mapping_),
                    policy_.nodes,
                    "Native integrity snapshot pre-touch range policy"
                );
            }
        } catch (...) {
            release();
            throw;
        }
        evidence_.contract_required = request.required;
        evidence_.integrity_check_shape_eligible = true;
        evidence_.integrity_check_executed = true;
        evidence_.snapshot_available = true;
        evidence_.logical_byte_count = logical_byte_count;
        evidence_.mapping_bytes = mapping_bytes_;
        evidence_.page_size = page_size;
        evidence_.mapping_page_count = page_count;
        evidence_.selected_nodes = policy_.nodes;
        evidence_.anonymous_private_mapping = true;
        evidence_.page_aligned_mapping = true;
        evidence_.bound_before_first_touch = request.required;
        evidence_.live_owner_policy_verified = request.required;
        evidence_.pre_touch_range_policy_verified = request.required;
#else
        (void)logical_byte_count;
        throw std::runtime_error(
            "Native integrity snapshot mappings are supported only on Linux"
        );
#endif
    }

    NativeIntegritySnapshotMapping(const NativeIntegritySnapshotMapping&) = delete;
    NativeIntegritySnapshotMapping& operator=(
        const NativeIntegritySnapshotMapping&
    ) = delete;

    ~NativeIntegritySnapshotMapping() noexcept { release(); }

    double* data() noexcept { return static_cast<double*>(mapping_); }

    void verify_before_vendor() {
#if defined(__linux__)
        if (evidence_.contract_required) {
#if defined(SYS_move_pages)
            require_exact_static_membind(
                query_numa_policy(policy_.maxnode), policy_.nodes,
                "Live owner-thread NUMA policy before the vendor call"
            );
            require_exact_static_membind(
                query_numa_policy(policy_.maxnode, mapping_), policy_.nodes,
                "Native integrity snapshot range policy before the vendor call"
            );
            Sha256 status_digest;
            size_t queried_pages = 0;
            size_t query_chunks = 0;
            std::map<int, size_t> histogram;
            while (queried_pages < evidence_.mapping_page_count) {
                const size_t count = std::min(
                    kNativeIntegritySnapshotQueryChunkPages,
                    evidence_.mapping_page_count - queried_pages
                );
                std::vector<void*> pages(count);
                std::vector<int> statuses(count, 0);
                for (size_t offset = 0; offset < count; ++offset) {
                    pages[offset] = static_cast<unsigned char*>(mapping_)
                        + (queried_pages + offset) * evidence_.page_size;
                }
                errno = 0;
                const long result = ::syscall(
                    SYS_move_pages,
                    0,
                    static_cast<unsigned long>(count),
                    pages.data(),
                    nullptr,
                    statuses.data(),
                    0
                );
                if (result != 0) {
                    const int error = errno == 0 ? EIO : errno;
                    throw std::runtime_error(
                        std::string(
                            "Native integrity snapshot exhaustive move_pages query failed: "
                        ) + std::strerror(error)
                    );
                }
                for (int status : statuses) {
                    if (status < 0) {
                        throw std::runtime_error(
                            "Native integrity snapshot page query returned a page error"
                        );
                    }
                    if (!std::binary_search(
                            policy_.nodes.begin(), policy_.nodes.end(), status
                        )) {
                        throw std::runtime_error(
                            "Native integrity snapshot page is outside the authenticated NUMA nodes"
                        );
                    }
                    ++histogram[status];
                    const uint32_t encoded = static_cast<uint32_t>(status);
                    const std::array<unsigned char, 4> little_endian = {
                        static_cast<unsigned char>(encoded & 0xffU),
                        static_cast<unsigned char>((encoded >> 8U) & 0xffU),
                        static_cast<unsigned char>((encoded >> 16U) & 0xffU),
                        static_cast<unsigned char>((encoded >> 24U) & 0xffU),
                    };
                    status_digest.update(
                        little_endian.data(), little_endian.size()
                    );
                }
                queried_pages += count;
                ++query_chunks;
            }
            if (queried_pages != evidence_.mapping_page_count) {
                throw std::runtime_error(
                    "Native integrity snapshot page query count is inconsistent"
                );
            }
            bind_native_snapshot_range(
                mapping_, mapping_bytes_, policy_, kMpolStrict,
                "Native integrity snapshot strict no-move mbind failed"
            );
            evidence_.pre_vendor_complete_page_query = true;
            evidence_.queried_pages = queried_pages;
            evidence_.resolved_pages = queried_pages;
            evidence_.query_chunks = query_chunks;
            evidence_.node_histogram = std::move(histogram);
            evidence_.ordered_status_sha256 = status_digest.finish();
            evidence_.pre_vendor_strict_policy_verified = true;
#else
            throw std::runtime_error(
                "Native integrity snapshot exhaustive page queries are unsupported"
            );
#endif
        }
        errno = 0;
        if (::mprotect(mapping_, mapping_bytes_, PROT_READ) != 0) {
            const int error = errno == 0 ? EACCES : errno;
            throw std::runtime_error(
                std::string(
                    "Native integrity snapshot read-only sealing failed: "
                ) + std::strerror(error)
            );
        }
        evidence_.sealed_read_only_before_vendor = true;
        evidence_.complete = evidence_.contract_required;
#else
        throw std::runtime_error(
            "Native integrity snapshot verification is unsupported"
        );
#endif
    }

    const NativeIntegritySnapshotNumaEvidence& evidence() const noexcept {
        return evidence_;
    }

private:
    void release() noexcept {
#if defined(__linux__)
        if (mapping_ != nullptr) {
            ::munmap(mapping_, mapping_bytes_);
            mapping_ = nullptr;
            mapping_bytes_ = 0;
        }
#endif
    }

    void* mapping_ = nullptr;
    size_t mapping_bytes_ = 0;
    AuthenticatedNativeNumaPolicy policy_;
    NativeIntegritySnapshotNumaEvidence evidence_;
};

class NativeGemmOutputAllocation {
public:
    // The physical storage is sized by ``capacity_byte_count`` (0 means the
    // first call's logical size): one maximum-capacity allocation per scratch
    // role, reused with smaller logical shapes instead of retaining one
    // mapping per exact shape.  The whole capacity is bound and pre-faulted
    // at construction so exhaustive page verification stays meaningful for
    // every later logical shape.
    NativeGemmOutputAllocation(
        size_t logical_rows,
        size_t logical_columns,
        const char* storage_layout,
        size_t logical_byte_count,
        const NativeNumaContractRequest& request,
        std::shared_ptr<SharedNativeGemmOutputNumaEvidence> evidence,
        size_t capacity_byte_count = 0
    ) : request_(request), evidence_(std::move(evidence)) {
        if (logical_rows == 0 || logical_columns == 0
            || logical_byte_count == 0) {
            throw std::runtime_error(
                "Native GEMM output dimensions and byte count must be positive"
            );
        }
        if (capacity_byte_count == 0) {
            capacity_byte_count = logical_byte_count;
        }
        if (capacity_byte_count < logical_byte_count
            || capacity_byte_count % sizeof(double) != 0) {
            throw std::runtime_error(
                "Native GEMM output capacity is smaller than the logical shape"
            );
        }
        if (storage_layout == nullptr
            || (std::strcmp(storage_layout, "column_major") != 0
                && std::strcmp(storage_layout, "row_major") != 0)) {
            throw std::runtime_error(
                "Native GEMM output storage layout is invalid"
            );
        }
        if (logical_rows
                > std::numeric_limits<size_t>::max() / logical_columns) {
            throw std::overflow_error(
                "Native GEMM output logical element count overflow"
            );
        }
        const size_t logical_elements = logical_rows * logical_columns;
        if (logical_elements
                > std::numeric_limits<size_t>::max() / sizeof(double)
            || logical_byte_count != logical_elements * sizeof(double)) {
            throw std::overflow_error(
                "Native GEMM output logical byte count is inconsistent"
            );
        }
        logical_rows_ = logical_rows;
        logical_columns_ = logical_columns;
        storage_layout_ = storage_layout;
        logical_byte_count_ = logical_byte_count;
        capacity_byte_count_ = capacity_byte_count;
        if (evidence_ == nullptr) {
            throw std::runtime_error(
                "Native GEMM output evidence owner is missing"
            );
        }
        evidence_->update([&](NativeGemmOutputNumaEvidenceData& data) {
            data.logical_rows = logical_rows;
            data.logical_columns = logical_columns;
            data.storage_layout = storage_layout;
            data.logical_byte_count = logical_byte_count;
            data.capacity_byte_count = capacity_byte_count;
        });

        if (!request_.required) {
            void* storage = nullptr;
            if (::posix_memalign(&storage, 64, capacity_byte_count) != 0) {
                throw std::bad_alloc();
            }
            std::memset(storage, 0, capacity_byte_count);
            storage_ = storage;
            legacy_allocation_ = true;
            try {
                evidence_->update([](NativeGemmOutputNumaEvidenceData& data) {
                    data.allocation_mode = "legacy_posix_memalign";
                });
            } catch (...) {
                release();
                throw;
            }
            return;
        }

#if defined(__linux__)
        policy_ = authenticated_native_numa_policy(request_);
        const long observed_page_size = ::sysconf(_SC_PAGESIZE);
        if (observed_page_size <= 0) {
            throw std::runtime_error(
                "Could not determine page size for the native GEMM output"
            );
        }
        const size_t page_size = static_cast<size_t>(observed_page_size);
        const size_t page_count = (capacity_byte_count - 1) / page_size + 1;
        if (page_count > std::numeric_limits<size_t>::max() / page_size) {
            throw std::overflow_error(
                "Native GEMM output mapping size overflow"
            );
        }
        const size_t mapping_bytes = page_count * page_size;
        page_size_ = page_size;
        mapping_page_count_ = page_count;
        storage_ = ::mmap(
            nullptr,
            mapping_bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0
        );
        if (storage_ == MAP_FAILED) {
            storage_ = nullptr;
            const int error = errno == 0 ? ENOMEM : errno;
            throw std::runtime_error(
                std::string("Native GEMM output mmap failed: ")
                + std::strerror(error)
            );
        }
        mapping_bytes_ = mapping_bytes;
        try {
            if (reinterpret_cast<uintptr_t>(storage_) % page_size != 0) {
                throw std::runtime_error(
                    "Native GEMM output mmap is not page aligned"
                );
            }
            require_exact_static_membind(
                query_numa_policy(policy_.maxnode), policy_.nodes,
                "Native GEMM output pre-touch live owner policy"
            );
            bind_native_snapshot_range(
                storage_, mapping_bytes_, policy_, 0UL,
                "Native GEMM output pre-touch mbind failed"
            );
            require_exact_static_membind(
                query_numa_policy(policy_.maxnode, storage_), policy_.nodes,
                "Native GEMM output pre-touch range policy"
            );
            // Fault every capacity page while it is already bound, so later
            // calls with smaller logical shapes never expose unmapped pages
            // to the exhaustive post-call page verification.
            std::memset(storage_, 0, mapping_bytes_);
            evidence_->update([&](NativeGemmOutputNumaEvidenceData& data) {
                data.allocation_mode = "mmap_private_anonymous";
                data.mapping_bytes = mapping_bytes_;
                data.page_size = page_size;
                data.mapping_page_count = page_count;
                data.selected_nodes = policy_.nodes;
                data.anonymous_private_mapping = true;
                data.page_aligned_mapping = true;
                data.writable_output = true;
                data.bound_before_first_touch = true;
                data.pre_touch_live_owner_policy_verified = true;
                data.pre_touch_range_policy_verified = true;
            });
        } catch (...) {
            release();
            throw;
        }
#else
        throw std::runtime_error(
            "Native GEMM output NUMA mappings are supported only on Linux"
        );
#endif
    }

    NativeGemmOutputAllocation(const NativeGemmOutputAllocation&) = delete;
    NativeGemmOutputAllocation& operator=(
        const NativeGemmOutputAllocation&
    ) = delete;

    ~NativeGemmOutputAllocation() noexcept { release(); }

    double* data() noexcept { return static_cast<double*>(storage_); }
    const double* data() const noexcept {
        return static_cast<const double*>(storage_);
    }

    size_t logical_byte_count() const noexcept { return logical_byte_count_; }

    void seal_read_only() {
        require_native_output_request_match(request_);
        if (!request_.required) {
            throw std::runtime_error(
                "Legacy native output allocation cannot be sealed in place"
            );
        }
#if defined(__linux__)
        if (storage_ == nullptr || mapping_bytes_ == 0
            || ::mprotect(storage_, mapping_bytes_, PROT_READ) != 0) {
            throw std::runtime_error(
                "Could not seal the NUMA-bound native mapping read-only"
            );
        }
        evidence_->update([](NativeGemmOutputNumaEvidenceData& data) {
            data.writable_output = false;
        });
#else
        throw std::runtime_error(
            "NUMA-bound native mapping seals require Linux"
        );
#endif
    }

    void reuse_for_call(
        size_t logical_rows,
        size_t logical_columns,
        const char* storage_layout,
        size_t logical_byte_count,
        const NativeNumaContractRequest& request,
        std::shared_ptr<SharedNativeGemmOutputNumaEvidence> evidence
    ) {
        if (logical_rows == 0 || logical_columns == 0
            || storage_layout == nullptr
            || storage_layout_ != storage_layout
            || logical_rows
                > std::numeric_limits<size_t>::max() / logical_columns
            || logical_rows * logical_columns
                > std::numeric_limits<size_t>::max() / sizeof(double)
            || logical_byte_count
                != logical_rows * logical_columns * sizeof(double)
            || logical_byte_count > capacity_byte_count_) {
            throw std::runtime_error(
                "Reusable native GEMM output logical shape exceeds its "
                "allocated capacity or changed layout"
            );
        }
        logical_rows_ = logical_rows;
        logical_columns_ = logical_columns;
        logical_byte_count_ = logical_byte_count;
        if (request.required != request_.required
            || request.nodes != request_.nodes) {
            throw std::runtime_error(
                "Reusable native GEMM output NUMA contract changed"
            );
        }
        if (evidence == nullptr) {
            throw std::runtime_error(
                "Reusable native GEMM output evidence owner is missing"
            );
        }
        require_native_output_request_match(request_);
        evidence_ = std::move(evidence);
        evidence_->update([&](NativeGemmOutputNumaEvidenceData& data) {
            data.logical_rows = logical_rows_;
            data.logical_columns = logical_columns_;
            data.storage_layout = storage_layout_;
            data.logical_byte_count = logical_byte_count_;
            data.capacity_byte_count = capacity_byte_count_;
            data.allocation_mode = legacy_allocation_
                ? "legacy_posix_memalign" : "mmap_private_anonymous";
        });
        if (!request_.required) return;
#if defined(__linux__)
        require_exact_static_membind(
            query_numa_policy(policy_.maxnode), policy_.nodes,
            "Reusable native GEMM output pre-call live owner policy"
        );
        require_exact_static_membind(
            query_numa_policy(policy_.maxnode, storage_), policy_.nodes,
            "Reusable native GEMM output pre-call range policy"
        );
        evidence_->update([&](NativeGemmOutputNumaEvidenceData& data) {
            data.mapping_bytes = mapping_bytes_;
            data.page_size = page_size_;
            data.mapping_page_count = mapping_page_count_;
            data.selected_nodes = policy_.nodes;
            data.anonymous_private_mapping = true;
            data.page_aligned_mapping = true;
            data.writable_output = true;
            data.bound_before_first_touch = true;
            data.pre_touch_live_owner_policy_verified = true;
            data.pre_touch_range_policy_verified = true;
        });
#else
        throw std::runtime_error(
            "Reusable native GEMM output NUMA mappings require Linux"
        );
#endif
    }

    void verify_after_repair() {
        require_native_output_request_match(request_);
        if (!request_.required) return;
#if defined(__linux__) && defined(SYS_move_pages)
        require_exact_static_membind(
            query_numa_policy(policy_.maxnode), policy_.nodes,
            "Native GEMM output post-repair live owner policy"
        );
        require_exact_static_membind(
            query_numa_policy(policy_.maxnode, storage_), policy_.nodes,
            "Native GEMM output post-repair range policy"
        );
        const NativeGemmOutputNumaEvidenceData initial = evidence_->snapshot();
        Sha256 status_digest;
        size_t queried_pages = 0;
        size_t query_chunks = 0;
        std::map<int, size_t> histogram;
        while (queried_pages < initial.mapping_page_count) {
            const size_t count = std::min(
                kNativeGemmOutputQueryChunkPages,
                initial.mapping_page_count - queried_pages
            );
            std::vector<void*> pages(count);
            std::vector<int> statuses(count, 0);
            for (size_t offset = 0; offset < count; ++offset) {
                pages[offset] = static_cast<unsigned char*>(storage_)
                    + (queried_pages + offset) * initial.page_size;
            }
            errno = 0;
            const long result = ::syscall(
                SYS_move_pages,
                0,
                static_cast<unsigned long>(count),
                pages.data(),
                nullptr,
                statuses.data(),
                0
            );
            if (result != 0) {
                const int error = errno == 0 ? EIO : errno;
                throw std::runtime_error(
                    std::string(
                        "Native GEMM output exhaustive move_pages query failed: "
                    ) + std::strerror(error)
                );
            }
            for (int status : statuses) {
                if (status < 0) {
                    throw std::runtime_error(
                        "Native GEMM output page query returned a page error"
                    );
                }
                if (!std::binary_search(
                        policy_.nodes.begin(), policy_.nodes.end(), status
                    )) {
                    throw std::runtime_error(
                        "Native GEMM output page is outside the authenticated NUMA nodes"
                    );
                }
                ++histogram[status];
                const uint32_t encoded = static_cast<uint32_t>(status);
                const std::array<unsigned char, 4> little_endian = {
                    static_cast<unsigned char>(encoded & 0xffU),
                    static_cast<unsigned char>((encoded >> 8U) & 0xffU),
                    static_cast<unsigned char>((encoded >> 16U) & 0xffU),
                    static_cast<unsigned char>((encoded >> 24U) & 0xffU),
                };
                status_digest.update(
                    little_endian.data(), little_endian.size()
                );
            }
            queried_pages += count;
            ++query_chunks;
        }
        if (queried_pages != initial.mapping_page_count) {
            throw std::runtime_error(
                "Native GEMM output page query count is inconsistent"
            );
        }
        bind_native_snapshot_range(
            storage_, mapping_bytes_, policy_, kMpolStrict,
            "Native GEMM output strict no-move mbind failed"
        );
        evidence_->update([&](NativeGemmOutputNumaEvidenceData& data) {
            data.post_repair_live_owner_policy_verified = true;
            data.post_repair_range_policy_verified = true;
            data.post_repair_complete_page_query = true;
            data.queried_pages = queried_pages;
            data.resolved_pages = queried_pages;
            data.query_chunks = query_chunks;
            data.node_histogram = std::move(histogram);
            data.ordered_status_sha256 = status_digest.finish();
            data.post_repair_strict_policy_verified = true;
            data.complete = true;
        });
#else
        throw std::runtime_error(
            "Native GEMM output exhaustive page queries are unsupported"
        );
#endif
    }

private:
    void release() noexcept {
        if (storage_ == nullptr) return;
        if (legacy_allocation_) {
            std::free(storage_);
        } else {
#if defined(__linux__)
            ::munmap(storage_, mapping_bytes_);
#endif
        }
        storage_ = nullptr;
        mapping_bytes_ = 0;
    }

    NativeNumaContractRequest request_;
    std::shared_ptr<SharedNativeGemmOutputNumaEvidence> evidence_;
    AuthenticatedNativeNumaPolicy policy_;
    void* storage_ = nullptr;
    size_t mapping_bytes_ = 0;
    size_t logical_rows_ = 0;
    size_t logical_columns_ = 0;
    std::string storage_layout_;
    size_t logical_byte_count_ = 0;
    size_t capacity_byte_count_ = 0;
    size_t page_size_ = 0;
    size_t mapping_page_count_ = 0;
    bool legacy_allocation_ = false;

public:
    size_t capacity_byte_count() const noexcept {
        return capacity_byte_count_;
    }
};

nb::dict native_integrity_snapshot_numa_to_dict(
    const NativeIntegritySnapshotNumaEvidence& evidence
) {
    nb::dict result;
    result["schema"] = kNativeIntegritySnapshotNumaSchema;
    result["schema_version"] = 1;
    result["operand_role"] = kNativeIntegritySnapshotOperandRole;
    result["integrity_check_shape_eligible"] =
        evidence.integrity_check_shape_eligible;
    result["integrity_check_executed"] = evidence.integrity_check_executed;
    result["snapshot_available"] = evidence.snapshot_available;
    result["contract_required"] = evidence.contract_required;
    result["complete"] = evidence.complete;
    if (!evidence.snapshot_available) return result;
    if (!evidence.contract_required) {
        result["sealed_read_only_before_vendor"] =
            evidence.sealed_read_only_before_vendor;
        return result;
    }

    result["logical_byte_count"] = evidence.logical_byte_count;
    result["mapping_bytes"] = evidence.mapping_bytes;
    result["page_size"] = evidence.page_size;
    result["mapping_page_count"] = evidence.mapping_page_count;
    nb::list selected_nodes;
    for (int node : evidence.selected_nodes) selected_nodes.append(node);
    result["selected_nodes"] = std::move(selected_nodes);
    result["policy_mode"] = "bind_static_nodes";
    result["policy_mode_value"] = kStaticMembindPolicy;
    result["anonymous_private_mapping"] = evidence.anonymous_private_mapping;
    result["page_aligned_mapping"] = evidence.page_aligned_mapping;
    result["bound_before_first_touch"] = evidence.bound_before_first_touch;
    result["live_owner_policy_verified"] =
        evidence.live_owner_policy_verified;
    result["pre_touch_range_policy_verified"] =
        evidence.pre_touch_range_policy_verified;
    result["pre_vendor_complete_page_query"] =
        evidence.pre_vendor_complete_page_query;
    result["queried_pages"] = evidence.queried_pages;
    result["resolved_pages"] = evidence.resolved_pages;
    result["query_chunks"] = evidence.query_chunks;
    result["query_chunk_page_limit"] =
        kNativeIntegritySnapshotQueryChunkPages;
    nb::dict node_histogram;
    for (const auto& item : evidence.node_histogram) {
        const std::string key = std::to_string(item.first);
        node_histogram[key.c_str()] = item.second;
    }
    result["node_histogram"] = std::move(node_histogram);
    result["ordered_status_sha256"] = evidence.ordered_status_sha256;
    result["ordered_status_encoding"] = "signed_int32_little_endian";
    result["pre_vendor_strict_policy_verified"] =
        evidence.pre_vendor_strict_policy_verified;
    result["sealed_read_only_before_vendor"] =
        evidence.sealed_read_only_before_vendor;
    result["strict_policy_check"] =
        "MPOL_MF_STRICT_without_MPOL_MF_MOVE";
    result["page_query_method"] = "move_pages_query_no_migration";
    result["page_migration_requested"] = false;
    result["placement_repair_performed"] = false;
    return result;
}

nb::dict test_native_integrity_snapshot_numa(size_t logical_byte_count) {
    constexpr size_t kTestMaximumBytes = 16U * 1024U * 1024U;
    if (logical_byte_count == 0 || logical_byte_count > kTestMaximumBytes) {
        throw std::runtime_error(
            "Native integrity snapshot test byte count must be in [1,16777216]"
        );
    }
    NativeIntegritySnapshotMapping snapshot(logical_byte_count);
    std::memset(snapshot.data(), 0xa5, logical_byte_count);
    snapshot.verify_before_vendor();
    return native_integrity_snapshot_numa_to_dict(snapshot.evidence());
}

bool test_native_integrity_snapshot_request_match(int evidence_node) {
    if (evidence_node < 0) {
        throw std::runtime_error(
            "Native integrity snapshot request-match test node must be nonnegative"
        );
    }
    NativeIntegritySnapshotNumaEvidence evidence;
    evidence.contract_required = true;
    evidence.selected_nodes = {evidence_node};
    require_native_snapshot_request_match(
        native_numa_contract_request(), evidence
    );
    return true;
}

nb::dict native_gemm_output_numa_to_dict(
    const NativeGemmOutputNumaEvidenceData& evidence
) {
    nb::dict result;
    result["schema"] = kNativeGemmOutputNumaSchema;
    // schema_version 2 adds capacity_byte_count: outputs may reuse one
    // maximum-capacity mapping, so mapping_bytes tracks the capacity rather
    // than the per-call logical shape.
    result["schema_version"] = 2;
    result["applicable"] = evidence.applicable;
    result["operand_role"] = kNativeGemmOutputOperandRole;
    result["contract_required"] = evidence.contract_required;
    result["complete"] = evidence.complete;
    if (!evidence.applicable) return result;

    result["call_id"] = evidence.call_id;
    result["logical_rows"] = evidence.logical_rows;
    result["logical_columns"] = evidence.logical_columns;
    result["storage_layout"] = evidence.storage_layout;
    result["logical_byte_count"] = evidence.logical_byte_count;
    result["capacity_byte_count"] = evidence.capacity_byte_count;
    result["allocation_mode"] = evidence.allocation_mode;
    if (!evidence.contract_required) return result;

    result["mapping_bytes"] = evidence.mapping_bytes;
    result["page_size"] = evidence.page_size;
    result["mapping_page_count"] = evidence.mapping_page_count;
    nb::list selected_nodes;
    for (int node : evidence.selected_nodes) selected_nodes.append(node);
    result["selected_nodes"] = std::move(selected_nodes);
    result["policy_mode"] = "bind_static_nodes";
    result["policy_mode_value"] = kStaticMembindPolicy;
    result["anonymous_private_mapping"] =
        evidence.anonymous_private_mapping;
    result["page_aligned_mapping"] = evidence.page_aligned_mapping;
    result["writable_output"] = evidence.writable_output;
    result["bound_before_first_touch"] = evidence.bound_before_first_touch;
    result["pre_touch_live_owner_policy_verified"] =
        evidence.pre_touch_live_owner_policy_verified;
    result["pre_touch_range_policy_verified"] =
        evidence.pre_touch_range_policy_verified;
    result["post_repair_live_owner_policy_verified"] =
        evidence.post_repair_live_owner_policy_verified;
    result["post_repair_range_policy_verified"] =
        evidence.post_repair_range_policy_verified;
    result["post_repair_complete_page_query"] =
        evidence.post_repair_complete_page_query;
    result["queried_pages"] = evidence.queried_pages;
    result["resolved_pages"] = evidence.resolved_pages;
    result["query_chunks"] = evidence.query_chunks;
    result["query_chunk_page_limit"] = kNativeGemmOutputQueryChunkPages;
    nb::dict node_histogram;
    for (const auto& item : evidence.node_histogram) {
        const std::string key = std::to_string(item.first);
        node_histogram[key.c_str()] = item.second;
    }
    result["node_histogram"] = std::move(node_histogram);
    result["ordered_status_sha256"] = evidence.ordered_status_sha256;
    result["ordered_status_encoding"] = "signed_int32_little_endian";
    result["post_repair_strict_policy_verified"] =
        evidence.post_repair_strict_policy_verified;
    result["verification_boundary"] =
        "after_partitioned_or_integrity_repair_before_python_return";
    result["strict_policy_check"] =
        "MPOL_MF_STRICT_without_MPOL_MF_MOVE";
    result["page_query_method"] = "move_pages_query_no_migration";
    result["page_migration_requested"] = false;
    result["placement_repair_performed"] = false;
    result["sealed_read_only"] = false;
    return result;
}

nb::dict native_gemm_output_numa_to_dict(
    const std::shared_ptr<SharedNativeGemmOutputNumaEvidence>& evidence
) {
    if (evidence == nullptr) {
        return native_gemm_output_numa_to_dict(
            NativeGemmOutputNumaEvidenceData{}
        );
    }
    return native_gemm_output_numa_to_dict(evidence->snapshot());
}

std::shared_ptr<SharedNativeGemmOutputNumaEvidence>
make_nonapplicable_native_gemm_output_numa_evidence(
    bool contract_required
) {
    NativeGemmOutputNumaEvidenceData data;
    data.contract_required = contract_required;
    return std::make_shared<SharedNativeGemmOutputNumaEvidence>(
        std::move(data)
    );
}

struct NativeGemmOutputNumaEvidenceBuffer {
    std::mutex mutex;
    std::deque<std::shared_ptr<SharedNativeGemmOutputNumaEvidence>> records;
    uint64_t next_call_id = 1;
    uint64_t attempted_calls = 0;
    uint64_t verified_calls = 0;
    uint64_t legacy_calls = 0;
    uint64_t failed_calls = 0;
    uint64_t captured_records = 0;
    uint64_t dropped_records = 0;
};

NativeGemmOutputNumaEvidenceBuffer& native_gemm_output_numa_evidence_buffer() {
    static NativeGemmOutputNumaEvidenceBuffer buffer;
    return buffer;
}

uint64_t reserve_native_gemm_output_numa_call() {
    auto& buffer = native_gemm_output_numa_evidence_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    const uint64_t call_id = buffer.next_call_id++;
    if (call_id == 0 || buffer.next_call_id == 0) {
        throw std::runtime_error(
            "Native GEMM output NUMA call ID space is exhausted"
        );
    }
    ++buffer.attempted_calls;
    return call_id;
}

void record_native_gemm_output_numa_failure() noexcept {
    try {
        auto& buffer = native_gemm_output_numa_evidence_buffer();
        const std::lock_guard<std::mutex> lock(buffer.mutex);
        ++buffer.failed_calls;
    } catch (...) {
    }
}

void publish_native_gemm_output_numa_evidence(
    const std::shared_ptr<SharedNativeGemmOutputNumaEvidence>& evidence
) {
    if (evidence == nullptr) {
        throw std::runtime_error(
            "Cannot publish missing native GEMM output NUMA evidence"
        );
    }
    const NativeGemmOutputNumaEvidenceData snapshot = evidence->snapshot();
    if (!snapshot.applicable || snapshot.call_id == 0) {
        throw std::runtime_error(
            "Native GEMM output NUMA evidence is not an applicable call"
        );
    }
    if (snapshot.contract_required && !snapshot.complete) {
        throw std::runtime_error(
            "Contracted native GEMM output NUMA evidence is incomplete"
        );
    }
    if (!snapshot.contract_required && snapshot.complete) {
        throw std::runtime_error(
            "Legacy native GEMM output cannot claim NUMA completion"
        );
    }
    auto& buffer = native_gemm_output_numa_evidence_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    if (snapshot.contract_required) {
        ++buffer.verified_calls;
    } else {
        ++buffer.legacy_calls;
    }
    ++buffer.captured_records;
    if (buffer.records.size() == kNativeGemmOutputEvidenceCapacity) {
        buffer.records.pop_front();
        ++buffer.dropped_records;
    }
    buffer.records.emplace_back(evidence);
}

void reset_native_gemm_output_numa_evidence() {
    auto& buffer = native_gemm_output_numa_evidence_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    buffer.records.clear();
    buffer.attempted_calls = 0;
    buffer.verified_calls = 0;
    buffer.legacy_calls = 0;
    buffer.failed_calls = 0;
    buffer.captured_records = 0;
    buffer.dropped_records = 0;
}

nb::list native_gemm_output_numa_records_to_list(
    const std::deque<std::shared_ptr<SharedNativeGemmOutputNumaEvidence>>& records
) {
    nb::list result;
    for (const auto& evidence : records) {
        result.append(native_gemm_output_numa_to_dict(evidence));
    }
    return result;
}

nb::list consume_native_gemm_output_numa_evidence() {
    std::deque<std::shared_ptr<SharedNativeGemmOutputNumaEvidence>> consumed;
    {
        auto& buffer = native_gemm_output_numa_evidence_buffer();
        const std::lock_guard<std::mutex> lock(buffer.mutex);
        consumed.swap(buffer.records);
    }
    return native_gemm_output_numa_records_to_list(consumed);
}

nb::list get_native_gemm_output_numa_evidence() {
    std::deque<std::shared_ptr<SharedNativeGemmOutputNumaEvidence>> snapshot;
    {
        auto& buffer = native_gemm_output_numa_evidence_buffer();
        const std::lock_guard<std::mutex> lock(buffer.mutex);
        snapshot = buffer.records;
    }
    return native_gemm_output_numa_records_to_list(snapshot);
}

nb::dict native_gemm_output_numa_evidence_status() {
    auto& buffer = native_gemm_output_numa_evidence_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    nb::dict result;
    result["schema_version"] = 1;
    result["capacity"] = kNativeGemmOutputEvidenceCapacity;
    result["buffered_records"] = buffer.records.size();
    result["captured_records"] = buffer.captured_records;
    result["dropped_records"] = buffer.dropped_records;
    result["next_call_id"] = buffer.next_call_id;
    result["attempted_calls"] = buffer.attempted_calls;
    result["verified_calls"] = buffer.verified_calls;
    result["legacy_calls"] = buffer.legacy_calls;
    result["failed_calls"] = buffer.failed_calls;
    result["query_chunk_page_limit"] = kNativeGemmOutputQueryChunkPages;
    return result;
}

struct GemmTelemetryRecord {
    uint64_t sequence = 0;
    std::string operation;
    std::string layout;
    std::string transpose_a;
    std::string transpose_b;
    int m = 0;
    int n = 0;
    int k = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    double alpha = 1.0;
    double beta = 0.0;
    double flop_count = 0.0;
    double wall_seconds = 0.0;
    double process_cpu_seconds = 0.0;
    int requested_threads = 1;
    int configured_threads = 1;
    int backend_threads = -1;
    OpenMPEntryState omp;
    CpuAffinityEvidence affinity;
    int exit_cpu = -1;
    std::string backend;
    std::string backend_corename;
    std::string backend_config;
    GemmNumaPageSamples numa_page_samples;
    NativeIntegritySnapshotNumaEvidence native_integrity_snapshot_numa;
    std::shared_ptr<SharedNativeGemmOutputNumaEvidence>
        native_gemm_output_numa;
};

struct GemmTelemetryBuffer {
    std::mutex mutex;
    std::deque<GemmTelemetryRecord> records;
    uint64_t next_sequence = 1;
    uint64_t captured_records = 0;
    uint64_t dropped_records = 0;
};

GemmTelemetryBuffer& gemm_telemetry_buffer() {
    static GemmTelemetryBuffer buffer;
    return buffer;
}

uint64_t reserve_gemm_telemetry_sequence() {
    auto& buffer = gemm_telemetry_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    const uint64_t sequence = buffer.next_sequence++;
    if (sequence == 0 || buffer.next_sequence == 0) {
        throw std::runtime_error("GxE GEMM telemetry sequence space is exhausted");
    }
    return sequence;
}

thread_local std::deque<GemmTelemetryRecord>*
    pending_native_gemm_output_telemetry = nullptr;
thread_local std::shared_ptr<SharedNativeGemmOutputNumaEvidence>
    active_native_gemm_output_evidence;
thread_local const double* active_native_gemm_output_begin = nullptr;
thread_local size_t active_native_gemm_output_byte_count = 0;
thread_local CBLAS_LAYOUT active_native_gemm_output_layout = CblasColMajor;

void record_gemm_telemetry_immediate(GemmTelemetryRecord&& record) {
    auto& buffer = gemm_telemetry_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    ++buffer.captured_records;
    if (buffer.records.size() == kGemmTelemetryCapacity) {
        buffer.records.pop_front();
        ++buffer.dropped_records;
    }
    buffer.records.emplace_back(std::move(record));
}

void record_gemm_telemetry(GemmTelemetryRecord&& record) {
    if (pending_native_gemm_output_telemetry != nullptr) {
        if (active_native_gemm_output_evidence == nullptr) {
            throw std::runtime_error(
                "Native GEMM output telemetry scope lost its shared evidence"
            );
        }
        record.native_gemm_output_numa =
            active_native_gemm_output_evidence;
        pending_native_gemm_output_telemetry->emplace_back(std::move(record));
        return;
    }
    record_gemm_telemetry_immediate(std::move(record));
}

class NativeGemmOutputTelemetryScope {
public:
    NativeGemmOutputTelemetryScope() {
        call_id_ = reserve_native_gemm_output_numa_call();
        try {
            request_ = native_numa_contract_request();
            if (pending_native_gemm_output_telemetry != nullptr
                || active_native_gemm_output_evidence != nullptr) {
                throw std::runtime_error(
                    "Nested native GEMM output telemetry scopes are forbidden"
                );
            }
            NativeGemmOutputNumaEvidenceData initial;
            initial.applicable = true;
            initial.call_id = call_id_;
            initial.contract_required = request_.required;
            evidence_ =
                std::make_shared<SharedNativeGemmOutputNumaEvidence>(
                    std::move(initial)
                );
            pending_native_gemm_output_telemetry = &pending_records_;
            active_native_gemm_output_evidence = evidence_;
            active_ = true;
        } catch (...) {
            record_native_gemm_output_numa_failure();
            throw;
        }
    }

    NativeGemmOutputTelemetryScope(
        const NativeGemmOutputTelemetryScope&
    ) = delete;
    NativeGemmOutputTelemetryScope& operator=(
        const NativeGemmOutputTelemetryScope&
    ) = delete;

    ~NativeGemmOutputTelemetryScope() noexcept {
        if (!active_) return;
        if (pending_native_gemm_output_telemetry == &pending_records_) {
            pending_native_gemm_output_telemetry = nullptr;
        }
        if (active_native_gemm_output_evidence == evidence_) {
            active_native_gemm_output_evidence.reset();
        }
        active_native_gemm_output_begin = nullptr;
        active_native_gemm_output_byte_count = 0;
        if (!completed_) record_native_gemm_output_numa_failure();
    }

    const NativeNumaContractRequest& request() const noexcept {
        return request_;
    }

    const std::shared_ptr<SharedNativeGemmOutputNumaEvidence>&
    evidence() const noexcept {
        return evidence_;
    }

    void register_output(
        const double* begin,
        size_t byte_count,
        CBLAS_LAYOUT layout
    ) {
        if (!active_ || completed_ || begin == nullptr || byte_count == 0) {
            throw std::runtime_error(
                "Native GEMM output span registration is invalid"
            );
        }
        if (active_native_gemm_output_begin != nullptr
            || active_native_gemm_output_byte_count != 0) {
            throw std::runtime_error(
                "Native GEMM output span was registered more than once"
            );
        }
        active_native_gemm_output_begin = begin;
        active_native_gemm_output_byte_count = byte_count;
        active_native_gemm_output_layout = layout;
    }

    void complete() {
        if (!active_ || completed_) {
            throw std::runtime_error(
                "Native GEMM output telemetry scope completion is invalid"
            );
        }
        if (active_native_gemm_output_begin == nullptr
            || active_native_gemm_output_byte_count == 0) {
            throw std::runtime_error(
                "Native GEMM output span was not registered"
            );
        }
        require_native_output_request_match(request_);
        publish_native_gemm_output_numa_evidence(evidence_);
        pending_native_gemm_output_telemetry = nullptr;
        active_native_gemm_output_evidence.reset();
        active_native_gemm_output_begin = nullptr;
        active_native_gemm_output_byte_count = 0;
        for (auto& record : pending_records_) {
            record_gemm_telemetry_immediate(std::move(record));
        }
        pending_records_.clear();
        completed_ = true;
        active_ = false;
    }

private:
    uint64_t call_id_ = 0;
    NativeNumaContractRequest request_;
    std::shared_ptr<SharedNativeGemmOutputNumaEvidence> evidence_;
    std::deque<GemmTelemetryRecord> pending_records_;
    bool active_ = false;
    bool completed_ = false;
};

void validate_active_native_gemm_output_vendor_span(
    CBLAS_LAYOUT layout,
    int m,
    int n,
    const double* c,
    int ldc
) {
    if (pending_native_gemm_output_telemetry == nullptr) return;
    if (active_native_gemm_output_begin == nullptr
        || active_native_gemm_output_byte_count == 0 || c == nullptr) {
        throw std::runtime_error(
            "Vendor GEMM output is not backed by the active native output span"
        );
    }
    if (layout != active_native_gemm_output_layout) {
        throw std::runtime_error(
            "Vendor GEMM output layout differs from the active native output"
        );
    }
    if (m <= 0 || n <= 0 || ldc <= 0) {
        throw std::runtime_error(
            "Vendor GEMM output dimensions are invalid for the active output"
        );
    }
    size_t span_elements = 0;
    const bool column_major = layout == CblasColMajor;
    if (!gemm_storage_span_elements(
            static_cast<size_t>(column_major ? n : m),
            static_cast<size_t>(ldc),
            static_cast<size_t>(column_major ? m : n),
            span_elements
        )
        || span_elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::runtime_error(
            "Vendor GEMM output span overflows the active native output"
        );
    }
    const size_t span_bytes = span_elements * sizeof(double);
    const uintptr_t active_begin = reinterpret_cast<uintptr_t>(
        active_native_gemm_output_begin
    );
    if (active_native_gemm_output_byte_count
        > std::numeric_limits<uintptr_t>::max() - active_begin) {
        throw std::runtime_error(
            "Active native GEMM output address range overflows"
        );
    }
    const uintptr_t active_end =
        active_begin + active_native_gemm_output_byte_count;
    const uintptr_t vendor_begin = reinterpret_cast<uintptr_t>(c);
    if (vendor_begin != active_begin
        || span_bytes != active_native_gemm_output_byte_count
        || vendor_begin > active_end
        || span_bytes > active_end - vendor_begin) {
        throw std::runtime_error(
            "Vendor GEMM output span is not the exact active native output"
        );
    }
}

nb::dict gemm_telemetry_record_to_dict(const GemmTelemetryRecord& record) {
    const double gflops_per_second = record.wall_seconds > 0.0
        ? record.flop_count / (record.wall_seconds * 1.0e9) : 0.0;
    const double process_cpu_to_wall_ratio = record.wall_seconds > 0.0
        ? record.process_cpu_seconds / record.wall_seconds : 0.0;
    nb::dict result;
    result["schema_version"] = 1;
    result["sequence"] = record.sequence;
    result["operation"] = record.operation;
    result["arithmetic_dtype"] = "float64";
    result["layout"] = record.layout;
    result["transpose_a"] = record.transpose_a;
    result["transpose_b"] = record.transpose_b;
    result["m"] = record.m;
    result["n"] = record.n;
    result["k"] = record.k;
    result["lda"] = record.lda;
    result["ldb"] = record.ldb;
    result["ldc"] = record.ldc;
    result["alpha"] = record.alpha;
    result["beta"] = record.beta;
    result["flop_count"] = record.flop_count;
    result["wall_seconds"] = record.wall_seconds;
    result["process_cpu_seconds"] = record.process_cpu_seconds;
    result["gflops_per_second"] = gflops_per_second;
    result["process_cpu_to_wall_ratio"] = process_cpu_to_wall_ratio;
    result["active_core_equivalents"] = process_cpu_to_wall_ratio;
    result["requested_threads"] = record.requested_threads;
    result["configured_threads"] = record.configured_threads;
    result["backend_threads"] = record.backend_threads;
    result["omp_in_parallel"] = record.omp.in_parallel;
    result["omp_level"] = record.omp.level;
    result["omp_active_level"] = record.omp.active_level;
    result["omp_max_active_levels"] = record.omp.max_active_levels;
    result["omp_max_threads"] = record.omp.max_threads;
    result["omp_num_threads"] = record.omp.num_threads;
    result["omp_thread_num"] = record.omp.thread_num;
    result["entry_cpu"] = record.affinity.current_cpu;
    result["exit_cpu"] = record.exit_cpu;
    result["cpu_affinity_count"] = record.affinity.cpu_count;
    result["cpu_affinity_list"] = record.affinity.cpu_list;
    result["backend"] = record.backend;
    result["backend_corename"] = record.backend_corename;
    result["backend_config"] = record.backend_config;
    result["operand_numa_page_samples"] = gemm_numa_page_samples_to_dict(
        record.numa_page_samples
    );
    result["native_integrity_snapshot_numa"] =
        native_integrity_snapshot_numa_to_dict(
            record.native_integrity_snapshot_numa
        );
    result["native_gemm_output_numa"] =
        native_gemm_output_numa_to_dict(record.native_gemm_output_numa);
    result["completed"] = true;
    return result;
}

nb::list gemm_telemetry_records_to_list(
    const std::deque<GemmTelemetryRecord>& records
) {
    nb::list result;
    for (const auto& record : records) {
        result.append(gemm_telemetry_record_to_dict(record));
    }
    return result;
}

void reset_gemm_telemetry() {
    {
        auto& buffer = gemm_telemetry_buffer();
        const std::lock_guard<std::mutex> lock(buffer.mutex);
        buffer.records.clear();
        buffer.captured_records = 0;
        buffer.dropped_records = 0;
    }
    reset_native_gemm_output_numa_evidence();
}

nb::list consume_gemm_telemetry() {
    std::deque<GemmTelemetryRecord> consumed;
    {
        auto& buffer = gemm_telemetry_buffer();
        const std::lock_guard<std::mutex> lock(buffer.mutex);
        consumed.swap(buffer.records);
    }
    return gemm_telemetry_records_to_list(consumed);
}

nb::list get_gemm_telemetry() {
    std::deque<GemmTelemetryRecord> snapshot;
    {
        auto& buffer = gemm_telemetry_buffer();
        const std::lock_guard<std::mutex> lock(buffer.mutex);
        snapshot = buffer.records;
    }
    return gemm_telemetry_records_to_list(snapshot);
}

nb::dict gemm_telemetry_status() {
    auto& buffer = gemm_telemetry_buffer();
    const std::lock_guard<std::mutex> lock(buffer.mutex);
    nb::dict result;
    result["schema_version"] = 1;
    result["capacity"] = kGemmTelemetryCapacity;
    result["buffered_records"] = buffer.records.size();
    result["captured_records"] = buffer.captured_records;
    result["dropped_records"] = buffer.dropped_records;
    result["next_sequence"] = buffer.next_sequence;
    result["operand_numa_sampling_method"] = "move_pages_query_no_migration";
    result["operand_numa_sample_limit_per_operand"] =
        kNumaPageSamplesPerOperand;
    result["operand_numa_address_selection_schema_version"] =
        kNumaAddressSelectionSchemaVersion;
    result["operand_numa_address_selection_policy"] =
        kNumaAddressSelectionPolicy;
    result["operand_numa_partial_boundary_pages_included"] = false;
    return result;
}

void observed_vendor_dgemm(bool transpose_a,
                           int m, int n, int k,
                           const double* a, int lda,
                           const double* b, int ldb,
                           double* c, int ldc,
                           int requested_threads,
                           int configured_threads,
                           int backend_threads,
                           double alpha, double beta,
                           const NativeIntegritySnapshotNumaEvidence*
                               native_integrity_snapshot_numa = nullptr);

void observed_vendor_dgemm_general(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transpose_a,
    CBLAS_TRANSPOSE transpose_b,
    int m, int n, int k,
    const double* a, int lda,
    const double* b, int ldb,
    double* c, int ldc,
    int requested_threads,
    int configured_threads,
    int backend_threads,
    double alpha, double beta,
    const NativeIntegritySnapshotNumaEvidence*
        native_integrity_snapshot_numa = nullptr
);

nb::dict test_vendor_entry_guard() {
    const OpenMPEntryState outside = capture_openmp_entry_state();
    int nested_checks = 0;
    int nested_rejections = 0;
    int production_boundary_checks = 0;
    int production_boundary_rejections = 0;
    int production_boundary_output_changes = 0;
#ifdef _OPENMP
    // Exercise the same rejecting function used at the production vendor
    // boundary, but catch inside the structured block so no exception crosses
    // an OpenMP boundary and no vendor routine is entered.
    #pragma omp parallel num_threads(2) \
        reduction(+:nested_checks,nested_rejections)
    {
        const OpenMPEntryState nested = capture_openmp_entry_state();
        ++nested_checks;
        try {
            require_vendor_entry_outside_openmp(nested);
        } catch (const std::runtime_error&) {
            ++nested_rejections;
        }
    }
    // Invoke the production observed-GEMM boundary from a lexical, serialized
    // OpenMP region.  Valid 1x1 operands make an unchanged output direct
    // evidence that rejection occurred before CBLAS, without concurrent BLAS
    // entry or an exception crossing the structured block.
    #pragma omp parallel num_threads(1) \
        reduction(+:production_boundary_checks,production_boundary_rejections,production_boundary_output_changes)
    {
        const double a = 2.0;
        const double b = 3.0;
        double c = 11.0;
        ++production_boundary_checks;
        try {
            observed_vendor_dgemm(
                false,
                1, 1, 1,
                &a, 1,
                &b, 1,
                &c, 1,
                1, 1, -1,
                1.0, 0.0
            );
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find(
                    "Refusing GxE vendor BLAS entry"
                ) != std::string::npos) {
                ++production_boundary_rejections;
            }
        }
        production_boundary_output_changes += c != 11.0 ? 1 : 0;
    }
#endif
    nb::dict result;
    result["openmp_enabled"] = bool(GWLDCORE_OPENMP_ENABLED);
    result["outside_allowed"] = !is_nested_vendor_entry(outside);
    result["outside_omp_in_parallel"] = outside.in_parallel;
    result["outside_omp_level"] = outside.level;
    result["nested_probe_executed"] = nested_checks > 0;
    result["nested_checks"] = nested_checks;
    result["nested_rejections"] = nested_rejections;
    result["all_nested_entries_rejected"] =
        nested_checks > 0 && nested_checks == nested_rejections;
    result["production_boundary_checks"] = production_boundary_checks;
    result["production_boundary_rejections"] = production_boundary_rejections;
    result["production_boundary_output_changes"] =
        production_boundary_output_changes;
    result["all_production_boundary_entries_rejected"] =
        production_boundary_checks > 0
        && production_boundary_checks == production_boundary_rejections;
    result["production_boundary_outputs_unchanged"] =
        production_boundary_output_changes == 0;
    return result;
}

size_t checked_add(size_t a, size_t b, const char* label) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error(std::string("GxE native size overflow in ") + label);
    }
    return a + b;
}

size_t checked_mul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("GxE native size overflow in ") + label);
    }
    return a * b;
}

int checked_blas_dim(size_t value, const char* label) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string("GxE native BLAS dimension overflow in ") + label);
    }
    return static_cast<int>(value);
}

void dgemm_nn_raw(int m, int n, int k,
                  const double* a, int lda,
                  const double* b, int ldb,
                  double* c, int ldc,
                  double alpha = 1.0, double beta = 0.0) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm_tn_raw(int m, int n, int k,
                  const double* a, int lda,
                  const double* b, int ldb,
                  double* c, int ldc,
                  double alpha = 1.0, double beta = 0.0) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

const char* cblas_transpose_name(CBLAS_TRANSPOSE transpose) {
    if (transpose == CblasNoTrans) return "N";
    if (transpose == CblasTrans) return "T";
    return "C";
}

#ifdef GWLDCORE_USE_BLIS
void execute_private_blis_gemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transpose_a,
    CBLAS_TRANSPOSE transpose_b,
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    double* c,
    int ldc,
    int requested_threads,
    double alpha,
    double beta
) {
    const bool column_major = layout == CblasColMajor;
    const bool trans_a = transpose_a != CblasNoTrans;
    const bool trans_b = transpose_b != CblasNoTrans;
    const dim_t a_rows = static_cast<dim_t>(trans_a ? k : m);
    const dim_t a_columns = static_cast<dim_t>(trans_a ? m : k);
    const dim_t b_rows = static_cast<dim_t>(trans_b ? n : k);
    const dim_t b_columns = static_cast<dim_t>(trans_b ? k : n);

    obj_t alpha_object;
    obj_t beta_object;
    obj_t a_object;
    obj_t b_object;
    obj_t c_object;
    bli_obj_create_1x1_with_attached_buffer(
        BLIS_DOUBLE, &alpha, &alpha_object
    );
    bli_obj_create_1x1_with_attached_buffer(
        BLIS_DOUBLE, &beta, &beta_object
    );
    bli_obj_create_with_attached_buffer(
        BLIS_DOUBLE, a_rows, a_columns, const_cast<double*>(a),
        column_major ? 1 : lda, column_major ? lda : 1, &a_object
    );
    bli_obj_create_with_attached_buffer(
        BLIS_DOUBLE, b_rows, b_columns, const_cast<double*>(b),
        column_major ? 1 : ldb, column_major ? ldb : 1, &b_object
    );
    bli_obj_create_with_attached_buffer(
        BLIS_DOUBLE, static_cast<dim_t>(m), static_cast<dim_t>(n), c,
        column_major ? 1 : ldc, column_major ? ldc : 1, &c_object
    );
    bli_obj_set_conjtrans(
        trans_a ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE, &a_object
    );
    bli_obj_set_conjtrans(
        trans_b ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE, &b_object
    );

    rntm_t runtime = BLIS_RNTM_INITIALIZER;
    bli_rntm_set_thread_impl(BLIS_POSIX, &runtime);
    bli_rntm_set_num_threads(
        static_cast<dim_t>(requested_threads), &runtime
    );
    // Use the regular packed path for consistent behavior across shapes.
    bli_rntm_disable_l3_sup(&runtime);
#if defined(__linux__)
    // OpenMP placement intentionally leaves the caller on its singleton
    // place. BLIS pthread workers inherit the caller mask, so temporarily
    // expose the already-authenticated selected CPU set and restore the
    // singleton before returning to application OpenMP code.
    ScopedBlisPthreadAffinity affinity;
#endif
    bli_gemm_ex(
        &alpha_object, &a_object, &b_object, &beta_object, &c_object,
        nullptr, &runtime
    );
#if defined(__linux__)
    affinity.restore();
#endif
}
#endif

void observed_vendor_dgemm_general(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transpose_a,
    CBLAS_TRANSPOSE transpose_b,
    int m, int n, int k,
    const double* a, int lda,
    const double* b, int ldb,
    double* c, int ldc,
    int requested_threads,
    int configured_threads,
    int backend_threads,
    double alpha,
    double beta,
    const NativeIntegritySnapshotNumaEvidence*
        native_integrity_snapshot_numa
) {
    const OpenMPEntryState omp = capture_openmp_entry_state();
    require_vendor_entry_outside_openmp(omp);
    validate_active_native_gemm_output_vendor_span(
        layout, m, n, c, ldc
    );
    const NativeNumaContractRequest contract_request =
        native_numa_contract_request();

    GemmTelemetryRecord record;
    record.native_gemm_output_numa =
        make_nonapplicable_native_gemm_output_numa_evidence(
            contract_request.required
        );
    if (native_integrity_snapshot_numa == nullptr) {
        record.native_integrity_snapshot_numa.contract_required =
            contract_request.required;
    } else {
        require_native_snapshot_request_match(
            contract_request, *native_integrity_snapshot_numa
        );
        record.native_integrity_snapshot_numa =
            *native_integrity_snapshot_numa;
    }
    record.sequence = reserve_gemm_telemetry_sequence();
    record.layout = layout == CblasRowMajor
        ? "row_major" : "column_major";
    record.transpose_a = cblas_transpose_name(transpose_a);
    record.transpose_b = cblas_transpose_name(transpose_b);
    record.operation = "dgemm_";
    if (layout == CblasRowMajor) record.operation += "row_";
    record.operation += record.transpose_a == "N" ? "n" : "t";
    record.operation += record.transpose_b == "N" ? "n" : "t";
    record.m = m;
    record.n = n;
    record.k = k;
    record.lda = lda;
    record.ldb = ldb;
    record.ldc = ldc;
    record.alpha = alpha;
    record.beta = beta;
    record.flop_count = 2.0
        * static_cast<double>(std::max(0, m))
        * static_cast<double>(std::max(0, n))
        * static_cast<double>(std::max(0, k));
    record.requested_threads = requested_threads;
    record.configured_threads = configured_threads;
    record.backend_threads = backend_threads;
    record.omp = omp;
    record.affinity = capture_cpu_affinity_evidence();
    record.backend = GWLDCORE_BLAS_VENDOR;
#ifdef GWLDCORE_USE_OPENBLAS
    if (const char* corename = openblas_get_corename()) {
        record.backend_corename = corename;
    }
    if (const char* config = openblas_get_config()) {
        record.backend_config = config;
    }
#elif defined(GWLDCORE_USE_BLIS)
    const arch_t architecture = bli_arch_query_id();
    if (const char* corename = bli_arch_string(architecture)) {
        record.backend_corename = corename;
    }
    if (const char* version = bli_info_get_version_str()) {
        record.backend_config = std::string("BLIS ") + version
            + " config=" + GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY;
    }
#endif

    // Affinity/configuration inspection stays outside this interval.  These
    // two clocks bracket only the CBLAS entry itself, so the CPU/wall ratio is
    // evidence about the vendor call rather than integrity or Python work.
    const double cpu_start = process_cpu_seconds_now();
    const auto wall_start = std::chrono::steady_clock::now();
#ifdef GWLDCORE_USE_BLIS
    execute_private_blis_gemm(
        layout, transpose_a, transpose_b,
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#else
    cblas_dgemm(
        layout, transpose_a, transpose_b,
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
#endif
    const auto wall_end = std::chrono::steady_clock::now();
    const double cpu_end = process_cpu_seconds_now();
    record.wall_seconds = std::chrono::duration<double>(
        wall_end - wall_start
    ).count();
    record.process_cpu_seconds = std::max(0.0, cpu_end - cpu_start);
    record.exit_cpu = current_cpu();
    // Query a fixed, tiny sample only after the timed vendor call.  The Linux
    // move_pages query mode observes placement without migrating pages.  Any
    // denial or kernel error is telemetry, never a GEMM failure.
    try {
        record.numa_page_samples = sample_gemm_operand_numa_pages(
            layout, transpose_a, transpose_b,
            m, n, k, a, lda, b, ldb, c, ldc
        );
    } catch (...) {
        record.numa_page_samples.syscall_errno = EFAULT;
        for (auto& operand : record.numa_page_samples.operands) {
            operand.query_state = NumaPageQueryState::syscall_error;
        }
    }
    record_gemm_telemetry(std::move(record));
}

void observed_vendor_dgemm(bool transpose_a,
                           int m, int n, int k,
                           const double* a, int lda,
                           const double* b, int ldb,
                           double* c, int ldc,
                           int requested_threads,
                           int configured_threads,
                           int backend_threads,
                           double alpha, double beta,
                           const NativeIntegritySnapshotNumaEvidence*
                               native_integrity_snapshot_numa) {
    observed_vendor_dgemm_general(
        CblasColMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        CblasNoTrans,
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, configured_threads, backend_threads,
        alpha, beta, native_integrity_snapshot_numa
    );
}

#ifndef GWLDCORE_USE_FIXED_VENDOR_BLAS
void dgemm_tn_vendor_observed(int m, int n, int k,
                              const double* a, int lda,
                              const double* b, int ldb,
                              double* c, int ldc,
                              int requested_threads,
                              double alpha = 1.0, double beta = 0.0) {
    observed_vendor_dgemm(
        true, m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, -1, -1, alpha, beta
    );
}

void dgemm_nn_vendor_observed(int m, int n, int k,
                              const double* a, int lda,
                              const double* b, int ldb,
                              double* c, int ldc,
                              int requested_threads,
                              double alpha = 1.0, double beta = 0.0,
                              const NativeIntegritySnapshotNumaEvidence*
                                  native_integrity_snapshot_numa = nullptr) {
    observed_vendor_dgemm(
        false, m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, -1, -1, alpha, beta,
        native_integrity_snapshot_numa
    );
}

void dgemm_tt_vendor_observed(int m, int n, int k,
                              const double* a, int lda,
                              const double* b, int ldb,
                              double* c, int ldc,
                              int requested_threads) {
    observed_vendor_dgemm_general(
        CblasColMajor, CblasTrans, CblasTrans,
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, -1, -1, 1.0, 0.0
    );
}

void dgemm_row_tn_vendor_observed(int m, int n, int k,
                                  const double* a, int lda,
                                  const double* b, int ldb,
                                  double* c, int ldc,
                                  int requested_threads) {
    observed_vendor_dgemm_general(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, -1, -1, 1.0, 0.0
    );
}
#endif

#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
std::mutex& fixed_vendor_gemm_call_mutex() {
    static std::mutex mutex;
    return mutex;
}
#endif

#ifdef GWLDCORE_USE_OPENBLAS
struct FixedOpenBLASRuntime {
    std::once_flag initialize;
    std::atomic<int> threads{0};
};

FixedOpenBLASRuntime& fixed_openblas_runtime() {
    static FixedOpenBLASRuntime runtime;
    return runtime;
}

int configure_fixed_openblas_threads(int requested_threads) {
    const int requested = std::max(1, requested_threads);
    auto& runtime = fixed_openblas_runtime();
    std::call_once(runtime.initialize, [&]() {
        // OpenBLAS documents its runtime setter for one-time initialization,
        // not per-call tuning. A private build makes this state unreachable
        // from NumPy/threadpoolctl; the checked shared fallback still fixes it
        // once and verifies that no external code later changes it.
        openblas_set_num_threads(requested);
        const int observed = openblas_get_num_threads();
        if (observed != requested) {
            throw std::runtime_error(
                "OpenBLAS did not accept the fixed GxE thread count"
            );
        }
        runtime.threads.store(observed, std::memory_order_release);
    });
    const int fixed_threads = runtime.threads.load(std::memory_order_acquire);
    if (requested != fixed_threads) {
        throw std::runtime_error(
            "The GxE OpenBLAS runtime was already initialized with a different "
            "thread count (configured=" + std::to_string(fixed_threads) +
            ", requested=" + std::to_string(requested) + ")"
        );
    }
#if defined(GWLDCORE_PRIVATE_OPENBLAS)
    const bool externally_visible_thread_state = openblas_get_parallel() == 2;
#else
    const bool externally_visible_thread_state = true;
#endif
    if (externally_visible_thread_state &&
        openblas_get_num_threads() != fixed_threads) {
#if defined(GWLDCORE_PRIVATE_OPENBLAS)
        throw std::runtime_error(
            "The GxE OpenBLAS thread count changed after initialization"
        );
#else
        // The public compatibility build shares OpenBLAS with NumPy and
        // threadpoolctl.  Those callers may legitimately change the shared
        // process-wide setting between protected GxE calls.  Reassert the
        // already-frozen GxE value at every protected boundary; a different
        // requested value above remains an error, and a concurrent or failed
        // reconfiguration still fails closed before entering CBLAS.
        openblas_set_num_threads(fixed_threads);
        if (openblas_get_num_threads() != fixed_threads) {
            throw std::runtime_error(
                "The shared GxE OpenBLAS thread count could not be restored"
            );
        }
#endif
    }
    return fixed_threads;
}
#endif

#ifdef GWLDCORE_USE_BLIS
struct BlisThreadWays {
    int jc = 1;
    int pc = 1;
    int ic = 1;
    int jr = 1;
    int ir = 1;

    bool operator==(const BlisThreadWays& other) const noexcept {
        return jc == other.jc && pc == other.pc && ic == other.ic
            && jr == other.jr && ir == other.ir;
    }
};

enum class BlisThreadStrategy {
    automatic,
    manual,
};

const char* blis_thread_strategy_name(BlisThreadStrategy strategy) noexcept {
    return strategy == BlisThreadStrategy::manual ? "manual" : "automatic";
}

struct BlisEnvironmentValue {
    std::string name;
    bool present = false;
    std::string value;

    bool operator==(const BlisEnvironmentValue& other) const noexcept {
        return name == other.name && present == other.present
            && value == other.value;
    }
};

using BlisEnvironmentSnapshot = std::vector<BlisEnvironmentValue>;

constexpr std::array<const char*, 22> kBlisEnvironmentContractNames{{
    "BLIS_NUM_THREADS",
    "BLIS_NT",
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_THREAD_IMPL",
    "BLIS_TI",
    "BLIS_ARCH_TYPE",
    "BLIS_ARCH_DEBUG",
    "BLIS_PACK_A",
    "BLIS_PACK_B",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_MAX_ACTIVE_LEVELS",
    "OMP_WAIT_POLICY",
    "GOMP_CPU_AFFINITY",
    "GOMP_SPINCOUNT",
}};

BlisEnvironmentSnapshot capture_blis_environment() {
    BlisEnvironmentSnapshot snapshot;
    snapshot.reserve(kBlisEnvironmentContractNames.size());
    for (const char* name : kBlisEnvironmentContractNames) {
        const char* value = std::getenv(name);
        snapshot.push_back(BlisEnvironmentValue{
            name,
            value != nullptr,
            value == nullptr ? "" : value,
        });
    }
    return snapshot;
}

const BlisEnvironmentValue& blis_environment_value(
    const BlisEnvironmentSnapshot& snapshot,
    const char* name
) {
    const auto found = std::find_if(
        snapshot.begin(), snapshot.end(),
        [name](const BlisEnvironmentValue& value) {
            return value.name == name;
        }
    );
    if (found == snapshot.end()) {
        throw std::logic_error(
            std::string("Internal BLIS environment contract omits ") + name
        );
    }
    return *found;
}

int parse_positive_blis_environment_integer(
    const BlisEnvironmentValue& environment
) {
    if (!environment.present || environment.value.empty()) {
        throw std::runtime_error(
            environment.name + " must be an explicitly set positive integer"
        );
    }
    int value = 0;
    for (const char character : environment.value) {
        if (character < '0' || character > '9') {
            throw std::runtime_error(
                environment.name + " must contain only decimal digits"
            );
        }
        const int digit = character - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / 10) {
            throw std::runtime_error(environment.name + " exceeds INT_MAX");
        }
        value = value * 10 + digit;
    }
    if (value <= 0) {
        throw std::runtime_error(environment.name + " must be positive");
    }
    return value;
}

struct BlisRequestedContract {
    int threads = 0;
    BlisThreadStrategy strategy = BlisThreadStrategy::automatic;
    BlisThreadWays ways;
    BlisEnvironmentSnapshot environment;
};

BlisRequestedContract requested_blis_contract() {
    BlisRequestedContract contract;
    contract.environment = capture_blis_environment();
    contract.threads = parse_positive_blis_environment_integer(
        blis_environment_value(contract.environment, "BLIS_NUM_THREADS")
    );
    for (const char* alias : {"BLIS_NT", "BLIS_TI"}) {
        if (blis_environment_value(contract.environment, alias).present) {
            throw std::runtime_error(
                std::string(alias) + " is forbidden by blis_process_start_v1; "
                "use the canonical BLIS environment name"
            );
        }
    }
    const auto& thread_impl = blis_environment_value(
        contract.environment, "BLIS_THREAD_IMPL"
    );
    if (thread_impl.present && thread_impl.value != "openmp") {
        throw std::runtime_error(
            "BLIS_THREAD_IMPL must be unset or exactly 'openmp'"
        );
    }
    if (blis_environment_value(
            contract.environment, "BLIS_ARCH_TYPE"
        ).present) {
        throw std::runtime_error(
            "BLIS_ARCH_TYPE overrides are forbidden for the attested BLIS build"
        );
    }
    for (const char* name : {
             "BLIS_ARCH_DEBUG", "BLIS_PACK_A", "BLIS_PACK_B"
         }) {
        if (blis_environment_value(contract.environment, name).present) {
            throw std::runtime_error(
                std::string(name)
                + " overrides are forbidden for the attested BLIS build"
            );
        }
    }

    constexpr std::array<const char*, 5> way_names{{
        "BLIS_JC_NT", "BLIS_PC_NT", "BLIS_IC_NT", "BLIS_JR_NT", "BLIS_IR_NT"
    }};
    int present_ways = 0;
    for (const char* name : way_names) {
        present_ways += blis_environment_value(
            contract.environment, name
        ).present ? 1 : 0;
    }
    if (present_ways != 0 && present_ways != int(way_names.size())) {
        throw std::runtime_error(
            "Manual BLIS loop threading requires all five BLIS_*_NT way variables"
        );
    }
    if (present_ways == int(way_names.size())) {
        contract.strategy = BlisThreadStrategy::manual;
        contract.ways = BlisThreadWays{
            parse_positive_blis_environment_integer(
                blis_environment_value(contract.environment, "BLIS_JC_NT")
            ),
            parse_positive_blis_environment_integer(
                blis_environment_value(contract.environment, "BLIS_PC_NT")
            ),
            parse_positive_blis_environment_integer(
                blis_environment_value(contract.environment, "BLIS_IC_NT")
            ),
            parse_positive_blis_environment_integer(
                blis_environment_value(contract.environment, "BLIS_JR_NT")
            ),
            parse_positive_blis_environment_integer(
                blis_environment_value(contract.environment, "BLIS_IR_NT")
            ),
        };
        if (contract.ways.pc != 1) {
            throw std::runtime_error(
                "Manual BLIS loop threading requires BLIS_PC_NT=1"
            );
        }
        int64_t product = 1;
        for (const int way : {
                 contract.ways.jc, contract.ways.pc, contract.ways.ic,
                 contract.ways.jr, contract.ways.ir
             }) {
            if (product > std::numeric_limits<int>::max() / way) {
                throw std::runtime_error("Manual BLIS loop-way product exceeds INT_MAX");
            }
            product *= way;
        }
        if (product != contract.threads) {
            throw std::runtime_error(
                "Manual BLIS loop-way product differs from BLIS_NUM_THREADS"
            );
        }
    }
    return contract;
}

BlisThreadWays observed_blis_thread_ways() {
    return BlisThreadWays{
        checked_blas_dim(
            static_cast<size_t>(bli_thread_get_jc_nt()), "BLIS jc threads"
        ),
        checked_blas_dim(
            static_cast<size_t>(bli_thread_get_pc_nt()), "BLIS pc threads"
        ),
        checked_blas_dim(
            static_cast<size_t>(bli_thread_get_ic_nt()), "BLIS ic threads"
        ),
        checked_blas_dim(
            static_cast<size_t>(bli_thread_get_jr_nt()), "BLIS jr threads"
        ),
        checked_blas_dim(
            static_cast<size_t>(bli_thread_get_ir_nt()), "BLIS ir threads"
        ),
    };
}

std::string blis_runtime_config_string() {
    const char* version = bli_info_get_version_str();
    return std::string("BLIS ") + (version == nullptr ? "" : version)
        + " config=" + GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY;
}

std::string blis_runtime_corename_string() {
    const char* name = bli_arch_string(bli_arch_query_id());
    return name == nullptr ? "" : name;
}

void validate_blis_build_capabilities() {
    const char* version = bli_info_get_version_str();
    if (version == nullptr || std::string(version) != BLIS_VERSION_STRING) {
        throw std::runtime_error(
            "The linked BLIS runtime version differs from its attested header"
        );
    }
    if (bli_info_get_enable_pthreads() == 0 ||
        bli_info_get_enable_pthreads_as_default() == 0) {
        throw std::runtime_error(
            "The private BLIS runtime lacks pthreads as its compiled default"
        );
    }
    if (bli_info_get_enable_tls() == 0) {
        throw std::runtime_error(
            "The private BLIS runtime lacks required application-thread TLS"
        );
    }
    if (blis_runtime_corename_string() != GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY) {
        throw std::runtime_error(
            "The private BLIS runtime architecture differs from its attested "
            "configuration family"
        );
    }
}

void validate_blis_openmp_capacity(int requested_threads) {
#ifdef _OPENMP
    validate_configured_openmp_placement_for_entry();
    if (omp_get_dynamic() != 0) {
        throw std::runtime_error(
            "The private BLIS contract requires OpenMP dynamic teams to be disabled"
        );
    }
    if (omp_get_thread_limit() < requested_threads) {
        throw std::runtime_error(
            "The OpenMP thread limit is smaller than BLIS_NUM_THREADS"
        );
    }
    if (effective_openmp_capacity() < requested_threads) {
        throw std::runtime_error(
            "The effective OpenMP capacity is smaller than BLIS_NUM_THREADS"
        );
    }
#else
    (void)requested_threads;
    throw std::runtime_error(
        "The private BLIS runtime requires gxeldcore OpenMP support"
    );
#endif
}

struct FixedBlisRuntime {
    std::once_flag initialize;
    std::atomic<int> threads{0};
    std::thread::id owner_thread;
    BlisThreadStrategy strategy = BlisThreadStrategy::automatic;
    BlisThreadWays ways;
    BlisEnvironmentSnapshot environment;
};

FixedBlisRuntime& fixed_blis_runtime() {
    static FixedBlisRuntime runtime;
    return runtime;
}

void validate_observed_blis_contract(
    const BlisRequestedContract& requested,
    const BlisThreadWays& expected_ways
) {
    validate_blis_build_capabilities();
    validate_blis_openmp_capacity(requested.threads);
    if (bli_thread_get_thread_impl() != BLIS_POSIX) {
        throw std::runtime_error(
            "The private BLIS TLS runtime is not using pthreads"
        );
    }
    if (bli_thread_get_num_threads() != requested.threads) {
        throw std::runtime_error(
            "The private BLIS TLS thread count differs from BLIS_NUM_THREADS"
        );
    }
    if (!(observed_blis_thread_ways() == expected_ways)) {
        throw std::runtime_error(
            "The private BLIS TLS loop ways changed after configuration"
        );
    }
}

int configure_fixed_blis_threads(int requested_threads) {
    if (requested_threads <= 0) {
        throw std::runtime_error("The GxE BLIS thread count must be positive");
    }
    const BlisRequestedContract requested = requested_blis_contract();
    if (requested_threads != requested.threads) {
        throw std::runtime_error(
            "The requested GxE BLIS thread count differs from process-start "
            "BLIS_NUM_THREADS"
        );
    }
    auto& runtime = fixed_blis_runtime();
    std::call_once(runtime.initialize, [&]() {
        validate_blis_build_capabilities();
        validate_blis_openmp_capacity(requested.threads);
        bli_thread_set_thread_impl(BLIS_POSIX);
        bli_thread_set_num_threads(requested.threads);
        if (requested.strategy == BlisThreadStrategy::manual) {
            bli_thread_set_ways(
                requested.ways.jc,
                requested.ways.pc,
                requested.ways.ic,
                requested.ways.jr,
                requested.ways.ir
            );
        }
        const BlisThreadWays expected_ways =
            requested.strategy == BlisThreadStrategy::manual
                ? requested.ways : BlisThreadWays{};
        validate_observed_blis_contract(requested, expected_ways);
        runtime.owner_thread = std::this_thread::get_id();
        runtime.strategy = requested.strategy;
        runtime.ways = expected_ways;
        runtime.environment = requested.environment;
        runtime.threads.store(requested.threads, std::memory_order_release);
    });

    const int fixed_threads = runtime.threads.load(std::memory_order_acquire);
    if (fixed_threads <= 0) {
        throw std::runtime_error("The private BLIS runtime was not configured");
    }
    if (std::this_thread::get_id() != runtime.owner_thread) {
        throw std::runtime_error(
            "The private BLIS runtime may be used only by its configuring owner thread"
        );
    }
    if (requested_threads != fixed_threads) {
        throw std::runtime_error(
            "The GxE BLIS runtime was already initialized with a different "
            "thread count (configured=" + std::to_string(fixed_threads) +
            ", requested=" + std::to_string(requested_threads) + ")"
        );
    }
    if (requested.strategy != runtime.strategy ||
        !(requested.ways == (runtime.strategy == BlisThreadStrategy::manual
                                 ? runtime.ways : BlisThreadWays{})) ||
        requested.environment != runtime.environment) {
        throw std::runtime_error(
            "The private BLIS process-start environment changed after configuration"
        );
    }
    validate_observed_blis_contract(
        requested,
        runtime.strategy == BlisThreadStrategy::manual
            ? runtime.ways : BlisThreadWays{}
    );
    return fixed_threads;
}

bool blis_owner_thread_configured() noexcept {
    return fixed_blis_runtime().threads.load(std::memory_order_acquire) > 0;
}

BlisRequestedContract blis_contract_for_build_info() {
    const BlisRequestedContract requested = requested_blis_contract();
    // BLIS settings are application-thread TLS state. Any BLIS query may
    // initialize that state, so build_info() must freeze the same immutable
    // owner/environment contract as the first numerical entry rather than
    // permitting a later call to re-baseline it.
    configure_fixed_blis_threads(requested.threads);
    auto& runtime = fixed_blis_runtime();
    BlisRequestedContract result = requested;
    result.strategy = runtime.strategy;
    result.ways = runtime.ways;
    return result;
}
#endif

#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
int configure_fixed_vendor_threads_unlocked(int requested_threads) {
#ifdef GWLDCORE_USE_OPENBLAS
    return configure_fixed_openblas_threads(requested_threads);
#else
    return configure_fixed_blis_threads(requested_threads);
#endif
}

int fixed_vendor_backend_threads() {
#ifdef GWLDCORE_USE_OPENBLAS
    return openblas_get_num_threads();
#else
    return checked_blas_dim(
        static_cast<size_t>(bli_thread_get_num_threads()), "BLIS threads"
    );
#endif
}

int configure_fixed_vendor_threads(int requested_threads) {
    const std::lock_guard<std::mutex> lock(fixed_vendor_gemm_call_mutex());
    return configure_fixed_vendor_threads_unlocked(requested_threads);
}

void dgemm_tn_fixed_vendor(int m, int n, int k,
                           const double* a, int lda,
                           const double* b, int ldb,
                           double* c, int ldc,
                           int requested_threads,
                           double alpha = 1.0, double beta = 0.0) {
    const std::lock_guard<std::mutex> lock(fixed_vendor_gemm_call_mutex());
    const int configured_threads = configure_fixed_vendor_threads_unlocked(
        requested_threads
    );
    observed_vendor_dgemm(
        true, m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, configured_threads, fixed_vendor_backend_threads(),
        alpha, beta
    );
}

void dgemm_nn_fixed_vendor(int m, int n, int k,
                           const double* a, int lda,
                           const double* b, int ldb,
                           double* c, int ldc,
                           int requested_threads,
                           double alpha = 1.0, double beta = 0.0,
                           const NativeIntegritySnapshotNumaEvidence*
                               native_integrity_snapshot_numa = nullptr) {
    const std::lock_guard<std::mutex> lock(fixed_vendor_gemm_call_mutex());
    const int configured_threads = configure_fixed_vendor_threads_unlocked(
        requested_threads
    );
    observed_vendor_dgemm(
        false, m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, configured_threads, fixed_vendor_backend_threads(),
        alpha, beta, native_integrity_snapshot_numa
    );
}

void dgemm_tt_fixed_vendor(int m, int n, int k,
                           const double* a, int lda,
                           const double* b, int ldb,
                           double* c, int ldc,
                           int requested_threads) {
    const std::lock_guard<std::mutex> lock(fixed_vendor_gemm_call_mutex());
    const int configured_threads = configure_fixed_vendor_threads_unlocked(
        requested_threads
    );
    observed_vendor_dgemm_general(
        CblasColMajor, CblasTrans, CblasTrans,
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, configured_threads, fixed_vendor_backend_threads(),
        1.0, 0.0
    );
}

void dgemm_row_tn_fixed_vendor(int m, int n, int k,
                               const double* a, int lda,
                               const double* b, int ldb,
                               double* c, int ldc,
                               int requested_threads) {
    const std::lock_guard<std::mutex> lock(fixed_vendor_gemm_call_mutex());
    const int configured_threads = configure_fixed_vendor_threads_unlocked(
        requested_threads
    );
    observed_vendor_dgemm_general(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, configured_threads, fixed_vendor_backend_threads(),
        1.0, 0.0
    );
}
#endif

struct MatrixFingerprint {
    uint64_t xor_hash = 0;
    uint64_t sum_hash = 0;

    bool operator==(const MatrixFingerprint& other) const noexcept {
        return xor_hash == other.xor_hash && sum_hash == other.sum_hash;
    }
};

class RetryableGemmInputMutation : public std::runtime_error {
public:
    RetryableGemmInputMutation()
        : std::runtime_error(
            "Vendor BLAS altered a reconstructable GxE GEMM input"
        ) {}
};

// These cache-tiled kernels are deliberately BLAS-independent. OpenMP assigns
// disjoint output tiles, every output entry is computed exactly once, and the
// reduction order within an entry is deterministic. They provide checksum
// construction, selective repair, and a portable vendor fallback.  The two
// kernels remain available in guard-free builds for cheap fixed-rank updates
// and optional feature-moment diagnostics.
void dgemm_tn_tiled(int m, int n, int k,
                    const double* a, int lda,
                    const double* b, int ldb,
                    double* c, int ldc,
                    int requested_threads,
                    double alpha = 1.0, double beta = 0.0) {
    constexpr int kRowTile = 8;
    constexpr int kColumnTile = 16;
    constexpr int kReductionTile = 1024;
    const int row_tiles = (m + kRowTile - 1) / kRowTile;
    const int column_tiles = (n + kColumnTile - 1) / kColumnTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * column_tiles;
    const int threads = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(requested_threads, tasks)
    ));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
        const int row0 = static_cast<int>(task / column_tiles) * kRowTile;
        const int column0 = static_cast<int>(task % column_tiles) * kColumnTile;
        const int rows = std::min(kRowTile, m - row0);
        const int columns = std::min(kColumnTile, n - column0);
        double sums[kRowTile * kColumnTile] = {};
        for (int reduction0 = 0; reduction0 < k;
             reduction0 += kReductionTile) {
            const int reduction1 = std::min(k, reduction0 + kReductionTile);
            for (int row = 0; row < rows; ++row) {
                const double* a_column = a +
                    static_cast<size_t>(row0 + row) *
                        static_cast<size_t>(lda);
                for (int column = 0; column < columns; ++column) {
                    const double* b_column = b +
                        static_cast<size_t>(column0 + column) *
                            static_cast<size_t>(ldb);
                    double partial = 0.0;
#ifdef _OPENMP
                    #pragma omp simd reduction(+:partial)
#endif
                    for (int reduction = reduction0;
                         reduction < reduction1; ++reduction) {
                        partial += a_column[reduction] * b_column[reduction];
                    }
                    sums[row * kColumnTile + column] += partial;
                }
            }
        }
        for (int column = 0; column < columns; ++column) {
            double* c_column = c +
                static_cast<size_t>(column0 + column) *
                    static_cast<size_t>(ldc);
            for (int row = 0; row < rows; ++row) {
                const int output_row = row0 + row;
                c_column[output_row] =
                    alpha * sums[row * kColumnTile + column]
                    + (beta == 0.0 ? 0.0 : beta * c_column[output_row]);
            }
        }
    }
}

void dgemm_nn_tiled(int m, int n, int k,
                    const double* a, int lda,
                    const double* b, int ldb,
                    double* c, int ldc,
                    int requested_threads,
                    double alpha = 1.0, double beta = 0.0) {
    constexpr int kRowTile = 256;
    constexpr int kColumnTile = 8;
    const int row_tiles = (m + kRowTile - 1) / kRowTile;
    const int column_tiles = (n + kColumnTile - 1) / kColumnTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * column_tiles;
    const int threads = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(requested_threads, tasks)
    ));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
        const int row0 = static_cast<int>(task / column_tiles) * kRowTile;
        const int column0 = static_cast<int>(task % column_tiles) * kColumnTile;
        const int row1 = std::min(m, row0 + kRowTile);
        const int column1 = std::min(n, column0 + kColumnTile);
        for (int column = column0; column < column1; ++column) {
            double* c_column = c +
                static_cast<size_t>(column) * static_cast<size_t>(ldc);
#ifdef _OPENMP
            #pragma omp simd
#endif
            for (int row = row0; row < row1; ++row) {
                c_column[row] =
                    beta == 0.0 ? 0.0 : beta * c_column[row];
            }
        }
        for (int reduction = 0; reduction < k; ++reduction) {
            const double* a_column = a +
                static_cast<size_t>(reduction) * static_cast<size_t>(lda);
            for (int column = column0; column < column1; ++column) {
                double* c_column = c +
                    static_cast<size_t>(column) * static_cast<size_t>(ldc);
                const double weight = alpha * b[
                    static_cast<size_t>(column) * static_cast<size_t>(ldb)
                    + static_cast<size_t>(reduction)
                ];
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int row = row0; row < row1; ++row) {
                    c_column[row] += a_column[row] * weight;
                }
            }
        }
    }
}

void dgemm_tt_tiled(int m, int n, int k,
                    const double* a, int lda,
                    const double* b, int ldb,
                    double* c, int ldc,
                    int requested_threads) {
    constexpr int kRowTile = 8;
    constexpr int kColumnTile = 16;
    constexpr int kReductionTile = 1024;
    const int row_tiles = (m + kRowTile - 1) / kRowTile;
    const int column_tiles = (n + kColumnTile - 1) / kColumnTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * column_tiles;
    const int threads = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(requested_threads, tasks)
    ));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
        const int row0 = static_cast<int>(task / column_tiles) * kRowTile;
        const int column0 = static_cast<int>(task % column_tiles) * kColumnTile;
        const int rows = std::min(kRowTile, m - row0);
        const int columns = std::min(kColumnTile, n - column0);
        double sums[kRowTile * kColumnTile] = {};
        for (int reduction0 = 0; reduction0 < k;
             reduction0 += kReductionTile) {
            const int reduction1 = std::min(k, reduction0 + kReductionTile);
            for (int reduction = reduction0; reduction < reduction1; ++reduction) {
                const double* b_column = b
                    + static_cast<size_t>(reduction)
                        * static_cast<size_t>(ldb)
                    + static_cast<size_t>(column0);
                for (int row = 0; row < rows; ++row) {
                    const double value = a[
                        static_cast<size_t>(row0 + row)
                            * static_cast<size_t>(lda)
                        + static_cast<size_t>(reduction)
                    ];
                    for (int column = 0; column < columns; ++column) {
                        sums[row * kColumnTile + column] +=
                            value * b_column[column];
                    }
                }
            }
        }
        for (int column = 0; column < columns; ++column) {
            double* output = c
                + static_cast<size_t>(column0 + column)
                    * static_cast<size_t>(ldc)
                + static_cast<size_t>(row0);
            for (int row = 0; row < rows; ++row) {
                output[row] = sums[row * kColumnTile + column];
            }
        }
    }
}

void dgemm_row_nn_tiled(int m, int n, int k,
                        const double* a, int lda,
                        const double* b, int ldb,
                        double* c, int ldc,
                        int requested_threads) {
    constexpr int kRowTile = 16;
    constexpr int kColumnTile = 16;
    const int row_tiles = (m + kRowTile - 1) / kRowTile;
    const int column_tiles = (n + kColumnTile - 1) / kColumnTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * column_tiles;
    const int threads = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(requested_threads, tasks)
    ));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
        const int row0 = static_cast<int>(task / column_tiles) * kRowTile;
        const int column0 = static_cast<int>(task % column_tiles) * kColumnTile;
        const int rows = std::min(kRowTile, m - row0);
        const int columns = std::min(kColumnTile, n - column0);
        double sums[kRowTile * kColumnTile] = {};
        for (int reduction = 0; reduction < k; ++reduction) {
            const double* b_row = b
                + static_cast<size_t>(reduction) * static_cast<size_t>(ldb)
                + static_cast<size_t>(column0);
            for (int row = 0; row < rows; ++row) {
                const double value = a[
                    static_cast<size_t>(row0 + row)
                        * static_cast<size_t>(lda)
                    + static_cast<size_t>(reduction)
                ];
                for (int column = 0; column < columns; ++column) {
                    sums[row * kColumnTile + column] += value * b_row[column];
                }
            }
        }
        for (int row = 0; row < rows; ++row) {
            double* output = c
                + static_cast<size_t>(row0 + row)
                    * static_cast<size_t>(ldc)
                + static_cast<size_t>(column0);
            for (int column = 0; column < columns; ++column) {
                output[column] = sums[row * kColumnTile + column];
            }
        }
    }
}

void dgemm_row_tn_tiled(int m, int n, int k,
                        const double* a, int lda,
                        const double* b, int ldb,
                        double* c, int ldc,
                        int requested_threads) {
    constexpr int kRowTile = 8;
    constexpr int kColumnTile = 16;
    const int row_tiles = (m + kRowTile - 1) / kRowTile;
    const int column_tiles = (n + kColumnTile - 1) / kColumnTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * column_tiles;
    const int threads = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(requested_threads, tasks)
    ));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
        const int row0 = static_cast<int>(task / column_tiles) * kRowTile;
        const int column0 = static_cast<int>(task % column_tiles) * kColumnTile;
        const int rows = std::min(kRowTile, m - row0);
        const int columns = std::min(kColumnTile, n - column0);
        double sums[kRowTile * kColumnTile] = {};
        for (int reduction = 0; reduction < k; ++reduction) {
            const double* b_row = b
                + static_cast<size_t>(reduction) * static_cast<size_t>(ldb)
                + static_cast<size_t>(column0);
            const double* a_row = a
                + static_cast<size_t>(reduction) * static_cast<size_t>(lda)
                + static_cast<size_t>(row0);
            for (int row = 0; row < rows; ++row) {
                const double value = a_row[row];
                for (int column = 0; column < columns; ++column) {
                    sums[row * kColumnTile + column] += value * b_row[column];
                }
            }
        }
        for (int row = 0; row < rows; ++row) {
            double* output = c
                + static_cast<size_t>(row0 + row)
                    * static_cast<size_t>(ldc)
                + static_cast<size_t>(column0);
            for (int column = 0; column < columns; ++column) {
                output[column] = sums[row * kColumnTile + column];
            }
        }
    }
}

struct GemmIntegrityResolution {
    int64_t checksum_recomputed_columns = 0;
    int64_t materially_repaired_columns = 0;

    int64_t roundoff_only_columns() const noexcept {
        return checksum_recomputed_columns - materially_repaired_columns;
    }
};

#if defined(GWLDCORE_GEMM_INTEGRITY)

constexpr int kGemmIntegrityChecks = 8;
constexpr int64_t kCheckedGemmMinimumFlops = 1000000000LL;
constexpr int kDeterministicTnMaximumColumns = 64;
constexpr uint64_t kDeterministicTnMaximumFlops = 2000000000ULL;
constexpr int kDeterministicFeatureTnMaximumRows = 128;
constexpr int kDeterministicFeatureTnMaximumColumns = 2048;
constexpr int kDeterministicFeatureTnMaximumReduction = 16384;
constexpr uint64_t kDeterministicFeatureTnMaximumFlops = 5000000000ULL;

bool gemm_requires_integrity_checks(int m, int n, int k) {
#if !defined(GWLDCORE_GEMM_CHECKSUM)
    (void)m;
    (void)n;
    (void)k;
    return false;
#else
    if (m <= 0 || n <= 0 || k <= 0) return false;
    constexpr uint64_t minimum_products =
        (static_cast<uint64_t>(kCheckedGemmMinimumFlops) + 1U) / 2U;
    const uint64_t mn =
        static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    const uint64_t required_mn =
        (minimum_products + static_cast<uint64_t>(k) - 1U) /
        static_cast<uint64_t>(k);
    return mn >= required_mn;
#endif
}

bool gemm_uses_deterministic_small_tn(int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0
        || n > kDeterministicTnMaximumColumns) {
        return false;
    }
    const uint64_t mn =
        static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    if (mn > kDeterministicTnMaximumFlops / 2U) return false;
    const uint64_t products = mn * static_cast<uint64_t>(k);
    return products <= kDeterministicTnMaximumFlops / 2U;
}

bool gemm_uses_deterministic_feature_tn(int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0
        || m > kDeterministicFeatureTnMaximumRows
        || n > kDeterministicFeatureTnMaximumColumns
        || k > kDeterministicFeatureTnMaximumReduction) {
        return false;
    }
    const uint64_t mn =
        static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    if (mn > kDeterministicFeatureTnMaximumFlops / 2U) return false;
    const uint64_t products = mn * static_cast<uint64_t>(k);
    return products <= kDeterministicFeatureTnMaximumFlops / 2U;
}

size_t gemm_integrity_workspace_elements(int m, int n, int k) {
    if (!gemm_requires_integrity_checks(m, n, k)) return 0;
    size_t dimensions = checked_add(
        static_cast<size_t>(m),
        checked_mul(2U, static_cast<size_t>(k), "GEMM integrity workspace"),
        "GEMM integrity workspace"
    );
    dimensions = checked_add(
        dimensions,
        checked_mul(2U, static_cast<size_t>(n), "GEMM integrity workspace"),
        "GEMM integrity workspace"
    );
    const size_t checks = checked_mul(
        static_cast<size_t>(kGemmIntegrityChecks), dimensions,
        "GEMM integrity workspace"
    );
    const size_t operand_b = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected GEMM operand B"
    );
    return checked_add(checks, operand_b, "protected GEMM workspace");
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::vector<double> gemm_integrity_coefficients(int rows) {
    std::vector<double> coefficients(
        checked_mul(
            static_cast<size_t>(rows),
            static_cast<size_t>(kGemmIntegrityChecks),
            "GEMM integrity coefficients"
        )
    );
    for (int check = 0; check < kGemmIntegrityChecks; ++check) {
        for (int row = 0; row < rows; ++row) {
            if (check == 0) {
                coefficients[static_cast<size_t>(row)] =
                    (row & 1) == 0 ? 1.0 : -1.0;
                continue;
            }
            const uint64_t key =
                (static_cast<uint64_t>(check + 1) << 32U)
                ^ static_cast<uint64_t>(row + 1);
            const uint64_t mixed = splitmix64(key);
            // Continuous, bounded weights avoid the exact pairwise
            // cancellation that can let a partition-swap error pass a
            // Rademacher checksum. The upper 53 bits map exactly onto a
            // binary64 fraction; the low bit supplies the sign.
            constexpr double kInverseTwoTo53 =
                1.0 / 9007199254740992.0;
            const double magnitude = 0.5 +
                static_cast<double>(mixed >> 11U) * kInverseTwoTo53;
            coefficients[
                static_cast<size_t>(check) * static_cast<size_t>(rows)
                + static_cast<size_t>(row)
            ] = (mixed & 1ULL) == 0ULL ? -magnitude : magnitude;
        }
    }
    return coefficients;
}

double floating_point_gamma(int reduction_terms) {
    const double product = static_cast<double>(std::max(1, reduction_terms))
        * std::numeric_limits<double>::epsilon();
    if (product >= 0.5) return std::numeric_limits<double>::infinity();
    return product / (1.0 - product);
}

bool gemm_integrity_disagrees(double expected, double observed,
                              int checksum_terms, int product_terms) {
    // The check compares (u^T A)B with u^T(AB).  A conservative multiple of
    // Higham's gamma_k bound covers the three differently ordered reductions
    // without using a shape-independent absolute tolerance.  The factor also
    // covers ordinary vendor/vectorized summation trees; observed host faults
    // are orders of magnitude larger than this roundoff envelope.
    const double relative_bound = 32.0 * (
        2.0 * floating_point_gamma(checksum_terms)
        + floating_point_gamma(product_terms)
        + std::numeric_limits<double>::epsilon()
    );
    const double tolerance = relative_bound * std::max(
        1.0, std::max(std::abs(expected), std::abs(observed))
    );
    return !std::isfinite(expected) || !std::isfinite(observed)
        || std::abs(expected - observed) > tolerance;
}

MatrixFingerprint fingerprint_col_major_matrix(
    int rows, int columns, const double* matrix, int leading_dimension,
    int requested_threads
) {
    uint64_t xor_hash = 0;
    uint64_t sum_hash = 0;
    const int threads = std::max(1, std::min(requested_threads, columns));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) \
        reduction(^:xor_hash) reduction(+:sum_hash)
#endif
    for (int column = 0; column < columns; ++column) {
        const double* values = matrix +
            static_cast<size_t>(column) *
                static_cast<size_t>(leading_dimension);
        for (int row = 0; row < rows; ++row) {
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(values[row]));
            std::memcpy(&bits, values + row, sizeof(bits));
            const uint64_t index =
                static_cast<uint64_t>(column) *
                    static_cast<uint64_t>(rows)
                + static_cast<uint64_t>(row);
            const uint64_t keyed = bits ^ (
                (index + 0x6a09e667f3bcc909ULL)
                * 0x9e3779b97f4a7c15ULL
            );
            const uint64_t mixed = keyed * 0xbf58476d1ce4e5b9ULL;
            const unsigned int shift = static_cast<unsigned int>(index & 63U);
            const uint64_t rotated = shift == 0U
                ? mixed
                : (mixed << shift) | (mixed >> (64U - shift));
            xor_hash ^= rotated;
            sum_hash += (keyed ^ rotated) * 0x94d049bb133111ebULL;
        }
    }
    return MatrixFingerprint{xor_hash, sum_hash};
}

MatrixFingerprint fingerprint_row_major_matrix(
    int rows, int columns, const double* matrix, int leading_dimension,
    int requested_threads
) {
    uint64_t xor_hash = 0;
    uint64_t sum_hash = 0;
    const int threads = std::max(1, std::min(requested_threads, rows));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) \
        reduction(^:xor_hash) reduction(+:sum_hash)
#endif
    for (int row = 0; row < rows; ++row) {
        const double* values = matrix
            + static_cast<size_t>(row)
                * static_cast<size_t>(leading_dimension);
        for (int column = 0; column < columns; ++column) {
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(values[column]));
            std::memcpy(&bits, values + column, sizeof(bits));
            const uint64_t index =
                static_cast<uint64_t>(row) * static_cast<uint64_t>(columns)
                + static_cast<uint64_t>(column);
            const uint64_t keyed = bits ^ (
                (index + 0x6a09e667f3bcc909ULL)
                * 0x9e3779b97f4a7c15ULL
            );
            const uint64_t mixed = keyed * 0xbf58476d1ce4e5b9ULL;
            const unsigned int shift = static_cast<unsigned int>(index & 63U);
            const uint64_t rotated = shift == 0U
                ? mixed
                : (mixed << shift) | (mixed >> (64U - shift));
            xor_hash ^= rotated;
            sum_hash += (keyed ^ rotated) * 0x94d049bb133111ebULL;
        }
    }
    return MatrixFingerprint{xor_hash, sum_hash};
}

void copy_col_major_matrix(int rows, int columns,
                           const double* source, int source_ld,
                           double* destination, int destination_ld,
                           int requested_threads) {
    const int threads = std::max(1, std::min(requested_threads, columns));
    const size_t column_bytes = checked_mul(
        static_cast<size_t>(rows), sizeof(double),
        "protected GEMM operand column"
    );
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int column = 0; column < columns; ++column) {
        std::memcpy(
            destination
                + static_cast<size_t>(column)
                    * static_cast<size_t>(destination_ld),
            source
                + static_cast<size_t>(column)
                    * static_cast<size_t>(source_ld),
            column_bytes
        );
    }
}

std::vector<double> gemm_integrity_coefficients_row_major(int rows) {
    const std::vector<double> column_major =
        gemm_integrity_coefficients(rows);
    std::vector<double> row_major(
        checked_mul(
            static_cast<size_t>(rows),
            static_cast<size_t>(kGemmIntegrityChecks),
            "row-major GEMM integrity coefficients"
        )
    );
    for (int row = 0; row < rows; ++row) {
        for (int check = 0; check < kGemmIntegrityChecks; ++check) {
            row_major[
                static_cast<size_t>(row)
                    * static_cast<size_t>(kGemmIntegrityChecks)
                + static_cast<size_t>(check)
            ] = column_major[
                static_cast<size_t>(check) * static_cast<size_t>(rows)
                + static_cast<size_t>(row)
            ];
        }
    }
    return row_major;
}

bool gemm_vendor_value_materially_disagrees(
    double vendor,
    double deterministic,
    long double absolute_product_sum,
    int reduction_terms
) {
    if (!std::isfinite(vendor) || !std::isfinite(deterministic)
        || !std::isfinite(absolute_product_sum)) {
        return true;
    }
    // Both values are binary64 reductions of the same products, but their
    // summation trees differ.  Bound their ordinary forward-error separation
    // by the absolute product sum rather than by the possibly cancelled
    // result.  The generous multiplier is still many orders of magnitude
    // below the column-scale failures that motivated the integrity layer.
    const double relative_bound = 64.0 * (
        2.0 * floating_point_gamma(reduction_terms)
        + std::numeric_limits<double>::epsilon()
    );
    const double scale = std::max(
        1.0,
        std::max(
            static_cast<double>(absolute_product_sum),
            std::max(std::abs(vendor), std::abs(deterministic))
        )
    );
    return std::abs(vendor - deterministic) > relative_bound * scale;
}

int64_t dgemm_tn_checked(int m, int n, int k,
                         const double* a, int lda,
                         const double* b, int ldb,
                         double* c, int ldc,
                         int requested_threads,
                         const MatrixFingerprint* immutable_b_fingerprint = nullptr,
                         bool immutable_b_is_read_only = false,
                         GemmIntegrityResolution* resolution = nullptr) {
    std::vector<double> coefficients = gemm_integrity_coefficients(m);
    std::vector<double> projected(
        checked_mul(
            static_cast<size_t>(k),
            static_cast<size_t>(kGemmIntegrityChecks),
            "GEMM integrity projection"
        )
    );
    const size_t check_elements = checked_mul(
        static_cast<size_t>(kGemmIntegrityChecks),
        static_cast<size_t>(n),
        "GEMM integrity checks"
    );
    std::vector<double> expected(check_elements);
    std::vector<double> observed(check_elements);
    dgemm_nn_tiled(
        k, kGemmIntegrityChecks, m,
        a, lda, coefficients.data(), m,
        projected.data(), k, requested_threads
    );
    std::unique_ptr<double[]> protected_b;
    const double* stable_b = b;
    int stable_ldb = ldb;
    if (immutable_b_fingerprint == nullptr && !immutable_b_is_read_only) {
        const size_t operand_b = checked_mul(
            static_cast<size_t>(k), static_cast<size_t>(n),
            "protected TN operand B"
        );
        protected_b.reset(new double[operand_b]);
        copy_col_major_matrix(
            k, n, b, ldb, protected_b.get(), k, requested_threads
        );
        stable_b = protected_b.get();
        stable_ldb = k;
    }
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, k,
        projected.data(), k, stable_b, stable_ldb,
        expected.data(), kGemmIntegrityChecks, requested_threads
    );
    const MatrixFingerprint a_before = fingerprint_col_major_matrix(
        k, m, a, lda, requested_threads
    );
    // The non-reconstructable right operand is snapshotted. The decoded left
    // operand is protected by a bitwise 128-bit fingerprint; its caller can
    // discard and re-decode the block on the rare affected-host mutation.
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_tn_fixed_vendor(
        m, n, k, a, lda, stable_b, stable_ldb, c, ldc,
        requested_threads
    );
#else
    dgemm_tn_vendor_observed(
        m, n, k, a, lda, stable_b, stable_ldb, c, ldc,
        requested_threads
    );
#endif
    if (immutable_b_fingerprint != nullptr &&
        !(fingerprint_col_major_matrix(
              k, n, stable_b, stable_ldb, requested_threads
          ) == *immutable_b_fingerprint)) {
        throw std::runtime_error(
            "Vendor BLAS altered an immutable protected GxE GEMM input"
        );
    }
    if (!(a_before == fingerprint_col_major_matrix(
              k, m, a, lda, requested_threads))) {
        throw RetryableGemmInputMutation();
    }
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, m,
        coefficients.data(), m, c, ldc,
        observed.data(), kGemmIntegrityChecks, requested_threads
    );
    const auto column_disagrees = [&](int column) {
        for (int check = 0; check < kGemmIntegrityChecks; ++check) {
            const size_t index =
                static_cast<size_t>(column) *
                    static_cast<size_t>(kGemmIntegrityChecks)
                + static_cast<size_t>(check);
            if (gemm_integrity_disagrees(
                    expected[index], observed[index], m, k)) {
                return true;
            }
        }
        return false;
    };
    // A factored checksum and a vendor GEMM use different legal floating-point
    // reduction trees.  Recompute every flagged range deterministically, then
    // distinguish ordinary roundoff from a material vendor error using a
    // forward-error bound based on a conservative upper bound for each dot
    // product's absolute product sum.
    GemmIntegrityResolution observed_resolution;
    std::vector<unsigned char> flagged(static_cast<size_t>(n), 0U);
    bool any_flagged = false;
    for (int column = 0; column < n; ++column) {
        flagged[static_cast<size_t>(column)] =
            column_disagrees(column) ? 1U : 0U;
        any_flagged = any_flagged
            || flagged[static_cast<size_t>(column)] != 0U;
    }
    if (!any_flagged) {
        if (resolution != nullptr) *resolution = observed_resolution;
        return 0;
    }
    double maximum_absolute_a = 0.0;
    const int a_threads = std::max(1, std::min(requested_threads, m));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(a_threads) \
        reduction(max:maximum_absolute_a)
#endif
    for (int column = 0; column < m; ++column) {
        const double* values = a
            + static_cast<size_t>(column) * static_cast<size_t>(lda);
        for (int reduction = 0; reduction < k; ++reduction) {
            maximum_absolute_a = std::max(
                maximum_absolute_a, std::abs(values[reduction])
            );
        }
    }
    std::vector<long double> right_absolute_sums(
        static_cast<size_t>(n), 0.0L
    );
    const int b_threads = std::max(1, std::min(requested_threads, n));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(b_threads)
#endif
    for (int column = 0; column < n; ++column) {
        if (flagged[static_cast<size_t>(column)] == 0U) continue;
        const double* values = stable_b
            + static_cast<size_t>(column) * static_cast<size_t>(stable_ldb);
        long double sum = 0.0L;
        for (int reduction = 0; reduction < k; ++reduction) {
            sum += std::abs(static_cast<long double>(values[reduction]));
        }
        right_absolute_sums[static_cast<size_t>(column)] = sum;
    }
    for (int column = 0; column < n;) {
        if (flagged[static_cast<size_t>(column)] == 0U) {
            ++column;
            continue;
        }
        const int first = column;
        do {
            ++observed_resolution.checksum_recomputed_columns;
            ++column;
        } while (column < n && flagged[static_cast<size_t>(column)] != 0U);
        const int count = column - first;
        std::vector<double> deterministic(
            checked_mul(
                static_cast<size_t>(m), static_cast<size_t>(count),
                "deterministic TN integrity reference"
            )
        );
        dgemm_tn_tiled(
            m, count, k, a, lda,
            stable_b
                + static_cast<size_t>(first)
                    * static_cast<size_t>(stable_ldb),
            stable_ldb,
            deterministic.data(), m, requested_threads
        );
        for (int local_column = 0; local_column < count; ++local_column) {
            bool material = false;
            const int source_column = first + local_column;
            const long double absolute_product_sum =
                static_cast<long double>(maximum_absolute_a)
                * right_absolute_sums[static_cast<size_t>(source_column)];
            for (int row = 0; row < m && !material; ++row) {
                material = gemm_vendor_value_materially_disagrees(
                    c[
                        static_cast<size_t>(row)
                            + static_cast<size_t>(source_column)
                                * static_cast<size_t>(ldc)
                    ],
                    deterministic[
                        static_cast<size_t>(row)
                            + static_cast<size_t>(local_column)
                                * static_cast<size_t>(m)
                    ],
                    absolute_product_sum,
                    k
                );
            }
            observed_resolution.materially_repaired_columns += material ? 1 : 0;
        }
        copy_col_major_matrix(
            m, count, deterministic.data(), m,
            c + static_cast<size_t>(first) * static_cast<size_t>(ldc),
            ldc, requested_threads
        );
    }
    if (resolution != nullptr) {
        *resolution = observed_resolution;
    }
    return observed_resolution.materially_repaired_columns;
}

int64_t dgemm_nn_checked(int m, int n, int k,
                         const double* a, int lda,
                         const double* b, int ldb,
                         double* c, int ldc,
                         int requested_threads,
                         GemmIntegrityResolution* resolution = nullptr) {
    std::vector<double> coefficients = gemm_integrity_coefficients(m);
    std::vector<double> projected(
        checked_mul(
            static_cast<size_t>(k),
            static_cast<size_t>(kGemmIntegrityChecks),
            "GEMM integrity projection"
        )
    );
    const size_t check_elements = checked_mul(
        static_cast<size_t>(kGemmIntegrityChecks),
        static_cast<size_t>(n),
        "GEMM integrity checks"
    );
    std::vector<double> expected(check_elements);
    std::vector<double> observed(check_elements);
    dgemm_tn_tiled(
        k, kGemmIntegrityChecks, m,
        a, lda, coefficients.data(), m,
        projected.data(), k, requested_threads
    );
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, k,
        projected.data(), k, b, ldb,
        expected.data(), kGemmIntegrityChecks, requested_threads
    );
    const size_t operand_b = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected NN operand B"
    );
    const size_t operand_b_bytes = checked_mul(
        operand_b, sizeof(double), "protected NN operand B bytes"
    );
    // Keep the vendor-visible copy out of reusable malloc arenas.  In an
    // explicit early-NUMA run the mapping is bound before this first touch and
    // exhaustively verified before CBLAS; legacy callers retain the isolated,
    // read-only snapshot without claiming the NUMA acceptance contract.
    NativeIntegritySnapshotMapping protected_b(operand_b_bytes);
    copy_col_major_matrix(
        k, n, b, ldb, protected_b.data(), k, requested_threads
    );
    const MatrixFingerprint a_before = fingerprint_col_major_matrix(
        m, k, a, lda, requested_threads
    );
    protected_b.verify_before_vendor();
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_nn_fixed_vendor(
        m, n, k, a, lda, protected_b.data(), k, c, ldc,
        requested_threads, 1.0, 0.0, &protected_b.evidence()
    );
#else
    dgemm_nn_vendor_observed(
        m, n, k, a, lda, protected_b.data(), k, c, ldc,
        requested_threads, 1.0, 0.0, &protected_b.evidence()
    );
#endif
    if (!(a_before == fingerprint_col_major_matrix(
              m, k, a, lda, requested_threads))) {
        throw RetryableGemmInputMutation();
    }
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, m,
        coefficients.data(), m, c, ldc,
        observed.data(), kGemmIntegrityChecks, requested_threads
    );
    const auto column_disagrees = [&](int column) {
        for (int check = 0; check < kGemmIntegrityChecks; ++check) {
            const size_t index =
                static_cast<size_t>(column) *
                    static_cast<size_t>(kGemmIntegrityChecks)
                + static_cast<size_t>(check);
            if (gemm_integrity_disagrees(
                    expected[index], observed[index], m, k)) {
                return true;
            }
        }
        return false;
    };
    GemmIntegrityResolution observed_resolution;
    std::vector<unsigned char> flagged(static_cast<size_t>(n), 0U);
    bool any_flagged = false;
    for (int column = 0; column < n; ++column) {
        flagged[static_cast<size_t>(column)] =
            column_disagrees(column) ? 1U : 0U;
        any_flagged = any_flagged
            || flagged[static_cast<size_t>(column)] != 0U;
    }
    if (!any_flagged) {
        if (resolution != nullptr) *resolution = observed_resolution;
        return 0;
    }

    // A single scan supplies a conservative forward-error scale for every
    // flagged dot product: sum(|A_rj B_jc|) is bounded by
    // max(|A|) * sum(|B_c|). This avoids an O(m*n*k) verification pass while
    // still separating ordinary reduction-order roundoff from material
    // corruption before the deterministic replacement is published.
    double maximum_absolute_a = 0.0;
    const int a_threads = std::max(1, std::min(requested_threads, k));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(a_threads) \
        reduction(max:maximum_absolute_a)
#endif
    for (int reduction = 0; reduction < k; ++reduction) {
        const double* column = a
            + static_cast<size_t>(reduction) * static_cast<size_t>(lda);
        for (int row = 0; row < m; ++row) {
            maximum_absolute_a = std::max(
                maximum_absolute_a, std::abs(column[row])
            );
        }
    }
    std::vector<long double> right_absolute_sums(
        static_cast<size_t>(n), 0.0L
    );
    const int b_threads = std::max(1, std::min(requested_threads, n));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(b_threads)
#endif
    for (int column = 0; column < n; ++column) {
        if (flagged[static_cast<size_t>(column)] == 0U) continue;
        const double* values = protected_b.data()
            + static_cast<size_t>(column) * static_cast<size_t>(k);
        long double sum = 0.0L;
        for (int reduction = 0; reduction < k; ++reduction) {
            sum += std::abs(static_cast<long double>(values[reduction]));
        }
        right_absolute_sums[static_cast<size_t>(column)] = sum;
    }

    for (int column = 0; column < n;) {
        if (flagged[static_cast<size_t>(column)] == 0U) {
            ++column;
            continue;
        }
        const int first = column;
        do {
            ++observed_resolution.checksum_recomputed_columns;
            ++column;
        } while (column < n && flagged[static_cast<size_t>(column)] != 0U);
        const int count = column - first;
        std::vector<double> deterministic(checked_mul(
            static_cast<size_t>(m), static_cast<size_t>(count),
            "deterministic NN integrity reference"
        ));
        dgemm_nn_tiled(
            m, count, k, a, lda,
            protected_b.data()
                + static_cast<size_t>(first) * static_cast<size_t>(k),
            k,
            deterministic.data(), m, requested_threads
        );
        for (int local_column = 0; local_column < count; ++local_column) {
            const int source_column = first + local_column;
            const long double absolute_product_sum =
                static_cast<long double>(maximum_absolute_a)
                * right_absolute_sums[static_cast<size_t>(source_column)];
            bool material = false;
            for (int row = 0; row < m && !material; ++row) {
                material = gemm_vendor_value_materially_disagrees(
                    c[
                        static_cast<size_t>(row)
                            + static_cast<size_t>(source_column)
                                * static_cast<size_t>(ldc)
                    ],
                    deterministic[
                        static_cast<size_t>(row)
                            + static_cast<size_t>(local_column)
                                * static_cast<size_t>(m)
                    ],
                    absolute_product_sum,
                    k
                );
            }
            observed_resolution.materially_repaired_columns +=
                material ? 1 : 0;
        }
        copy_col_major_matrix(
            m, count, deterministic.data(), m,
            c + static_cast<size_t>(first) * static_cast<size_t>(ldc),
            ldc, requested_threads
        );
    }
    if (resolution != nullptr) {
        *resolution = observed_resolution;
    }
    return observed_resolution.materially_repaired_columns;
}

// This evidence path is intentionally separate from dgemm_nn_checked().  It is
// exposed only in integrity builds through a private test entry point, so the
// production protected_matmul_nn call has no diagnostic branches, allocations,
// fingerprints, or reference work added to its hot path.
struct NnIntegrityCheckDiagnostic {
    int column = -1;
    int check = -1;
    double expected = 0.0;
    double observed = 0.0;
    double difference = 0.0;
    double absolute_difference = 0.0;
    double current_relative_bound = 0.0;
    double current_tolerance = 0.0;
    double direct_absolute_product_sum = 0.0;
    double factored_expected_absolute_sum = 0.0;
    double observed_checksum_absolute_sum = 0.0;
    double projection_roundoff_component = 0.0;
    double expected_reduction_roundoff_component = 0.0;
    double vendor_product_roundoff_component = 0.0;
    double observed_reduction_roundoff_component = 0.0;
    double cancellation_aware_bound = 0.0;
    bool cancellation_aware_disagrees = false;
};

struct NnIntegrityColumnDiagnostic {
    int column = -1;
    std::string classification = "unclassified";
    int64_t raw_vendor_nonfinite_count = 0;
    int64_t deterministic_tiled_nonfinite_count = 0;
    int64_t long_double_reference_nonfinite_count = 0;
    int64_t vendor_tiled_unequal_count = 0;
    int64_t vendor_reference_unequal_count = 0;
    int64_t tiled_reference_unequal_count = 0;
    int64_t vendor_rows_outside_forward_error_bound = 0;
    int64_t tiled_rows_outside_forward_error_bound = 0;
    double max_abs_vendor_minus_tiled = 0.0;
    double max_abs_vendor_minus_long_double = 0.0;
    double max_abs_tiled_minus_long_double = 0.0;
    double max_vendor_forward_error_ratio = 0.0;
    double max_tiled_forward_error_ratio = 0.0;
};

struct NnIntegrityDiagnostic {
    int m = 0;
    int n = 0;
    int k = 0;
    bool integrity_check_eligible = false;
    bool diagnostic_executed = false;
    std::string classification = "not_executed";
    std::vector<int> flagged_columns;
    std::vector<std::pair<int, int>> flagged_check_ids;
    std::vector<NnIntegrityCheckDiagnostic> flagged_checks;
    std::vector<int> captured_flagged_columns;
    std::vector<NnIntegrityColumnDiagnostic> column_comparisons;
    std::vector<double> raw_vendor_columns;
    std::vector<double> deterministic_tiled_columns;
    std::vector<double> long_double_reference_columns;
    std::vector<std::string> long_double_reference_decimal_columns;
    MatrixFingerprint original_b_initial{};
    MatrixFingerprint original_b_after_expected{};
    MatrixFingerprint original_b_after_copy{};
    MatrixFingerprint original_b_after_vendor{};
    MatrixFingerprint original_b_after_deterministic{};
    MatrixFingerprint original_b_after_reference{};
    MatrixFingerprint protected_b_before_vendor{};
    MatrixFingerprint protected_b_after_vendor{};
    bool fault_injection_enabled = false;
    int fault_injection_row = -1;
    int fault_injection_column = -1;
    double fault_injection_delta = 0.0;
};

constexpr size_t kNnIntegrityDiagnosticColumnLimit = 16;

double nn_integrity_current_relative_bound(int m, int k) {
    return 32.0 * (
        2.0 * floating_point_gamma(m)
        + floating_point_gamma(k)
        + std::numeric_limits<double>::epsilon()
    );
}

std::string long_double_decimal(long double value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<long double>::max_digits10)
           << value;
    return stream.str();
}

void update_diagnostic_max(double& current, long double candidate) {
    if (!std::isfinite(candidate)) {
        current = std::numeric_limits<double>::infinity();
        return;
    }
    current = std::max(current, static_cast<double>(candidate));
}

NnIntegrityDiagnostic dgemm_nn_checked_diagnostic(
    int m, int n, int k,
    const double* a, int lda,
    const double* b, int ldb,
    double* c, int ldc,
    int requested_threads,
    int fault_injection_row,
    int fault_injection_column,
    double fault_injection_delta
) {
    NnIntegrityDiagnostic diagnostic;
    diagnostic.m = m;
    diagnostic.n = n;
    diagnostic.k = k;
    diagnostic.integrity_check_eligible = true;
    diagnostic.diagnostic_executed = true;
    diagnostic.fault_injection_enabled = fault_injection_row >= 0;
    diagnostic.fault_injection_row = fault_injection_row;
    diagnostic.fault_injection_column = fault_injection_column;
    diagnostic.fault_injection_delta = fault_injection_delta;

    std::vector<double> coefficients = gemm_integrity_coefficients(m);
    std::vector<double> projected(
        checked_mul(
            static_cast<size_t>(k),
            static_cast<size_t>(kGemmIntegrityChecks),
            "diagnostic GEMM integrity projection"
        )
    );
    const size_t check_elements = checked_mul(
        static_cast<size_t>(kGemmIntegrityChecks),
        static_cast<size_t>(n),
        "diagnostic GEMM integrity checks"
    );
    std::vector<double> expected(check_elements);
    std::vector<double> observed(check_elements);

    diagnostic.original_b_initial = fingerprint_col_major_matrix(
        k, n, b, ldb, requested_threads
    );
    dgemm_tn_tiled(
        k, kGemmIntegrityChecks, m,
        a, lda, coefficients.data(), m,
        projected.data(), k, requested_threads
    );
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, k,
        projected.data(), k, b, ldb,
        expected.data(), kGemmIntegrityChecks, requested_threads
    );
    diagnostic.original_b_after_expected = fingerprint_col_major_matrix(
        k, n, b, ldb, requested_threads
    );

    const size_t operand_b = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "diagnostic protected NN operand B"
    );
    std::unique_ptr<double[]> protected_b(new double[operand_b]);
    copy_col_major_matrix(
        k, n, b, ldb, protected_b.get(), k, requested_threads
    );
    diagnostic.original_b_after_copy = fingerprint_col_major_matrix(
        k, n, b, ldb, requested_threads
    );
    diagnostic.protected_b_before_vendor = fingerprint_col_major_matrix(
        k, n, protected_b.get(), k, requested_threads
    );

    const MatrixFingerprint a_before = fingerprint_col_major_matrix(
        m, k, a, lda, requested_threads
    );
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_nn_fixed_vendor(
        m, n, k, a, lda, protected_b.get(), k, c, ldc,
        requested_threads
    );
#else
    dgemm_nn_vendor_observed(
        m, n, k, a, lda, protected_b.get(), k, c, ldc,
        requested_threads
    );
#endif
    diagnostic.original_b_after_vendor = fingerprint_col_major_matrix(
        k, n, b, ldb, requested_threads
    );
    diagnostic.protected_b_after_vendor = fingerprint_col_major_matrix(
        k, n, protected_b.get(), k, requested_threads
    );
    if (diagnostic.fault_injection_enabled) {
        c[static_cast<size_t>(fault_injection_column)
              * static_cast<size_t>(ldc)
          + static_cast<size_t>(fault_injection_row)] += fault_injection_delta;
    }
    if (!(a_before == fingerprint_col_major_matrix(
              m, k, a, lda, requested_threads))) {
        throw RetryableGemmInputMutation();
    }

    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, m,
        coefficients.data(), m, c, ldc,
        observed.data(), kGemmIntegrityChecks, requested_threads
    );

    std::vector<unsigned char> flagged_column_mask(
        static_cast<size_t>(n), 0U
    );
    const double current_relative_bound =
        nn_integrity_current_relative_bound(m, k);
    for (int column = 0; column < n; ++column) {
        for (int check = 0; check < kGemmIntegrityChecks; ++check) {
            const size_t index =
                static_cast<size_t>(column)
                    * static_cast<size_t>(kGemmIntegrityChecks)
                + static_cast<size_t>(check);
            if (!gemm_integrity_disagrees(
                    expected[index], observed[index], m, k)) {
                continue;
            }
            flagged_column_mask[static_cast<size_t>(column)] = 1U;
            NnIntegrityCheckDiagnostic record;
            record.column = column;
            record.check = check;
            record.expected = expected[index];
            record.observed = observed[index];
            record.difference = expected[index] - observed[index];
            record.absolute_difference = std::abs(record.difference);
            record.current_relative_bound = current_relative_bound;
            record.current_tolerance = current_relative_bound * std::max(
                1.0,
                std::max(std::abs(expected[index]), std::abs(observed[index]))
            );
            diagnostic.flagged_check_ids.emplace_back(column, check);
            diagnostic.flagged_checks.push_back(record);
        }
    }
    for (int column = 0; column < n; ++column) {
        if (flagged_column_mask[static_cast<size_t>(column)] != 0U) {
            diagnostic.flagged_columns.push_back(column);
            if (diagnostic.captured_flagged_columns.size()
                < kNnIntegrityDiagnosticColumnLimit) {
                diagnostic.captured_flagged_columns.push_back(column);
            }
        }
    }
    diagnostic.flagged_checks.erase(
        std::remove_if(
            diagnostic.flagged_checks.begin(),
            diagnostic.flagged_checks.end(),
            [&](const NnIntegrityCheckDiagnostic& record) {
                return std::find(
                    diagnostic.captured_flagged_columns.begin(),
                    diagnostic.captured_flagged_columns.end(),
                    record.column
                ) == diagnostic.captured_flagged_columns.end();
            }
        ),
        diagnostic.flagged_checks.end()
    );

    if (diagnostic.flagged_columns.empty()) {
        diagnostic.original_b_after_deterministic =
            diagnostic.original_b_after_vendor;
        diagnostic.original_b_after_reference =
            diagnostic.original_b_after_vendor;
        const bool original_b_unchanged =
            diagnostic.original_b_initial ==
                diagnostic.original_b_after_expected
            && diagnostic.original_b_initial == diagnostic.original_b_after_copy
            && diagnostic.original_b_initial ==
                diagnostic.original_b_after_vendor;
        const bool protected_b_unchanged =
            diagnostic.protected_b_before_vendor ==
                diagnostic.protected_b_after_vendor;
        const bool protected_b_matches_original =
            diagnostic.original_b_after_copy ==
                diagnostic.protected_b_before_vendor
            && diagnostic.original_b_after_vendor ==
                diagnostic.protected_b_after_vendor;
        diagnostic.classification =
            original_b_unchanged && protected_b_unchanged
                && protected_b_matches_original
            ? "no_current_gate_flags" : "input_fingerprint_changed";
        return diagnostic;
    }

    // Cancellation-aware terms are diagnostic evidence only.  Construct them
    // after the production current gate has flagged at least one check so they
    // cannot perturb the vendor call being classified.
    std::vector<double> absolute_projected(
        checked_mul(
            static_cast<size_t>(k),
            static_cast<size_t>(kGemmIntegrityChecks),
            "diagnostic absolute integrity projection"
        )
    );
    for (int check = 0; check < kGemmIntegrityChecks; ++check) {
        for (int reduction = 0; reduction < k; ++reduction) {
            long double sum = 0.0L;
            for (int row = 0; row < m; ++row) {
                sum += std::abs(
                    static_cast<long double>(coefficients[
                        static_cast<size_t>(check) * static_cast<size_t>(m)
                        + static_cast<size_t>(row)
                    ]) * static_cast<long double>(a[
                        static_cast<size_t>(reduction)
                            * static_cast<size_t>(lda)
                        + static_cast<size_t>(row)
                    ])
                );
            }
            absolute_projected[
                static_cast<size_t>(check) * static_cast<size_t>(k)
                + static_cast<size_t>(reduction)
            ] = static_cast<double>(sum);
        }
    }

    const double gamma_m = floating_point_gamma(m);
    const double gamma_k = floating_point_gamma(k);
    for (auto& record : diagnostic.flagged_checks) {
        long double direct_absolute_product_sum = 0.0L;
        long double factored_expected_absolute_sum = 0.0L;
        const double* b_column = b
            + static_cast<size_t>(record.column) * static_cast<size_t>(ldb);
        for (int reduction = 0; reduction < k; ++reduction) {
            const long double absolute_b = std::abs(
                static_cast<long double>(b_column[reduction])
            );
            direct_absolute_product_sum += absolute_b *
                static_cast<long double>(absolute_projected[
                    static_cast<size_t>(record.check) * static_cast<size_t>(k)
                    + static_cast<size_t>(reduction)
                ]);
            factored_expected_absolute_sum += std::abs(
                static_cast<long double>(projected[
                    static_cast<size_t>(record.check) * static_cast<size_t>(k)
                    + static_cast<size_t>(reduction)
                ]) * static_cast<long double>(b_column[reduction])
            );
        }
        long double observed_checksum_absolute_sum = 0.0L;
        const double* c_column = c
            + static_cast<size_t>(record.column) * static_cast<size_t>(ldc);
        for (int row = 0; row < m; ++row) {
            observed_checksum_absolute_sum += std::abs(
                static_cast<long double>(coefficients[
                    static_cast<size_t>(record.check) * static_cast<size_t>(m)
                    + static_cast<size_t>(row)
                ]) * static_cast<long double>(c_column[row])
            );
        }
        record.direct_absolute_product_sum =
            static_cast<double>(direct_absolute_product_sum);
        record.factored_expected_absolute_sum =
            static_cast<double>(factored_expected_absolute_sum);
        record.observed_checksum_absolute_sum =
            static_cast<double>(observed_checksum_absolute_sum);
        record.projection_roundoff_component = static_cast<double>(
            static_cast<long double>(gamma_m)
                * (1.0L + static_cast<long double>(gamma_k))
                * direct_absolute_product_sum
        );
        record.expected_reduction_roundoff_component = static_cast<double>(
            static_cast<long double>(gamma_k)
                * factored_expected_absolute_sum
        );
        record.vendor_product_roundoff_component = static_cast<double>(
            static_cast<long double>(gamma_k)
                * (1.0L + static_cast<long double>(gamma_m))
                * direct_absolute_product_sum
        );
        record.observed_reduction_roundoff_component = static_cast<double>(
            static_cast<long double>(gamma_m)
                * observed_checksum_absolute_sum
        );
        const long double component_sum =
            static_cast<long double>(record.projection_roundoff_component)
            + static_cast<long double>(
                record.expected_reduction_roundoff_component)
            + static_cast<long double>(
                record.vendor_product_roundoff_component)
            + static_cast<long double>(
                record.observed_reduction_roundoff_component);
        const long double epsilon_scale = std::max(
            1.0L,
            std::max(
                direct_absolute_product_sum,
                std::max(
                    factored_expected_absolute_sum,
                    observed_checksum_absolute_sum
                )
            )
        );
        record.cancellation_aware_bound = static_cast<double>(32.0L * (
            component_sum
            + static_cast<long double>(
                std::numeric_limits<double>::epsilon()) * epsilon_scale
        ));
        record.cancellation_aware_disagrees =
            !std::isfinite(record.expected)
            || !std::isfinite(record.observed)
            || record.absolute_difference > record.cancellation_aware_bound;
    }

    const size_t flagged_count = diagnostic.captured_flagged_columns.size();
    const size_t flagged_elements = checked_mul(
        static_cast<size_t>(m), flagged_count,
        "diagnostic flagged NN columns"
    );
    diagnostic.raw_vendor_columns.resize(flagged_elements);
    diagnostic.deterministic_tiled_columns.resize(flagged_elements);
    diagnostic.long_double_reference_columns.resize(flagged_elements);
    diagnostic.long_double_reference_decimal_columns.resize(flagged_elements);

    for (size_t selected = 0; selected < flagged_count; ++selected) {
        const int column = diagnostic.captured_flagged_columns[selected];
        std::memcpy(
            diagnostic.raw_vendor_columns.data()
                + selected * static_cast<size_t>(m),
            c + static_cast<size_t>(column) * static_cast<size_t>(ldc),
            static_cast<size_t>(m) * sizeof(double)
        );
        dgemm_nn_tiled(
            m, 1, k,
            a, lda,
            b + static_cast<size_t>(column) * static_cast<size_t>(ldb),
            ldb,
            diagnostic.deterministic_tiled_columns.data()
                + selected * static_cast<size_t>(m),
            m, requested_threads
        );
    }
    for (size_t selected = flagged_count;
         selected < diagnostic.flagged_columns.size(); ++selected) {
        const int column = diagnostic.flagged_columns[selected];
        dgemm_nn_tiled(
            m, 1, k,
            a, lda,
            b + static_cast<size_t>(column) * static_cast<size_t>(ldb),
            ldb,
            c + static_cast<size_t>(column) * static_cast<size_t>(ldc),
            ldc, requested_threads
        );
    }
    diagnostic.original_b_after_deterministic = fingerprint_col_major_matrix(
        k, n, b, ldb, requested_threads
    );

    const long double long_double_product =
        static_cast<long double>(std::max(1, k))
        * std::numeric_limits<long double>::epsilon();
    const long double gamma_long_double = long_double_product >= 0.5L
        ? std::numeric_limits<long double>::infinity()
        : long_double_product / (1.0L - long_double_product);
    bool every_column_is_cancellation_false_positive = true;
    bool any_vendor_corruption_classification = false;
    bool any_unclassified_column = false;
    for (size_t selected = 0; selected < flagged_count; ++selected) {
        const int column = diagnostic.captured_flagged_columns[selected];
        const double* b_column =
            b + static_cast<size_t>(column) * static_cast<size_t>(ldb);
        NnIntegrityColumnDiagnostic comparison;
        comparison.column = column;
        for (int row = 0; row < m; ++row) {
            long double reference = 0.0L;
            long double direct_absolute_sum = 0.0L;
            for (int reduction = 0; reduction < k; ++reduction) {
                const long double product =
                    static_cast<long double>(a[
                        static_cast<size_t>(reduction)
                            * static_cast<size_t>(lda)
                        + static_cast<size_t>(row)
                    ]) * static_cast<long double>(b_column[reduction]);
                reference += product;
                direct_absolute_sum += std::abs(product);
            }
            const size_t output_index =
                selected * static_cast<size_t>(m)
                + static_cast<size_t>(row);
            const double reference_as_double = static_cast<double>(reference);
            diagnostic.long_double_reference_columns[output_index] =
                reference_as_double;
            diagnostic.long_double_reference_decimal_columns[output_index] =
                long_double_decimal(reference);

            const double raw = diagnostic.raw_vendor_columns[output_index];
            const double tiled =
                diagnostic.deterministic_tiled_columns[output_index];
            comparison.raw_vendor_nonfinite_count +=
                std::isfinite(raw) ? 0 : 1;
            comparison.deterministic_tiled_nonfinite_count +=
                std::isfinite(tiled) ? 0 : 1;
            comparison.long_double_reference_nonfinite_count +=
                std::isfinite(reference) ? 0 : 1;
            comparison.vendor_tiled_unequal_count += raw == tiled ? 0 : 1;
            comparison.vendor_reference_unequal_count +=
                raw == reference_as_double ? 0 : 1;
            comparison.tiled_reference_unequal_count +=
                tiled == reference_as_double ? 0 : 1;

            const long double vendor_tiled_error = std::abs(
                static_cast<long double>(raw)
                - static_cast<long double>(tiled)
            );
            const long double vendor_reference_error = std::abs(
                static_cast<long double>(raw) - reference
            );
            const long double tiled_reference_error = std::abs(
                static_cast<long double>(tiled) - reference
            );
            update_diagnostic_max(
                comparison.max_abs_vendor_minus_tiled,
                vendor_tiled_error
            );
            update_diagnostic_max(
                comparison.max_abs_vendor_minus_long_double,
                vendor_reference_error
            );
            update_diagnostic_max(
                comparison.max_abs_tiled_minus_long_double,
                tiled_reference_error
            );

            const long double forward_error_bound =
                32.0L * static_cast<long double>(gamma_k)
                    * direct_absolute_sum
                + gamma_long_double * direct_absolute_sum
                + static_cast<long double>(
                    std::numeric_limits<double>::epsilon())
                    * std::max(1.0L, std::abs(reference));
            const bool vendor_outside =
                !std::isfinite(vendor_reference_error)
                || vendor_reference_error > forward_error_bound;
            const bool tiled_outside =
                !std::isfinite(tiled_reference_error)
                || tiled_reference_error > forward_error_bound;
            comparison.vendor_rows_outside_forward_error_bound +=
                vendor_outside ? 1 : 0;
            comparison.tiled_rows_outside_forward_error_bound +=
                tiled_outside ? 1 : 0;
            if (forward_error_bound > 0.0L) {
                update_diagnostic_max(
                    comparison.max_vendor_forward_error_ratio,
                    vendor_reference_error / forward_error_bound
                );
                update_diagnostic_max(
                    comparison.max_tiled_forward_error_ratio,
                    tiled_reference_error / forward_error_bound
                );
            }
        }

        bool all_column_checks_within_cancellation_bound = true;
        for (const auto& record : diagnostic.flagged_checks) {
            if (record.column == column) {
                all_column_checks_within_cancellation_bound =
                    all_column_checks_within_cancellation_bound
                    && !record.cancellation_aware_disagrees;
            }
        }
        if (comparison.vendor_rows_outside_forward_error_bound > 0
            && comparison.tiled_rows_outside_forward_error_bound == 0) {
            comparison.classification =
                "vendor_result_outside_forward_error_bound";
        } else if (all_column_checks_within_cancellation_bound
                   && comparison.vendor_rows_outside_forward_error_bound == 0
                   && comparison.tiled_rows_outside_forward_error_bound == 0) {
            comparison.classification =
                "checksum_tolerance_false_positive_under_cancellation_aware_bound";
        }
        any_vendor_corruption_classification =
            any_vendor_corruption_classification
            || comparison.classification ==
                "vendor_result_outside_forward_error_bound";
        any_unclassified_column = any_unclassified_column
            || comparison.classification == "unclassified";
        every_column_is_cancellation_false_positive =
            every_column_is_cancellation_false_positive
            && comparison.classification ==
                "checksum_tolerance_false_positive_under_cancellation_aware_bound";
        diagnostic.column_comparisons.push_back(std::move(comparison));
    }
    diagnostic.original_b_after_reference = fingerprint_col_major_matrix(
        k, n, b, ldb, requested_threads
    );

    const bool original_b_unchanged =
        diagnostic.original_b_initial == diagnostic.original_b_after_expected
        && diagnostic.original_b_initial == diagnostic.original_b_after_copy
        && diagnostic.original_b_initial == diagnostic.original_b_after_vendor
        && diagnostic.original_b_initial ==
            diagnostic.original_b_after_deterministic
        && diagnostic.original_b_initial ==
            diagnostic.original_b_after_reference;
    const bool protected_b_unchanged =
        diagnostic.protected_b_before_vendor ==
            diagnostic.protected_b_after_vendor;
    const bool protected_b_matches_original =
        diagnostic.original_b_after_copy ==
            diagnostic.protected_b_before_vendor
        && diagnostic.original_b_after_vendor ==
            diagnostic.protected_b_after_vendor;
    if (!original_b_unchanged || !protected_b_unchanged
        || !protected_b_matches_original) {
        diagnostic.classification = "input_fingerprint_changed";
        for (auto& comparison : diagnostic.column_comparisons) {
            comparison.classification = "input_fingerprint_changed";
        }
    } else if (diagnostic.flagged_columns.size()
               > diagnostic.captured_flagged_columns.size()) {
        diagnostic.classification = "unclassified_detail_cap_exceeded";
    } else if (every_column_is_cancellation_false_positive) {
        diagnostic.classification =
            "checksum_tolerance_false_positive_under_cancellation_aware_bound";
    } else if (any_vendor_corruption_classification
               && !any_unclassified_column) {
        diagnostic.classification =
            "vendor_result_outside_forward_error_bound";
    } else {
        diagnostic.classification = "unclassified";
    }

    // Match production behavior only after retaining the raw vendor evidence.
    for (size_t selected = 0; selected < flagged_count; ++selected) {
        const int column = diagnostic.captured_flagged_columns[selected];
        std::memcpy(
            c + static_cast<size_t>(column) * static_cast<size_t>(ldc),
            diagnostic.deterministic_tiled_columns.data()
                + selected * static_cast<size_t>(m),
            static_cast<size_t>(m) * sizeof(double)
        );
    }
    return diagnostic;
}

int64_t dgemm_tt_checked(int m, int n, int k,
                         const double* a, int lda,
                         const double* b, int ldb,
                         double* c, int ldc,
                         int requested_threads) {
    std::vector<double> coefficients = gemm_integrity_coefficients(m);
    std::vector<double> projected(
        checked_mul(
            static_cast<size_t>(k),
            static_cast<size_t>(kGemmIntegrityChecks),
            "TT GEMM integrity projection"
        )
    );
    const size_t check_elements = checked_mul(
        static_cast<size_t>(kGemmIntegrityChecks),
        static_cast<size_t>(n),
        "TT GEMM integrity checks"
    );
    std::vector<double> expected(check_elements);
    std::vector<double> observed(check_elements);

    const size_t operand_a = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(m),
        "protected TT operand A"
    );
    std::unique_ptr<double[]> protected_a(new double[operand_a]);
    const MatrixFingerprint original_a_before = fingerprint_col_major_matrix(
        k, m, a, lda, requested_threads
    );
    copy_col_major_matrix(
        k, m, a, lda, protected_a.get(), k, requested_threads
    );
    if (!(original_a_before == fingerprint_col_major_matrix(
              k, m, a, lda, requested_threads))) {
        throw std::runtime_error(
            "Protected GxE TT input changed while it was snapshotted"
        );
    }
    const MatrixFingerprint protected_a_fingerprint =
        fingerprint_col_major_matrix(
            k, m, protected_a.get(), k, requested_threads
        );
    if (!(original_a_before == protected_a_fingerprint)) {
        throw std::runtime_error(
            "Protected GxE TT weight snapshot does not match its input"
        );
    }
    dgemm_nn_tiled(
        k, kGemmIntegrityChecks, m,
        protected_a.get(), k, coefficients.data(), m,
        projected.data(), k, requested_threads
    );
    dgemm_tt_tiled(
        kGemmIntegrityChecks, n, k,
        projected.data(), k, b, ldb,
        expected.data(), kGemmIntegrityChecks, requested_threads
    );
    const MatrixFingerprint b_before = fingerprint_col_major_matrix(
        n, k, b, ldb, requested_threads
    );
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_tt_fixed_vendor(
        m, n, k, protected_a.get(), k, b, ldb, c, ldc,
        requested_threads
    );
#else
    dgemm_tt_vendor_observed(
        m, n, k, protected_a.get(), k, b, ldb, c, ldc,
        requested_threads
    );
#endif
    if (!(protected_a_fingerprint == fingerprint_col_major_matrix(
              k, m, protected_a.get(), k, requested_threads))) {
        throw std::runtime_error(
            "Vendor BLAS altered a protected GxE TT weight snapshot"
        );
    }
    if (!(b_before == fingerprint_col_major_matrix(
              n, k, b, ldb, requested_threads))) {
        throw RetryableGemmInputMutation();
    }
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, m,
        coefficients.data(), m, c, ldc,
        observed.data(), kGemmIntegrityChecks, requested_threads
    );
    const auto column_disagrees = [&](int column) {
        for (int check = 0; check < kGemmIntegrityChecks; ++check) {
            const size_t index =
                static_cast<size_t>(column)
                    * static_cast<size_t>(kGemmIntegrityChecks)
                + static_cast<size_t>(check);
            if (gemm_integrity_disagrees(
                    expected[index], observed[index], m, k)) {
                return true;
            }
        }
        return false;
    };
    int64_t repaired = 0;
    for (int column = 0; column < n;) {
        if (!column_disagrees(column)) {
            ++column;
            continue;
        }
        const int first = column;
        do {
            ++repaired;
            ++column;
        } while (column < n && column_disagrees(column));
        dgemm_tt_tiled(
            m, column - first, k,
            protected_a.get(), k, b + first, ldb,
            c + static_cast<size_t>(first) * static_cast<size_t>(ldc),
            ldc, requested_threads
        );
    }
    return repaired;
}

int64_t dgemm_row_tn_checked(
    int m, int n, int k,
    const double* a, int lda,
    const double* b, int ldb,
    double* c, int ldc,
    int requested_threads,
    const MatrixFingerprint& immutable_b_fingerprint
) {
    std::vector<double> coefficients =
        gemm_integrity_coefficients_row_major(m);
    std::vector<double> projected(
        checked_mul(
            static_cast<size_t>(k),
            static_cast<size_t>(kGemmIntegrityChecks),
            "row-major TN integrity projection"
        )
    );
    const size_t check_elements = checked_mul(
        static_cast<size_t>(kGemmIntegrityChecks),
        static_cast<size_t>(n),
        "row-major TN integrity checks"
    );
    std::vector<double> expected(check_elements);
    std::vector<double> observed(check_elements);
    dgemm_row_nn_tiled(
        k, kGemmIntegrityChecks, m,
        a, lda, coefficients.data(), kGemmIntegrityChecks,
        projected.data(), kGemmIntegrityChecks, requested_threads
    );
    dgemm_row_tn_tiled(
        kGemmIntegrityChecks, n, k,
        projected.data(), kGemmIntegrityChecks, b, ldb,
        expected.data(), n, requested_threads
    );
    const MatrixFingerprint a_before = fingerprint_row_major_matrix(
        k, m, a, lda, requested_threads
    );
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_row_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#else
    dgemm_row_tn_vendor_observed(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
    if (!(immutable_b_fingerprint == fingerprint_row_major_matrix(
              k, n, b, ldb, requested_threads))) {
        throw std::runtime_error(
            "Vendor BLAS altered an immutable row-major GxE target pair"
        );
    }
    if (!(a_before == fingerprint_row_major_matrix(
              k, m, a, lda, requested_threads))) {
        throw RetryableGemmInputMutation();
    }
    dgemm_row_tn_tiled(
        kGemmIntegrityChecks, n, m,
        coefficients.data(), kGemmIntegrityChecks, c, ldc,
        observed.data(), n, requested_threads
    );
    const auto column_disagrees = [&](int column) {
        for (int check = 0; check < kGemmIntegrityChecks; ++check) {
            const size_t index =
                static_cast<size_t>(check) * static_cast<size_t>(n)
                + static_cast<size_t>(column);
            if (gemm_integrity_disagrees(
                    expected[index], observed[index], m, k)) {
                return true;
            }
        }
        return false;
    };
    int64_t repaired = 0;
    for (int column = 0; column < n;) {
        if (!column_disagrees(column)) {
            ++column;
            continue;
        }
        const int first = column;
        do {
            ++repaired;
            ++column;
        } while (column < n && column_disagrees(column));
        dgemm_row_tn_tiled(
            m, column - first, k,
            a, lda, b + first, ldb, c + first, ldc,
            requested_threads
        );
    }
    return repaired;
}
#endif

size_t partitioned_gemm_integrity_workspace_elements(int m, int n, int k) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
    return gemm_integrity_workspace_elements(m, n, k);
#else
    (void)m; (void)n; (void)k;
    return 0;
#endif
}

int64_t dgemm_tn_partitioned_columns(int m, int n, int k,
                                     const double* a, int lda,
                                     const double* b, int ldb,
                                     double* c, int ldc,
                                     int requested_threads,
                                     GemmIntegrityResolution* resolution = nullptr) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    if (m <= 64 || gemm_uses_deterministic_feature_tn(m, n, k)) {
        // Q^T S products have a persistent design basis and source panel on
        // both sides. They are narrow enough to compute deterministically,
        // even when a large probe count would cross the generic ABFT cutoff.
        // The bounded 10K feature shape is included because it was the only
        // remaining intermittently failing BLIS call in that pipeline.
        dgemm_tn_tiled(
            m, n, k, a, lda, b, ldb, c, ldc, requested_threads
        );
        if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
        return 0;
    }
    if (gemm_requires_integrity_checks(m, n, k)) {
        return dgemm_tn_checked(
            m, n, k, a, lda, b, ldb, c, ldc, requested_threads,
            nullptr, false, resolution
        );
    }
#if defined(GWLDCORE_GEMM_CHECKSUM)
    // Vendor microkernel selection can depend on how a small product is
    // partitioned, producing last-bit differences across thread counts even
    // though every output cell has a single owner.  Keep the reproducibility
    // contract exact for these inexpensive products.
    dgemm_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#else
    dgemm_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
#else
    dgemm_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
    return 0;
#else
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#else
    dgemm_tn_vendor_observed(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
    return 0;
#endif
}

int64_t dgemm_nn_partitioned_rows(int m, int n, int k,
                                  const double* a, int lda,
                                  const double* b, int ldb,
                                  double* c, int ldc,
                                  int requested_threads,
                                  double alpha = 1.0, double beta = 0.0,
                                  GemmIntegrityResolution* resolution = nullptr) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    if (alpha != 1.0 || beta != 0.0) {
        // Integrity verification below is formulated for an overwrite
        // product. Projection updates use C <- C - Q(Q^T C); routing that
        // case through unchecked vendor BLAS defeats the protection and can
        // corrupt an otherwise valid projected panel. The reduction rank is
        // only the fixed-effect rank, so the deterministic update is cheap.
        dgemm_nn_tiled(
            m, n, k, a, lda, b, ldb, c, ldc,
            requested_threads, alpha, beta
        );
        if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
        return 0;
    }
    if (gemm_requires_integrity_checks(m, n, k)) {
        return dgemm_nn_checked(
            m, n, k, a, lda, b, ldb, c, ldc, requested_threads,
            resolution
        );
    }
#if defined(GWLDCORE_GEMM_CHECKSUM)
    dgemm_nn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#else
    dgemm_nn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
#else
    dgemm_nn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
    if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
#endif
    return 0;
#else
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_nn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#else
    dgemm_nn_vendor_observed(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
    return 0;
#endif
}

int64_t dgemm_tn_partitioned_rows(int m, int n, int k,
                                  const double* a, int lda,
                                  const double* b, int ldb,
                                  double* c, int ldc,
                                  int requested_threads,
                                  double alpha = 1.0, double beta = 0.0,
                                  const MatrixFingerprint* immutable_b_fingerprint = nullptr,
                                  bool immutable_b_is_read_only = false,
                                  GemmIntegrityResolution* resolution = nullptr) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    if (alpha != 1.0 || beta != 0.0) {
        dgemm_tn_tiled(
            m, n, k, a, lda, b, ldb, c, ldc,
            requested_threads, alpha, beta
        );
        if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
        return 0;
    }
    // This bounded regime is inexpensive enough to compute once with the
    // deterministic OpenMP kernel. Avoiding vendor startup and a possible
    // checksum-triggered recomputation is cheaper for these narrow calls.
    if (gemm_uses_deterministic_small_tn(m, n, k)) {
        dgemm_tn_tiled(
            m, n, k, a, lda, b, ldb, c, ldc,
            requested_threads, alpha, beta
        );
        if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
        return 0;
    }
    if (gemm_requires_integrity_checks(m, n, k)) {
        return dgemm_tn_checked(
            m, n, k, a, lda, b, ldb, c, ldc, requested_threads,
            immutable_b_fingerprint, immutable_b_is_read_only, resolution
        );
    }
#if defined(GWLDCORE_GEMM_CHECKSUM)
    dgemm_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#else
    dgemm_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    if (resolution != nullptr) *resolution = GemmIntegrityResolution{};
#else
    dgemm_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    return 0;
#else
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#else
    dgemm_tn_vendor_observed(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    return 0;
#endif
}

int64_t dgemm_tt_partitioned(int m, int n, int k,
                             const double* a, int lda,
                             const double* b, int ldb,
                             double* c, int ldc,
                             int requested_threads) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    if (gemm_requires_integrity_checks(m, n, k)) {
        return dgemm_tt_checked(
            m, n, k, a, lda, b, ldb, c, ldc, requested_threads
        );
    }
#if !defined(GWLDCORE_GEMM_CHECKSUM)
    dgemm_tt_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
    return 0;
#endif
#endif
    dgemm_tt_tiled(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
    return 0;
#else
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_tt_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#else
    dgemm_tt_vendor_observed(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
    return 0;
#endif
}

int64_t dgemm_row_tn_partitioned(
    int m, int n, int k,
    const double* a, int lda,
    const double* b, int ldb,
    double* c, int ldc,
    int requested_threads,
    const MatrixFingerprint& immutable_b_fingerprint
) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    if (gemm_requires_integrity_checks(m, n, k)) {
        return dgemm_row_tn_checked(
            m, n, k, a, lda, b, ldb, c, ldc,
            requested_threads, immutable_b_fingerprint
        );
    }
#if !defined(GWLDCORE_GEMM_CHECKSUM)
    dgemm_row_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
    return 0;
#endif
#endif
    dgemm_row_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
    return 0;
#else
    (void)immutable_b_fingerprint;
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
    dgemm_row_tn_fixed_vendor(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#else
    dgemm_row_tn_vendor_observed(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
    return 0;
#endif
}

#if defined(__linux__)

struct FileState {
    dev_t device{};
    ino_t inode{};
    off_t size{};
    nlink_t links{};
    timespec mtime{};
    timespec ctime{};
};

bool same_timespec(const timespec& a, const timespec& b) {
    return a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec;
}

FileState state_from_stat(const struct stat& observed) {
    return FileState{
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_nlink,
        observed.st_mtim,
        observed.st_ctim,
    };
}

bool same_state(const FileState& a, const FileState& b) {
    return a.device == b.device && a.inode == b.inode && a.size == b.size &&
           a.links == b.links &&
           same_timespec(a.mtime, b.mtime) && same_timespec(a.ctime, b.ctime);
}

int duplicate_cloexec(int descriptor, const char* label) {
    if (descriptor < 0) {
        throw std::runtime_error(std::string("Invalid GxE native ") + label + " descriptor");
    }
#ifdef F_DUPFD_CLOEXEC
    int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
#else
    int duplicate = ::dup(descriptor);
    if (duplicate >= 0) {
        const int flags = ::fcntl(duplicate, F_GETFD);
        if (flags < 0 || ::fcntl(duplicate, F_SETFD, flags | FD_CLOEXEC) != 0) {
            const int saved = errno;
            ::close(duplicate);
            errno = saved;
            duplicate = -1;
        }
    }
#endif
    if (duplicate < 0) {
        throw std::runtime_error(
            std::string("Failed to duplicate GxE native ") + label +
            " descriptor with close-on-exec: " + std::strerror(errno)
        );
    }
    return duplicate;
}

FileState validate_regular_fd(int descriptor, const char* label) {
    struct stat observed{};
    if (::fstat(descriptor, &observed) != 0) {
        throw std::runtime_error(
            std::string("Failed to stat GxE native ") + label + " descriptor: " +
            std::strerror(errno)
        );
    }
    if (!S_ISREG(observed.st_mode) || observed.st_size < 0) {
        throw std::runtime_error(
            std::string("GxE native ") + label + " descriptor must reference a regular file"
        );
    }
    return state_from_stat(observed);
}

int count_validated_rows_fd(int descriptor, const FileState& state, const char* label) {
    constexpr size_t chunk_size = 1U << 20;
    std::vector<unsigned char> buffer(chunk_size);
    off_t offset = 0;
    int64_t rows = 0;
    bool line_has_content = false;
    bool line_started = false;
    bool saw_any_byte = false;
    while (offset < state.size) {
        const size_t want = static_cast<size_t>(
            std::min<off_t>(static_cast<off_t>(chunk_size), state.size - offset)
        );
        const ssize_t got = ::pread(descriptor, buffer.data(), want, offset);
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("Failed to read GxE native ") + label + ": " +
                std::strerror(errno)
            );
        }
        if (got == 0) {
            throw std::runtime_error(std::string("Unexpected EOF in GxE native ") + label);
        }
        saw_any_byte = true;
        for (ssize_t i = 0; i < got; ++i) {
            const unsigned char value = buffer[static_cast<size_t>(i)];
            if (value == '\n') {
                if (!line_has_content) {
                    throw std::runtime_error(std::string("GxE native ") + label + " contains a blank row");
                }
                ++rows;
                line_has_content = false;
                line_started = false;
            } else if (value != ' ' && value != '\t' && value != '\r' && value != '\f' && value != '\v') {
                line_has_content = true;
                line_started = true;
            } else {
                line_started = true;
            }
        }
        offset += got;
    }
    if (line_started) {
        if (!line_has_content) {
            throw std::runtime_error(std::string("GxE native ") + label + " contains a blank final row");
        }
        ++rows;
    }
    if (!saw_any_byte || rows <= 0 || rows > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("GxE native ") + label + " has invalid row count");
    }
    return static_cast<int>(rows);
}

// Release only complete pages from a decoded BED range. The underlying file is
// unchanged, and a later source/target pass can fault the same pages back in.
// This mirrors the bounded-residency policy used by the additive BED reader.
void madvise_dontneed_consumed_range(unsigned char* base,
                                     size_t file_size,
                                     size_t offset,
                                     size_t length) noexcept {
    if (base == nullptr || length == 0 || offset >= file_size) return;
    const size_t available = file_size - offset;
    const size_t bounded_length = std::min(length, available);
    const size_t end = offset + bounded_length;
    if (end <= offset) return;

    const long observed_page_size = ::sysconf(_SC_PAGESIZE);
    const size_t page_size = observed_page_size > 0
        ? static_cast<size_t>(observed_page_size)
        : static_cast<size_t>(4096);
    const uintptr_t first_address = reinterpret_cast<uintptr_t>(base + offset);
    const uintptr_t end_address = reinterpret_cast<uintptr_t>(base + end);
    const uintptr_t first_full_page =
        (first_address + page_size - 1) & ~(static_cast<uintptr_t>(page_size) - 1U);
    const uintptr_t after_last_full_page =
        end_address & ~(static_cast<uintptr_t>(page_size) - 1U);
    if (after_last_full_page <= first_full_page) return;
    (void)::madvise(
        reinterpret_cast<void*>(first_full_page),
        static_cast<size_t>(after_last_full_page - first_full_page),
        MADV_DONTNEED
    );
}

#endif

static uint64_t next_context_id() {
    static std::atomic<uint64_t> next{1};
    const uint64_t value = next.fetch_add(1, std::memory_order_relaxed);
    if (value == 0 || value == std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("GxE native context identity space is exhausted");
    }
    return value;
}

class ProjectedPanel {
public:
    ProjectedPanel(ProjectedPanel&&) noexcept = default;
    ProjectedPanel& operator=(ProjectedPanel&&) noexcept = default;
    ProjectedPanel(const ProjectedPanel&) = delete;
    ProjectedPanel& operator=(const ProjectedPanel&) = delete;

    int columns() const noexcept { return columns_; }
    double leakage() const noexcept { return leakage_; }

private:
    friend class DirectContext;

    ProjectedPanel(uint64_t context_id,
                   int rows,
                   int columns,
                   double leakage,
                   size_t elements,
                   MatrixFingerprint fingerprint,
                   std::unique_ptr<double[]>&& data)
        : context_id_(context_id), rows_(rows), columns_(columns),
          leakage_(leakage), elements_(elements), fingerprint_(fingerprint),
          data_(std::move(data)) {}

    uint64_t context_id_ = 0;
    int rows_ = 0;
    int columns_ = 0;
    double leakage_ = 0.0;
    size_t elements_ = 0;
    MatrixFingerprint fingerprint_{};
    // One column-major [S, E*S] allocation lets target work consume both
    // left operators in one wide GEMM without copying the persistent panel.
    std::unique_ptr<double[]> data_;
};

class MultiEnvironmentDirectContext;
class GeneralizedGxELDScoreDirectContext;
struct DescriptorOnlyDirectContextTag {};
void validate_protected_gemm_threads(int requested_threads);

#if defined(__linux__)
class ReadOnlyDoubleMapping {
public:
    explicit ReadOnlyDoubleMapping(size_t elements) {
        const long observed_page_size = ::sysconf(_SC_PAGESIZE);
        if (observed_page_size <= 0) {
            throw std::runtime_error(
                "Could not determine the page size for a protected GxE panel"
            );
        }
        const size_t page_size = static_cast<size_t>(observed_page_size);
        const size_t requested_bytes = checked_mul(
            elements, sizeof(double), "protected GxE panel mapping"
        );
        bytes_ = checked_mul(
            (checked_add(requested_bytes, page_size - 1U,
                         "protected GxE panel mapping") / page_size),
            page_size,
            "protected GxE panel mapping"
        );
        void* observed = ::mmap(
            nullptr, bytes_, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
        );
        if (observed == MAP_FAILED) {
            bytes_ = 0;
            throw std::bad_alloc();
        }
        data_ = static_cast<double*>(observed);
    }

    ~ReadOnlyDoubleMapping() { release(); }
    ReadOnlyDoubleMapping(const ReadOnlyDoubleMapping&) = delete;
    ReadOnlyDoubleMapping& operator=(const ReadOnlyDoubleMapping&) = delete;

    ReadOnlyDoubleMapping(ReadOnlyDoubleMapping&& other) noexcept
        : data_(other.data_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.bytes_ = 0;
    }

    ReadOnlyDoubleMapping& operator=(ReadOnlyDoubleMapping&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            bytes_ = other.bytes_;
            other.data_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    double* data() noexcept { return data_; }
    const double* data() const noexcept { return data_; }

    void seal_read_only() {
        if (data_ == nullptr || bytes_ == 0 ||
            ::mprotect(data_, bytes_, PROT_READ) != 0) {
            throw std::runtime_error(
                "Could not seal the protected GxE target panel read-only"
            );
        }
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            (void)::munmap(data_, bytes_);
            data_ = nullptr;
            bytes_ = 0;
        }
    }

    double* data_ = nullptr;
    size_t bytes_ = 0;
};
#endif

class ProtectedRightPair {
public:
    ProtectedRightPair(ProtectedRightPair&&) noexcept = default;
    ProtectedRightPair& operator=(ProtectedRightPair&&) noexcept = default;
    ProtectedRightPair(const ProtectedRightPair&) = delete;
    ProtectedRightPair& operator=(const ProtectedRightPair&) = delete;

    int rows() const noexcept { return rows_; }
    int columns() const noexcept { return columns_; }
    size_t elements() const noexcept { return elements_; }
    const double* data() const noexcept { return data_.data(); }
    const MatrixFingerprint& fingerprint() const noexcept {
        return fingerprint_;
    }

    ProtectedRightPair(int rows,
                       int columns,
                       size_t elements,
                       MatrixFingerprint fingerprint,
                       ReadOnlyDoubleMapping&& data)
        : rows_(rows), columns_(columns), elements_(elements),
          fingerprint_(fingerprint), data_(std::move(data)) {}

private:
    int rows_ = 0;
    int columns_ = 0;
    size_t elements_ = 0;
    MatrixFingerprint fingerprint_{};
    // Immutable column-major [S, row_weight*S].  Python receives no view of
    // this memory, so one sealed snapshot can safely serve every target block.
    ReadOnlyDoubleMapping data_;
};

class ProtectedRightPairBuilder {
public:
    ProtectedRightPairBuilder(int rows, int columns)
        : rows_(rows), columns_(columns),
          panel_elements_(checked_mul(
              static_cast<size_t>(rows), static_cast<size_t>(columns),
              "protected right-pair builder panel"
          )),
          pair_elements_(checked_mul(
              2U, panel_elements_, "protected right-pair builder allocation"
          )),
          data_(pair_elements_) {
        if (rows_ <= 0 || columns_ <= 0) {
            throw std::runtime_error(
                "Protected GxE right-pair builder dimensions must be positive"
            );
        }
        // Accumulation uses +=. Touch only the source half here; the weighted
        // half remains physically uncommitted until sealing.
        std::fill(data_.data(), data_.data() + panel_elements_, 0.0);
    }

    ProtectedRightPairBuilder(const ProtectedRightPairBuilder&) = delete;
    ProtectedRightPairBuilder& operator=(
        const ProtectedRightPairBuilder&
    ) = delete;

    nb_numpy_mat2f<double> source_view() {
        if (sealed_) {
            throw std::runtime_error(
                "Protected GxE right-pair builder is already sealed"
            );
        }
        nb::capsule owner(data_.data(), [](void*) noexcept {});
        return nb_numpy_mat2f<double>(
            data_.data(),
            {static_cast<size_t>(rows_), static_cast<size_t>(columns_)},
            owner
        );
    }

    ProtectedRightPair seal(
        nb_mat2f_ro<double> row_weights, int requested_threads
    ) {
        if (sealed_) {
            throw std::runtime_error(
                "Protected GxE right-pair builder cannot be sealed twice"
            );
        }
        validate_protected_gemm_threads(requested_threads);
        const int weight_rows = checked_blas_dim(
            row_weights.shape(0), "protected right-pair builder weight rows"
        );
        const int groups = checked_blas_dim(
            row_weights.shape(1), "protected right-pair builder weight groups"
        );
        if (groups <= 0 || weight_rows != rows_ || columns_ % groups != 0) {
            throw std::runtime_error(
                "Protected GxE right-pair builder weights are incompatible"
            );
        }
        const int columns_per_group = columns_ / groups;
        MatrixFingerprint pair_fingerprint{};
        {
            nb::gil_scoped_release release;
#if defined(GWLDCORE_GEMM_INTEGRITY)
            const MatrixFingerprint source_before =
                fingerprint_col_major_matrix(
                    rows_, columns_, data_.data(), rows_, requested_threads
                );
            const MatrixFingerprint weights_before =
                fingerprint_col_major_matrix(
                    rows_, groups, row_weights.data(), rows_, requested_threads
                );
#endif
            int invalid = 0;
            const int threads = std::max(
                1, std::min(requested_threads, columns_)
            );
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads) \
                reduction(|:invalid)
#endif
            for (int column = 0; column < columns_; ++column) {
                const int group = column / columns_per_group;
                const double* source = data_.data()
                    + static_cast<size_t>(column)
                        * static_cast<size_t>(rows_);
                const double* weights = row_weights.data()
                    + static_cast<size_t>(group)
                        * static_cast<size_t>(rows_);
                double* weighted = data_.data() + panel_elements_
                    + static_cast<size_t>(column)
                        * static_cast<size_t>(rows_);
                for (int row = 0; row < rows_; ++row) {
                    weighted[row] = source[row] * weights[row];
                    invalid |= !std::isfinite(source[row])
                        || !std::isfinite(weights[row])
                        || !std::isfinite(weighted[row]);
                }
            }
            if (invalid != 0) {
                throw std::runtime_error(
                    "Protected GxE right-pair builder contains NaN or infinity"
                );
            }
#if defined(GWLDCORE_GEMM_INTEGRITY)
            if (!(source_before == fingerprint_col_major_matrix(
                      rows_, columns_, data_.data(), rows_, requested_threads))
                || !(weights_before == fingerprint_col_major_matrix(
                      rows_, groups, row_weights.data(), rows_, requested_threads))) {
                throw std::runtime_error(
                    "Protected GxE right-pair builder changed while sealing"
                );
            }
            pair_fingerprint = fingerprint_col_major_matrix(
                rows_,
                checked_blas_dim(
                    2U * static_cast<size_t>(columns_),
                    "protected right-pair builder sealed columns"
                ),
                data_.data(), rows_, requested_threads
            );
#endif
            data_.seal_read_only();
        }
        sealed_ = true;
        return ProtectedRightPair(
            rows_, columns_, pair_elements_, pair_fingerprint,
            std::move(data_)
        );
    }

private:
    int rows_ = 0;
    int columns_ = 0;
    size_t panel_elements_ = 0;
    size_t pair_elements_ = 0;
    bool sealed_ = false;
    ReadOnlyDoubleMapping data_;
};

// The packed target kernel reads S and forms e*S virtually in bounded RHS
// panels. It therefore needs only one source mapping, rather than the former
// two-half [S,e*S] allocation.
class PackedSourcePanel {
public:
    PackedSourcePanel(
        int rows, int columns, const NativeNumaContractRequest& request,
        std::string operand_role = "persistent_packed_source_panel",
        std::string verification_boundary =
            "after_projection_before_target_scoring"
    )
        : rows_(rows), columns_(columns),
          elements_(checked_mul(
              static_cast<size_t>(rows), static_cast<size_t>(columns),
              "packed source panel"
          )), request_(request), operand_role_(std::move(operand_role)),
          verification_boundary_(std::move(verification_boundary)) {
        if (rows_ <= 0 || columns_ <= 0) {
            throw std::runtime_error(
                "Packed GxE source-panel dimensions must be positive"
            );
        }
        if (request_.required) {
            NativeGemmOutputNumaEvidenceData initial;
            initial.applicable = true;
            initial.contract_required = true;
            evidence_ =
                std::make_shared<SharedNativeGemmOutputNumaEvidence>(
                    std::move(initial)
                );
            bound_data_ = std::make_unique<NativeGemmOutputAllocation>(
                static_cast<size_t>(rows_), static_cast<size_t>(columns_),
                "column_major",
                checked_mul(elements_, sizeof(double), "packed source panel"),
                request_, evidence_
            );
        } else {
            legacy_data_ = std::make_unique<ReadOnlyDoubleMapping>(elements_);
        }
        // MAP_ANONYMOUS storage is logically zero. Leave it untouched here so
        // the source workers, not the constructor thread, establish NUMA
        // first-touch placement while accumulating their disjoint row slabs.
    }

    PackedSourcePanel(const PackedSourcePanel&) = delete;
    PackedSourcePanel& operator=(const PackedSourcePanel&) = delete;

    nb_numpy_mat2f<double> writable_view() {
        return writable_column_view(0, columns_);
    }

    nb_numpy_mat2f<double> writable_column_view(
        int column_start, int column_count
    ) {
        if (sealed_) {
            throw std::runtime_error("Packed GxE source panel is read-only");
        }
        if (column_start < 0 || column_count <= 0
            || column_start > columns_ - column_count) {
            throw std::runtime_error(
                "Packed GxE source-panel column view is out of range"
            );
        }
        double* view_data = data()
            + checked_mul(
                static_cast<size_t>(column_start),
                static_cast<size_t>(rows_),
                "packed source-panel column view"
            );
        nb::capsule owner(view_data, [](void*) noexcept {});
        return nb_numpy_mat2f<double>(
            view_data,
            {static_cast<size_t>(rows_), static_cast<size_t>(column_count)},
            owner
        );
    }

    nb_numpy_mat2f<double> readonly_view() const {
        if (!sealed_) {
            throw std::runtime_error(
                "Packed GxE source panel must be sealed before target scoring"
            );
        }
        nb::capsule owner(
            const_cast<double*>(data()), [](void*) noexcept {}
        );
        return nb_numpy_mat2f<double>(
            const_cast<double*>(data()),
            {static_cast<size_t>(rows_), static_cast<size_t>(columns_)},
            owner
        );
    }

    void seal() {
        if (sealed_) {
            throw std::runtime_error(
                "Packed GxE source panel cannot be sealed twice"
            );
        }
        if (bound_data_ != nullptr) {
            bound_data_->verify_after_repair();
            bound_data_->seal_read_only();
        } else {
            legacy_data_->seal_read_only();
        }
        sealed_ = true;
    }

    nb::dict numa_evidence() const {
        if (!sealed_) {
            throw std::runtime_error(
                "Packed GxE source-panel NUMA evidence is not final"
            );
        }
        nb::dict result;
        if (request_.required) {
            result = native_gemm_output_numa_to_dict(evidence_);
        } else {
            result["schema_version"] = 1;
            result["applicable"] = true;
            result["contract_required"] = false;
            result["complete"] = false;
            result["logical_rows"] = rows_;
            result["logical_columns"] = columns_;
            result["storage_layout"] = "column_major";
            result["logical_byte_count"] = checked_mul(
                elements_, sizeof(double), "packed source evidence"
            );
            result["allocation_mode"] = "legacy_anonymous_mapping";
        }
        result["schema"] = "summit.packed_source_panel_numa.v1";
        result["schema_version"] = 1;
        result["operand_role"] = operand_role_;
        result["verification_boundary"] = verification_boundary_;
        result["sealed_read_only"] = true;
        return result;
    }

private:
    double* data() noexcept {
        return bound_data_ != nullptr
            ? bound_data_->data() : legacy_data_->data();
    }
    const double* data() const noexcept {
        return bound_data_ != nullptr
            ? bound_data_->data()
            : legacy_data_->data();
    }

    int rows_ = 0;
    int columns_ = 0;
    size_t elements_ = 0;
    bool sealed_ = false;
    NativeNumaContractRequest request_;
    std::shared_ptr<SharedNativeGemmOutputNumaEvidence> evidence_;
    std::unique_ptr<NativeGemmOutputAllocation> bound_data_;
    std::unique_ptr<ReadOnlyDoubleMapping> legacy_data_;
    std::string operand_role_;
    std::string verification_boundary_;
};

class ProtectedRowMajorPair {
public:
    ProtectedRowMajorPair(ProtectedRowMajorPair&&) noexcept = default;
    ProtectedRowMajorPair& operator=(ProtectedRowMajorPair&&) noexcept = default;
    ProtectedRowMajorPair(const ProtectedRowMajorPair&) = delete;
    ProtectedRowMajorPair& operator=(const ProtectedRowMajorPair&) = delete;

    int rows() const noexcept { return rows_; }
    int columns() const noexcept { return columns_; }
    size_t elements() const noexcept { return elements_; }
    const double* data() const noexcept { return data_.data(); }
    const MatrixFingerprint& fingerprint() const noexcept {
        return fingerprint_;
    }

    ProtectedRowMajorPair(int rows,
                          int columns,
                          size_t elements,
                          MatrixFingerprint fingerprint,
                          ReadOnlyDoubleMapping&& data)
        : rows_(rows), columns_(columns), elements_(elements),
          fingerprint_(fingerprint), data_(std::move(data)) {}

private:
    int rows_ = 0;
    int columns_ = 0;
    size_t elements_ = 0;
    MatrixFingerprint fingerprint_{};
    // Immutable row-major [S, row_weight*S]. Python receives no view of this
    // memory, and the target API accepts only the owning opaque object.
    ReadOnlyDoubleMapping data_;
};

class DirectContext {
public:
    DirectContext(int bed_descriptor,
                  int bim_descriptor,
                  int fam_descriptor,
                  nb::object row_sel_obj,
                  int ddof,
                  nb_vec1_ro<double> env,
                  nb_mat2f_ro<double> q_basis,
                  int decode_threads,
                  uint64_t max_workspace_bytes,
                  int target_panel_columns,
                  bool strict_feature_moment_verification,
                  int blas_threads)
        : context_id_(next_context_id()),
          ddof_(ddof),
          decode_threads_(decode_threads),
          blas_threads_(blas_threads > 0 ? blas_threads : decode_threads),
          max_workspace_bytes_(max_workspace_bytes),
          target_panel_columns_(target_panel_columns),
          strict_feature_moment_verification_(strict_feature_moment_verification) {
#if !defined(__linux__)
        (void)bed_descriptor; (void)bim_descriptor; (void)fam_descriptor;
        (void)row_sel_obj; (void)env; (void)q_basis;
        throw std::runtime_error("The bounded GxE native context requires Linux");
#else
        if (ddof_ != 0 && ddof_ != 1) {
            throw std::runtime_error("GxE native context supports ddof 0 or 1");
        }
        if (decode_threads_ <= 0) {
            throw std::runtime_error("GxE native decode_threads must be positive");
        }
        if (blas_threads_ <= 0) {
            throw std::runtime_error("GxE native blas_threads must be positive");
        }
#ifdef _OPENMP
        validate_configured_openmp_placement_for_entry();
        const int thread_limit = effective_openmp_capacity();
        if (decode_threads_ > thread_limit) {
            throw std::runtime_error(
                "GxE native decode_threads exceeds the OpenMP/CPU-affinity limit"
            );
        }
        if (blas_threads_ > thread_limit) {
            throw std::runtime_error(
                "GxE native blas_threads exceeds the OpenMP/CPU-affinity limit"
            );
        }
#else
        if (decode_threads_ != 1) {
            throw std::runtime_error(
                "GxE native decode_threads must be one when OpenMP is unavailable"
            );
        }
        if (blas_threads_ != 1) {
            throw std::runtime_error(
                "GxE native blas_threads must be one when OpenMP is unavailable"
            );
        }
#endif
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
        configure_fixed_vendor_threads(blas_threads_);
#endif
        if (max_workspace_bytes_ == 0 ||
            max_workspace_bytes_ > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("GxE native max_workspace_bytes is invalid");
        }
        if (target_panel_columns_ <= 0) {
            throw std::runtime_error("GxE native target_panel_columns must be positive");
        }
        try {
            bed_fd_ = duplicate_cloexec(bed_descriptor, "BED");
            bim_fd_ = duplicate_cloexec(bim_descriptor, "BIM");
            fam_fd_ = duplicate_cloexec(fam_descriptor, "FAM");
            bed_state_ = validate_regular_fd(bed_fd_, "BED");
            bim_state_ = validate_regular_fd(bim_fd_, "BIM");
            fam_state_ = validate_regular_fd(fam_fd_, "FAM");
            n_total_ = count_validated_rows_fd(fam_fd_, fam_state_, "FAM");
            m_total_ = count_validated_rows_fd(bim_fd_, bim_state_, "BIM");
            const size_t bytes_per_snp = checked_add(static_cast<size_t>(n_total_), 3, "BED stride") / 4;
            const size_t expected = checked_add(
                3, checked_mul(bytes_per_snp, static_cast<size_t>(m_total_), "BED byte size"),
                "BED byte size"
            );
            if (static_cast<uint64_t>(bed_state_.size) != static_cast<uint64_t>(expected)) {
                throw std::runtime_error("GxE native BED byte size disagrees with FAM/BIM dimensions");
            }
            bed_size_ = expected;
            bytes_per_snp_ = bytes_per_snp;
            bed_base_ = static_cast<unsigned char*>(
                ::mmap(nullptr, bed_size_, PROT_READ, MAP_PRIVATE, bed_fd_, 0)
            );
            if (bed_base_ == MAP_FAILED) {
                bed_base_ = nullptr;
                throw std::runtime_error(
                    std::string("Failed to mmap GxE native BED: ") + std::strerror(errno)
                );
            }
            (void)::madvise(bed_base_, bed_size_, MADV_SEQUENTIAL);
            if (bed_base_[0] != 0x6c || bed_base_[1] != 0x1b || bed_base_[2] != 0x01) {
                throw std::runtime_error("GxE native input is not a SNP-major PLINK BED");
            }
            parse_rows(std::move(row_sel_obj));
            copy_and_validate_design(env, q_basis);
            check_files_unchanged();
        } catch (...) {
            close_internal();
            throw;
        }
#endif
    }

    DirectContext(int bed_descriptor,
                  int bim_descriptor,
                  int fam_descriptor,
                  nb::object row_sel_obj,
                  int ddof,
                  int decode_threads,
                  uint64_t max_workspace_bytes,
                  int blas_threads,
                  DescriptorOnlyDirectContextTag)
        : context_id_(next_context_id()),
          ddof_(ddof),
          decode_threads_(decode_threads),
          blas_threads_(blas_threads > 0 ? blas_threads : decode_threads),
          max_workspace_bytes_(max_workspace_bytes),
          target_panel_columns_(1),
          strict_feature_moment_verification_(false) {
#if !defined(__linux__)
        (void)bed_descriptor; (void)bim_descriptor; (void)fam_descriptor;
        (void)row_sel_obj;
        throw std::runtime_error(
            "The descriptor-backed genotype operator requires Linux"
        );
#else
        if (ddof_ != 0 && ddof_ != 1) {
            throw std::runtime_error(
                "Descriptor-backed genotype operator supports ddof 0 or 1"
            );
        }
        if (decode_threads_ <= 0 || blas_threads_ <= 0) {
            throw std::runtime_error(
                "Descriptor-backed genotype operator threads must be positive"
            );
        }
#ifdef _OPENMP
        validate_configured_openmp_placement_for_entry();
        const int thread_limit = effective_openmp_capacity();
        if (decode_threads_ > thread_limit || blas_threads_ > thread_limit) {
            throw std::runtime_error(
                "Descriptor-backed genotype operator exceeds the OpenMP/CPU-affinity limit"
            );
        }
#else
        if (decode_threads_ != 1 || blas_threads_ != 1) {
            throw std::runtime_error(
                "Descriptor-backed genotype operator requires one thread without OpenMP"
            );
        }
#endif
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
        configure_fixed_vendor_threads(blas_threads_);
#endif
        if (max_workspace_bytes_ == 0 ||
            max_workspace_bytes_ >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error(
                "Descriptor-backed genotype max_workspace_bytes is invalid"
            );
        }
        try {
            bed_fd_ = duplicate_cloexec(bed_descriptor, "BED");
            bim_fd_ = duplicate_cloexec(bim_descriptor, "BIM");
            fam_fd_ = duplicate_cloexec(fam_descriptor, "FAM");
            bed_state_ = validate_regular_fd(bed_fd_, "BED");
            bim_state_ = validate_regular_fd(bim_fd_, "BIM");
            fam_state_ = validate_regular_fd(fam_fd_, "FAM");
            n_total_ = count_validated_rows_fd(fam_fd_, fam_state_, "FAM");
            m_total_ = count_validated_rows_fd(bim_fd_, bim_state_, "BIM");
            const size_t bytes_per_snp = checked_add(
                static_cast<size_t>(n_total_), 3, "BED stride"
            ) / 4;
            const size_t expected = checked_add(
                3,
                checked_mul(
                    bytes_per_snp, static_cast<size_t>(m_total_),
                    "BED byte size"
                ),
                "BED byte size"
            );
            if (static_cast<uint64_t>(bed_state_.size) !=
                static_cast<uint64_t>(expected)) {
                throw std::runtime_error(
                    "Descriptor-backed BED size disagrees with FAM/BIM dimensions"
                );
            }
            bed_size_ = expected;
            bytes_per_snp_ = bytes_per_snp;
            bed_base_ = static_cast<unsigned char*>(
                ::mmap(nullptr, bed_size_, PROT_READ, MAP_PRIVATE, bed_fd_, 0)
            );
            if (bed_base_ == MAP_FAILED) {
                bed_base_ = nullptr;
                throw std::runtime_error(
                    std::string("Failed to mmap descriptor-backed BED: ")
                    + std::strerror(errno)
                );
            }
            (void)::madvise(bed_base_, bed_size_, MADV_SEQUENTIAL);
            if (bed_base_[0] != 0x6c || bed_base_[1] != 0x1b ||
                bed_base_[2] != 0x01) {
                throw std::runtime_error(
                    "Descriptor-backed input is not a SNP-major PLINK BED"
                );
            }
            parse_rows(std::move(row_sel_obj));
            check_files_unchanged();
        } catch (...) {
            close_internal();
            throw;
        }
#endif
    }

    ~DirectContext() { close_internal(); }
    DirectContext(const DirectContext&) = delete;
    DirectContext& operator=(const DirectContext&) = delete;

    void close() {
        auto guard = acquire_call_lock();
        close_internal();
    }

    nb::dict info() const {
        auto guard = acquire_call_lock();
        ensure_open();
        nb::dict result;
        result["n_total"] = n_total_;
        result["m_total"] = m_total_;
        result["n_selected"] = n_;
        result["q_rank"] = q_;
        result["ddof"] = ddof_;
        result["decode_threads"] = decode_threads_;
        result["blas_threads"] = blas_threads_;
        result["max_workspace_bytes"] = max_workspace_bytes_;
        result["target_panel_columns"] = target_panel_columns_;
        result["projected_target_full_width"] = true;
        result["strict_feature_moment_verification"] = strict_feature_moment_verification_;
        result["feature_moment_integrity_mode"] = strict_feature_moment_verification_
            ? "strict_duplicate"
            : "deterministic_disjoint_output_tiled_gemm";
        result["repaired_gemm_output_columns"] =
            repaired_gemm_output_columns_.load(std::memory_order_relaxed);
        result["retried_gemm_input_mutations"] =
            retried_gemm_input_mutations_.load(std::memory_order_relaxed);
        result["environment_mean"] = environment_mean_;
        result["environment_variance"] = environment_variance_;
        result["max_q_gram_error"] = max_q_gram_error_;
        return result;
    }

    nb::dict feature_block(int blk_start,
                           int blk_end,
                           double eps_var,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        if (!(eps_var > 0.0) || !std::isfinite(eps_var)) {
            throw std::runtime_error("GxE native eps_var must be positive and finite");
        }
        check_files_unchanged();
        const int l = blk_end - blk_start;
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "feature genotype");
        const size_t moment_copies = strict_feature_moment_verification_ ? 8U : 4U;
        elements = checked_add(elements, checked_mul(moment_copies, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(l), "feature moments"), "feature moments"), "feature workspace");
        elements = checked_add(elements, checked_mul(11U, static_cast<size_t>(l), "feature vectors"), "feature workspace");
        ensure_workspace(elements, "feature block");

        double* scale_x = nullptr;
        double* scale_w = nullptr;
        double* norm_x = nullptr;
        double* norm_w = nullptr;
        double* diag_x = nullptr;
        double* diag_w = nullptr;
        double* corr_xw = nullptr;
        auto scale_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &scale_x);
        auto scale_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &scale_w);
        auto norm_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &norm_x);
        auto norm_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &norm_w);
        auto diag_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &diag_x);
        auto diag_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &diag_w);
        auto corr_xw_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &corr_xw);

        int64_t missing = 0;
        int64_t repaired_feature_moment_columns = 0;
        double max_leak_x = 0.0;
        double max_leak_w = 0.0;
        {
            nb::gil_scoped_release release;
            std::unique_ptr<double[]> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            compute_feature_moments_from_genotype(
                geno.get(), l, eps_var,
                scale_x, scale_w, norm_x, norm_w,
                diag_x, diag_w, corr_xw,
                repaired_feature_moment_columns, max_leak_x, max_leak_w
            );
            check_files_unchanged();
        }
        nb::dict result;
        result["scale_x"] = scale_x_out;
        result["scale_w"] = scale_w_out;
        result["norm_x"] = norm_x_out;
        result["norm_w"] = norm_w_out;
        result["diag_nxe_x"] = diag_x_out;
        result["diag_nxe_w"] = diag_w_out;
        result["corr_xw"] = corr_xw_out;
        result["max_projection_leakage_additive"] = max_leak_x;
        result["max_projection_leakage_interaction"] = max_leak_w;
        result["repaired_feature_moment_columns"] = repaired_feature_moment_columns;
        result["repaired_additive_moment_columns"] = repaired_feature_moment_columns;
        result["strict_feature_moment_verification"] = strict_feature_moment_verification_;
        result["feature_moment_integrity_mode"] = strict_feature_moment_verification_
            ? "strict_duplicate"
            : "deterministic_disjoint_output_tiled_gemm";
        result["missing_genotype_calls"] = missing;
        return result;
    }

    nb::tuple source_block(int blk_start,
                           int blk_end,
                           nb_vec1_ro<double> scale_x,
                           nb_vec1_ro<double> scale_w,
                           nb_vec1_ro<double> sqrt_annotation,
                           nb_mat2f_ro<double> probes,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int v = checked_blas_dim(probes.shape(1), "source probes");
        if (v <= 0 || checked_blas_dim(probes.shape(0), "source variants") != l ||
            checked_blas_dim(scale_x.shape(0), "source scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "source scale_w") != l ||
            checked_blas_dim(sqrt_annotation.shape(0), "source annotation") != l) {
            throw std::runtime_error("GxE native source input shape mismatch");
        }
        const size_t columns_size = static_cast<size_t>(v);
        const int columns = checked_blas_dim(columns_size, "source columns");
        const size_t fused_columns_size = checked_mul(
            2U, columns_size, "fused source columns"
        );
        const int fused_columns = checked_blas_dim(
            fused_columns_size, "fused source columns"
        );
        const size_t probe_elements = checked_mul(
            static_cast<size_t>(l), static_cast<size_t>(v), "source probe snapshot"
        );
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "source genotype");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(l), columns_size, "source weights"), "source weights"), "source workspace");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(n_), columns_size, "source outputs"), "source outputs"), "source workspace");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(q_), columns_size, "source projection"), "source projection"), "source workspace");
        elements = checked_add(elements, probe_elements, "source input snapshots");
        elements = checked_add(elements, checked_mul(3U, static_cast<size_t>(l), "source vector snapshots"), "source input snapshots");
        elements = checked_add(
            elements,
            std::max(
                partitioned_gemm_integrity_workspace_elements(
                    n_, fused_columns, l
                ),
                partitioned_gemm_integrity_workspace_elements(
                    q_, fused_columns, n_
                )
            ),
            "source integrity workspace"
        );
        ensure_workspace(elements, "source block");

        // Nanobind array views borrow caller memory.  Snapshot every input while
        // the GIL is held so no borrowed buffer is read during native compute.
        const std::vector<double> scale_x_snapshot(
            scale_x.data(), scale_x.data() + static_cast<size_t>(l)
        );
        const std::vector<double> scale_w_snapshot(
            scale_w.data(), scale_w.data() + static_cast<size_t>(l)
        );
        const std::vector<double> annotation_snapshot(
            sqrt_annotation.data(), sqrt_annotation.data() + static_cast<size_t>(l)
        );
        const std::vector<double> probe_snapshot(
            probes.data(), probes.data() + probe_elements
        );
        double* fused_source = nullptr;
        auto fused_source_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(n_), fused_columns_size, &fused_source
        );
        nb::object fused_source_owner = nb::cast(fused_source_out);
        // Keep the established two-array API without allocating or copying
        // the halves. Both returned views retain the one fused owner.
        auto source_x_out = nb_numpy_mat2f<double>(
            fused_source,
            {static_cast<size_t>(n_), columns_size},
            fused_source_owner
        );
        auto source_w_out = nb_numpy_mat2f<double>(
            fused_source + checked_mul(
                static_cast<size_t>(n_), columns_size, "source view offset"
            ),
            {static_cast<size_t>(n_), columns_size},
            fused_source_owner
        );
        int64_t missing = 0;
        {
            nb::gil_scoped_release release;
            std::unique_ptr<double[]> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            std::vector<double> weighted(
                checked_mul(static_cast<size_t>(l), fused_columns_size, "source weights"), 0.0
            );
            const double* zp = probe_snapshot.data();
            const double* sxp = scale_x_snapshot.data();
            const double* swp = scale_w_snapshot.data();
            const double* ap = annotation_snapshot.data();
            for (int j = 0; j < l; ++j) {
                if (!std::isfinite(sxp[j]) || !std::isfinite(swp[j]) ||
                    sxp[j] <= 0.0 || swp[j] <= 0.0 ||
                    !std::isfinite(ap[j]) || ap[j] < 0.0) {
                    throw std::runtime_error("GxE native source scale/annotation is invalid");
                }
                for (int c = 0; c < v; ++c) {
                    const double probe = zp[static_cast<size_t>(c) * static_cast<size_t>(l) + static_cast<size_t>(j)];
                    if (!std::isfinite(probe)) {
                        throw std::runtime_error("GxE native source probe contains a non-finite value");
                    }
                    const size_t index = static_cast<size_t>(c) * static_cast<size_t>(l) + static_cast<size_t>(j);
                    weighted[index] = probe * ap[j] * sxp[j];
                    // Form the interaction weight directly.  Reusing the X
                    // weight through ``*(scale_w / scale_x)`` is
                    // algebraically unnecessary and can overflow at the
                    // intermediate ratio even when this final product is
                    // finite.
                    weighted[
                        index + static_cast<size_t>(l) * columns_size
                    ] = probe * ap[j] * swp[j];
                }
            }
            const auto compute_source = [&]() {
                return dgemm_nn_partitioned_rows(
                    n_, fused_columns, l, geno.get(), n_, weighted.data(), l,
                    fused_source, n_, blas_threads_
                );
            };
            try {
                record_gemm_repairs(compute_source());
            } catch (const RetryableGemmInputMutation&) {
                record_gemm_input_retry();
                std::unique_ptr<double[]> fresh_geno;
                std::vector<int> fresh_observed;
                missing = decode_block(
                    blk_start, blk_end, require_missing_free,
                    fresh_geno, fresh_observed
                );
                geno = std::move(fresh_geno);
                observed = std::move(fresh_observed);
                dgemm_nn_tiled(
                    n_, fused_columns, l, geno.get(), n_,
                    weighted.data(), l, fused_source, n_, blas_threads_
                );
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
            for (int c = 0; c < columns; ++c) {
                double* column = fused_source +
                    static_cast<size_t>(columns + c) * static_cast<size_t>(n_);
                for (int i = 0; i < n_; ++i) column[i] *= env_[static_cast<size_t>(i)];
            }
            project_panel_inplace(fused_source, fused_columns);
            validate_finite_output(
                fused_source,
                checked_mul(static_cast<size_t>(n_), fused_columns_size, "source output"),
                "fused source"
            );
            check_files_unchanged();
        }
        return nb::make_tuple(source_x_out, source_w_out, missing);
    }

    ProjectedPanel prepare_projected_sources(nb_mat2f_ro<double> sources,
                                             double tolerance) const {
        auto guard = acquire_call_lock();
        ensure_open();
        check_files_unchanged();
        const int columns = checked_blas_dim(sources.shape(1), "projected source columns");
        if (columns <= 0 || checked_blas_dim(sources.shape(0), "projected source rows") != n_) {
            throw std::runtime_error("GxE native projected source shape mismatch");
        }
        if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
            throw std::runtime_error("GxE native projected source tolerance must be finite and nonnegative");
        }
        const size_t source_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(columns), "projected sources"
        );
        const size_t coefficient_elements = checked_mul(
            static_cast<size_t>(q_), static_cast<size_t>(columns),
            "projected source coefficients"
        );
        ensure_workspace(
            checked_add(
                checked_add(
                    checked_mul(2U, source_elements, "projected source snapshots"),
                    coefficient_elements, "projected source preparation"
                ),
                partitioned_gemm_integrity_workspace_elements(
                    q_, columns, n_
                ),
                "projected source integrity workspace"
            ),
            "projected source preparation"
        );
        const size_t snapshot_elements = checked_mul(
            2U, source_elements, "projected source snapshots"
        );
        const size_t source_column_bytes = checked_mul(
            static_cast<size_t>(n_), sizeof(double), "projected source column"
        );
        // Both halves are fully populated below. Leave the allocation
        // untouched until parallel column loops write it so large panels are
        // distributed across the decoder's NUMA nodes.
        std::unique_ptr<double[]> snapshot(new double[snapshot_elements]);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int column = 0; column < columns; ++column) {
            std::memcpy(
                snapshot.get() + static_cast<size_t>(column) * static_cast<size_t>(n_),
                sources.data() + static_cast<size_t>(column) * static_cast<size_t>(n_),
                source_column_bytes
            );
        }
        double leakage = 0.0;
        {
            nb::gil_scoped_release release;
            leakage = projected_source_leakage(snapshot.get(), columns);
            if (leakage > tolerance) {
                throw std::runtime_error(
                    "GxE native source is not in the projected fixed-effect complement: leakage=" +
                    std::to_string(leakage) + ", tolerance=" + std::to_string(tolerance)
                );
            }
            // Float32 sketch storage can reintroduce small fixed-effect
            // components after an exact source projection.  Seal the opaque
            // panel only after a float64 reprojection; target W-left products
            // use e*S and therefore require S itself to be projected.
            project_panel_inplace(snapshot.get(), columns);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
            for (int column = 0; column < columns; ++column) {
                const double* source = snapshot.get() +
                    static_cast<size_t>(column) * static_cast<size_t>(n_);
                double* weighted = snapshot.get() + source_elements +
                    static_cast<size_t>(column) * static_cast<size_t>(n_);
                for (int row = 0; row < n_; ++row) {
                    weighted[row] = env_[static_cast<size_t>(row)] * source[row];
                }
            }
            validate_finite_output(
                snapshot.get() + source_elements, source_elements,
                "environment-weighted projected sources"
            );
            check_files_unchanged();
        }
        MatrixFingerprint fingerprint{};
#if defined(GWLDCORE_GEMM_INTEGRITY)
        fingerprint = fingerprint_col_major_matrix(
            n_, checked_blas_dim(2U * static_cast<size_t>(columns),
                                 "projected source snapshot columns"),
            snapshot.get(), n_, decode_threads_
        );
#endif
        return ProjectedPanel(
            context_id_, n_, columns, leakage, snapshot_elements,
            fingerprint, std::move(snapshot)
        );
    }

    double validate_projected_sources(nb_mat2f_ro<double> sources,
                                      double tolerance) const {
        return prepare_projected_sources(sources, tolerance).leakage();
    }

    nb::tuple target_block(int blk_start,
                           int blk_end,
                           nb_vec1_ro<double> scale_x,
                           nb_vec1_ro<double> scale_w,
                           nb_mat2f_ro<double> sources,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int columns = checked_blas_dim(sources.shape(1), "target source columns");
        if (columns <= 0 || checked_blas_dim(sources.shape(0), "target source rows") != n_ ||
            checked_blas_dim(scale_x.shape(0), "target scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "target scale_w") != l) {
            throw std::runtime_error("GxE native target input shape mismatch");
        }
        const int width = std::min(target_panel_columns_, columns);
        const size_t source_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(columns), "target source snapshot"
        );
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "target genotype");
        elements = checked_add(elements, source_elements, "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, static_cast<size_t>(l), "target scale snapshots"), "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(l), static_cast<size_t>(columns), "target outputs"), "target outputs"), "target workspace");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(n_), static_cast<size_t>(width), "target panels"), "target panels"), "target workspace");
        elements = checked_add(elements, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(width), "target QTS"), "target workspace");
        elements = checked_add(
            elements,
            std::max(
                partitioned_gemm_integrity_workspace_elements(
                    q_, width, n_
                ),
                partitioned_gemm_integrity_workspace_elements(
                    l, width, n_
                )
            ),
            "target integrity workspace"
        );
        ensure_workspace(elements, "target block");

        const std::vector<double> scale_x_snapshot(
            scale_x.data(), scale_x.data() + static_cast<size_t>(l)
        );
        const std::vector<double> scale_w_snapshot(
            scale_w.data(), scale_w.data() + static_cast<size_t>(l)
        );
        const std::vector<double> source_snapshot(
            sources.data(), sources.data() + source_elements
        );

        double* work_x = nullptr;
        double* work_w = nullptr;
        auto work_x_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), static_cast<size_t>(columns), &work_x);
        auto work_w_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), static_cast<size_t>(columns), &work_w);
        int64_t missing = 0;
        double max_source_leakage = 0.0;
        {
            nb::gil_scoped_release release;
            validate_scales(scale_x_snapshot, scale_w_snapshot);
            validate_finite_output(source_snapshot.data(), source_elements, "target sources");
            std::unique_ptr<double[]> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            std::vector<double> projected(
                checked_mul(static_cast<size_t>(n_), static_cast<size_t>(width), "target projected panel"), 0.0
            );
            std::vector<double> env_projected(projected.size(), 0.0);
            std::vector<double> qts(
                checked_mul(static_cast<size_t>(q_), static_cast<size_t>(width), "target QTS"), 0.0
            );
            for (int c0 = 0; c0 < columns; c0 += width) {
                const int count = std::min(width, columns - c0);
                const double* source_panel = source_snapshot.data() + static_cast<size_t>(c0) * static_cast<size_t>(n_);
                const size_t panel_elements = checked_mul(
                    static_cast<size_t>(n_), static_cast<size_t>(count), "target panel"
                );
                record_gemm_repairs(dgemm_tn_partitioned_columns(
                    q_, count, n_, q_basis_.data(), n_, source_panel, n_,
                    qts.data(), q_, blas_threads_
                ));
                max_source_leakage = std::max(
                    max_source_leakage,
                    projected_source_leakage_from_coefficients(
                        source_panel, panel_elements, qts.data(),
                        checked_mul(static_cast<size_t>(q_), static_cast<size_t>(count), "target coefficients")
                    )
                );
                std::memcpy(
                    projected.data(), source_panel,
                    checked_mul(panel_elements, sizeof(double), "target panel copy")
                );
                record_gemm_repairs(dgemm_nn_partitioned_rows(
                    n_, count, q_, q_basis_.data(), n_, qts.data(), q_,
                    projected.data(), n_, blas_threads_, -1.0, 1.0
                ));
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
                for (int c = 0; c < count; ++c) {
                    const double* source = projected.data() + static_cast<size_t>(c) * static_cast<size_t>(n_);
                    double* weighted = env_projected.data() + static_cast<size_t>(c) * static_cast<size_t>(n_);
                    for (int i = 0; i < n_; ++i) weighted[i] = env_[static_cast<size_t>(i)] * source[i];
                }
                const auto compute_target = [&](const double* right,
                                                double* output) {
                    return dgemm_tn_partitioned_rows(
                        l, count, n_, geno.get(), n_, right, n_,
                        output, l, blas_threads_
                    );
                };
                const auto run_with_fresh_decode = [&](const double* right,
                                                       double* output) {
                    try {
                        record_gemm_repairs(compute_target(right, output));
                    } catch (const RetryableGemmInputMutation&) {
                        record_gemm_input_retry();
                        std::unique_ptr<double[]> fresh_geno;
                        std::vector<int> fresh_observed;
                        missing = decode_block(
                            blk_start, blk_end, require_missing_free,
                            fresh_geno, fresh_observed
                        );
                        geno = std::move(fresh_geno);
                        observed = std::move(fresh_observed);
                        dgemm_tn_tiled(
                            l, count, n_, geno.get(), n_, right, n_,
                            output, l, blas_threads_
                        );
                    }
                };
                run_with_fresh_decode(
                    projected.data(),
                    work_x + static_cast<size_t>(c0) * static_cast<size_t>(l)
                );
                run_with_fresh_decode(
                    env_projected.data(),
                    work_w + static_cast<size_t>(c0) * static_cast<size_t>(l)
                );
            }
            scale_target_outputs(work_x, work_w, l, columns, scale_x_snapshot, scale_w_snapshot);
            check_files_unchanged();
        }
        return nb::make_tuple(work_x_out, work_w_out, missing, max_source_leakage);
    }

    nb::tuple target_projected_block(int blk_start,
                                     int blk_end,
                                     nb_vec1_ro<double> scale_x,
                                     nb_vec1_ro<double> scale_w,
                                     const ProjectedPanel& sources,
                                     bool require_missing_free) const {
        return target_projected_panel(
            blk_start, blk_end, scale_x, scale_w, sources,
            require_missing_free
        );
    }

    nb::dict phenotype_score_block(int blk_start,
                                    int blk_end,
                                    const ProjectedPanel& phenotype,
                                    double eps_var,
                                    bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        validate_projected_panel_handle(phenotype);
        if (!(eps_var > 0.0) || !std::isfinite(eps_var)) {
            throw std::runtime_error(
                "GxE native phenotype-score eps_var must be positive and finite"
            );
        }
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int columns = phenotype.columns_;
        const int fused_columns = checked_blas_dim(
            checked_mul(
                2U, static_cast<size_t>(columns),
                "fused phenotype-score columns"
            ),
            "fused phenotype-score columns"
        );
        const int moment_rows = 4 * q_;
        const size_t moment_copies = strict_feature_moment_verification_ ? 8U : 4U;
        size_t elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(l),
            "phenotype-score genotype"
        );
        elements = checked_add(
            elements,
            checked_mul(
                moment_copies,
                checked_mul(
                    static_cast<size_t>(q_), static_cast<size_t>(l),
                    "phenotype-score feature moments"
                ),
                "phenotype-score feature moments"
            ),
            "phenotype-score workspace"
        );
        elements = checked_add(
            elements,
            checked_mul(11U, static_cast<size_t>(l), "phenotype-score feature vectors"),
            "phenotype-score workspace"
        );
        const size_t score_elements = checked_mul(
            static_cast<size_t>(l), static_cast<size_t>(columns),
            "phenotype-score outputs"
        );
        elements = checked_add(
            elements,
            checked_mul(4U, score_elements, "phenotype-score output and fused buffers"),
            "phenotype-score workspace"
        );
        elements = checked_add(
            elements,
            std::max(
                partitioned_gemm_integrity_workspace_elements(
                    moment_rows, l, n_
                ),
                partitioned_gemm_integrity_workspace_elements(
                    l, fused_columns, n_
                )
            ),
            "phenotype-score integrity workspace"
        );
        ensure_workspace(elements, "phenotype-score block");

        double* score_x = nullptr;
        double* score_w = nullptr;
        double* scale_x = nullptr;
        double* scale_w = nullptr;
        double* norm_x = nullptr;
        double* norm_w = nullptr;
        double* diag_x = nullptr;
        double* diag_w = nullptr;
        double* corr_xw = nullptr;
        auto score_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(l), static_cast<size_t>(columns), &score_x
        );
        auto score_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(l), static_cast<size_t>(columns), &score_w
        );
        auto scale_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &scale_x);
        auto scale_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &scale_w);
        auto norm_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &norm_x);
        auto norm_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &norm_w);
        auto diag_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &diag_x);
        auto diag_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &diag_w);
        auto corr_xw_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &corr_xw);

        int64_t missing = 0;
        int64_t repaired_feature_moment_columns = 0;
        double max_leak_x = 0.0;
        double max_leak_w = 0.0;
        {
            nb::gil_scoped_release release;
            std::unique_ptr<double[]> geno;
            std::vector<int> observed;
            missing = decode_block(
                blk_start, blk_end, require_missing_free, geno, observed
            );
            compute_feature_moments_from_genotype(
                geno.get(), l, eps_var,
                scale_x, scale_w, norm_x, norm_w,
                diag_x, diag_w, corr_xw,
                repaired_feature_moment_columns, max_leak_x, max_leak_w
            );

            std::vector<double> fused_scores(
                checked_mul(2U, score_elements, "fused phenotype scores"),
                0.0
            );
            const auto compute_scores = [&]() {
                return dgemm_tn_partitioned_rows(
                    l, fused_columns, n_, geno.get(), n_,
                    phenotype.data_.get(), n_, fused_scores.data(), l,
                    blas_threads_, 1.0, 0.0, &phenotype.fingerprint_
                );
            };
            try {
                record_gemm_repairs(compute_scores());
            } catch (const RetryableGemmInputMutation&) {
                record_gemm_input_retry();
                std::unique_ptr<double[]> fresh_geno;
                std::vector<int> fresh_observed;
                missing = decode_block(
                    blk_start, blk_end, require_missing_free,
                    fresh_geno, fresh_observed
                );
                geno = std::move(fresh_geno);
                observed = std::move(fresh_observed);
                // The rejected score call may have changed the decoded left
                // operand after feature moments were derived.  Keep the
                // guarded process-shared fallback internally consistent by
                // rebuilding both scales and diagnostics from the same fresh
                // decode used for the fallback score product.  Private-static
                // guard-free builds cannot enter this branch.
                compute_feature_moments_from_genotype(
                    geno.get(), l, eps_var,
                    scale_x, scale_w, norm_x, norm_w,
                    diag_x, diag_w, corr_xw,
                    repaired_feature_moment_columns, max_leak_x, max_leak_w
                );
                dgemm_tn_tiled(
                    l, fused_columns, n_, geno.get(), n_,
                    phenotype.data_.get(), n_, fused_scores.data(), l,
                    blas_threads_
                );
            }
            std::memcpy(
                score_x, fused_scores.data(),
                checked_mul(score_elements, sizeof(double), "additive phenotype scores")
            );
            std::memcpy(
                score_w, fused_scores.data() + score_elements,
                checked_mul(score_elements, sizeof(double), "interaction phenotype scores")
            );
            const std::vector<double> scale_x_snapshot(
                scale_x, scale_x + static_cast<size_t>(l)
            );
            const std::vector<double> scale_w_snapshot(
                scale_w, scale_w + static_cast<size_t>(l)
            );
            scale_target_outputs(
                score_x, score_w, l, columns,
                scale_x_snapshot, scale_w_snapshot
            );
            check_files_unchanged();
        }

        nb::dict result;
        result["score_x"] = score_x_out;
        result["score_w"] = score_w_out;
        result["scale_x"] = scale_x_out;
        result["scale_w"] = scale_w_out;
        result["norm_x"] = norm_x_out;
        result["norm_w"] = norm_w_out;
        result["diag_nxe_x"] = diag_x_out;
        result["diag_nxe_w"] = diag_w_out;
        result["corr_xw"] = corr_xw_out;
        result["max_projection_leakage_additive"] = max_leak_x;
        result["max_projection_leakage_interaction"] = max_leak_w;
        result["phenotype_projection_leakage"] = phenotype.leakage_;
        result["repaired_feature_moment_columns"] = repaired_feature_moment_columns;
        result["missing_genotype_calls"] = missing;
        return result;
    }

private:
    friend class MultiEnvironmentDirectContext;
    friend class GeneralizedGxELDScoreDirectContext;
    void compute_feature_moments_from_genotype(
        const double* genotype,
        int columns,
        double eps_var,
        double* scale_x,
        double* scale_w,
        double* norm_x,
        double* norm_w,
        double* diag_x,
        double* diag_w,
        double* corr_xw,
        int64_t& repaired_feature_moment_columns,
        double& max_leak_x,
        double& max_leak_w
    ) const {
        const int rank = n_ - q_;
        const int moment_rows = 4 * q_;
        std::vector<double> moments(
            checked_mul(
                static_cast<size_t>(moment_rows),
                static_cast<size_t>(columns),
                "feature moments"
            ),
            0.0
        );
        std::vector<double> moment_verification;
        if (strict_feature_moment_verification_) {
            moment_verification.assign(moments.size(), 0.0);
        }
        std::vector<double> scalar(
            checked_mul(4U, static_cast<size_t>(columns), "feature scalar moments"),
            0.0
        );
        double* s0 = scalar.data();
        double* s1 = s0 + columns;
        double* s2 = s1 + columns;
        double* s4 = s2 + columns;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int j = 0; j < columns; ++j) {
            const double* column = genotype +
                static_cast<size_t>(j) * static_cast<size_t>(n_);
            double a0 = 0.0;
            double a1 = 0.0;
            double a2 = 0.0;
            double a4 = 0.0;
            for (int i = 0; i < n_; ++i) {
                const double environment = env_[static_cast<size_t>(i)];
                const double genotype_squared = column[i] * column[i];
                const double environment_squared = environment * environment;
                a0 += genotype_squared;
                a1 += environment * genotype_squared;
                a2 += environment_squared * genotype_squared;
                a4 += environment_squared * environment_squared * genotype_squared;
            }
            s0[j] = a0;
            s1[j] = a1;
            s2[j] = a2;
            s4[j] = a4;
        }
        // [Q, E Q, E^2 Q, E^3 Q]^T G supplies all exact projected
        // feature moments without materializing additive or interaction panels.
        record_gemm_repairs(dgemm_tn_partitioned_columns(
            moment_rows, columns, n_, feature_moment_basis_.data(), n_,
            genotype, n_, moments.data(), moment_rows, blas_threads_
        ));
        if (strict_feature_moment_verification_) {
            dgemm_tn_tiled(
                moment_rows, columns, n_, feature_moment_basis_.data(), n_,
                genotype, n_, moment_verification.data(), moment_rows,
                blas_threads_
            );
        }
        for (int j = 0; j < columns; ++j) {
            bool disagrees = false;
            double* moment_column = moments.data() +
                static_cast<size_t>(j) * static_cast<size_t>(moment_rows);
            if (strict_feature_moment_verification_) {
                const double* repeated = moment_verification.data() +
                    static_cast<size_t>(j) * static_cast<size_t>(moment_rows);
                for (int row = 0; row < moment_rows; ++row) {
                    const double first = moment_column[row];
                    const double second = repeated[row];
                    const double tolerance = 1.0e-12 * std::max(
                        1.0, std::max(std::abs(first), std::abs(second))
                    );
                    if (!std::isfinite(first) || !std::isfinite(second) ||
                        std::abs(first - second) > tolerance) {
                        disagrees = true;
                        break;
                    }
                }
                if (!disagrees) {
                    std::memcpy(
                        moment_column, repeated,
                        checked_mul(
                            static_cast<size_t>(moment_rows), sizeof(double),
                            "strictly verified feature moments"
                        )
                    );
                }
            }
            if (disagrees) {
                ++repaired_feature_moment_columns;
                record_gemm_repairs(1);
                const double* genotype_column = genotype +
                    static_cast<size_t>(j) * static_cast<size_t>(n_);
                for (int row = 0; row < moment_rows; ++row) {
                    const double* basis_column = feature_moment_basis_.data() +
                        static_cast<size_t>(row) * static_cast<size_t>(n_);
                    long double dot = 0.0L;
                    for (int i = 0; i < n_; ++i) {
                        dot += static_cast<long double>(basis_column[i]) *
                            static_cast<long double>(genotype_column[i]);
                    }
                    moment_column[row] = static_cast<double>(dot);
                }
            }
        }
        for (int j = 0; j < columns; ++j) {
            const double* u0 = moments.data() +
                static_cast<size_t>(j) * 4U * static_cast<size_t>(q_);
            const double* u1 = u0 + q_;
            const double* u2 = u1 + q_;
            const double* u3 = u2 + q_;
            double u00 = 0.0;
            double u11 = 0.0;
            double u01 = 0.0;
            double u0u2 = 0.0;
            double u1u3 = 0.0;
            double u0e2u0 = 0.0;
            double u1e2u1 = 0.0;
            double leak_x_sq = 0.0;
            double leak_w_sq = 0.0;
            for (int a = 0; a < q_; ++a) {
                u00 += u0[a] * u0[a];
                u11 += u1[a] * u1[a];
                u01 += u0[a] * u1[a];
                u0u2 += u0[a] * u2[a];
                u1u3 += u1[a] * u3[a];
                double e2u0 = 0.0;
                double e2u1 = 0.0;
                double gram_u0 = 0.0;
                double gram_u1 = 0.0;
                for (int b = 0; b < q_; ++b) {
                    const size_t index =
                        static_cast<size_t>(b) * static_cast<size_t>(q_) +
                        static_cast<size_t>(a);
                    e2u0 += q_e2_q_[index] * u0[b];
                    e2u1 += q_e2_q_[index] * u1[b];
                    gram_u0 += q_gram_[index] * u0[b];
                    gram_u1 += q_gram_[index] * u1[b];
                }
                u0e2u0 += u0[a] * e2u0;
                u1e2u1 += u1[a] * e2u1;
                const double lx = u0[a] - gram_u0;
                const double lw = u1[a] - gram_u1;
                leak_x_sq += lx * lx;
                leak_w_sq += lw * lw;
            }
            const double ssx = s0[j] - u00;
            const double ssw = s2[j] - u11;
            const double varx = ssx / static_cast<double>(rank);
            const double varw = ssw / static_cast<double>(rank);
            if (!std::isfinite(varx) || !std::isfinite(varw) ||
                varx <= eps_var || varw <= eps_var) {
                throw std::runtime_error(
                    "GxE native projected feature has zero or invalid variance at block offset " +
                    std::to_string(j)
                );
            }
            scale_x[j] = 1.0 / std::sqrt(varx);
            scale_w[j] = 1.0 / std::sqrt(varw);
            norm_x[j] = scale_x[j] * scale_x[j] * ssx /
                static_cast<double>(rank);
            norm_w[j] = scale_w[j] * scale_w[j] * ssw /
                static_cast<double>(rank);
            diag_x[j] = scale_x[j] * scale_x[j] *
                (s2[j] - 2.0 * u0u2 + u0e2u0) /
                static_cast<double>(rank);
            diag_w[j] = scale_w[j] * scale_w[j] *
                (s4[j] - 2.0 * u1u3 + u1e2u1) /
                static_cast<double>(rank);
            corr_xw[j] = scale_x[j] * scale_w[j] * (s1[j] - u01) /
                static_cast<double>(rank);
            max_leak_x = std::max(
                max_leak_x,
                std::sqrt(
                    std::max(0.0, leak_x_sq) /
                    std::max(ssx, std::numeric_limits<double>::min())
                )
            );
            max_leak_w = std::max(
                max_leak_w,
                std::sqrt(
                    std::max(0.0, leak_w_sq) /
                    std::max(ssw, std::numeric_limits<double>::min())
                )
            );
            if (!std::isfinite(scale_x[j]) || !std::isfinite(scale_w[j]) ||
                !std::isfinite(norm_x[j]) || !std::isfinite(norm_w[j]) ||
                !std::isfinite(diag_x[j]) || !std::isfinite(diag_w[j]) ||
                !std::isfinite(corr_xw[j])) {
                throw std::runtime_error(
                    "GxE native feature output contains a non-finite value"
                );
            }
        }
    }

    double projected_source_leakage_from_coefficients(
        const double* sources,
        size_t source_elements,
        const double* coefficients,
        size_t coefficient_elements) const {
        long double source_ss = 0.0L;
        long double coefficient_ss = 0.0L;
        for (size_t index = 0; index < source_elements; ++index) {
            const long double value = static_cast<long double>(sources[index]);
            source_ss += value * value;
        }
        for (size_t index = 0; index < coefficient_elements; ++index) {
            const long double value = static_cast<long double>(coefficients[index]);
            coefficient_ss += value * value;
        }
        const double leakage = std::sqrt(static_cast<double>(
            coefficient_ss /
            std::max(source_ss, static_cast<long double>(std::numeric_limits<double>::min()))
        ));
        if (!std::isfinite(leakage)) {
            throw std::runtime_error("GxE native projected source diagnostic is non-finite");
        }
        return leakage;
    }

    double projected_source_leakage(const double* sources, int columns) const {
        const size_t source_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(columns), "projected sources"
        );
        const size_t coefficient_elements = checked_mul(
            static_cast<size_t>(q_), static_cast<size_t>(columns),
            "projected source coefficients"
        );
        validate_finite_output(sources, source_elements, "projected sources");
        std::vector<double> coefficients(coefficient_elements, 0.0);
        record_gemm_repairs(dgemm_tn_partitioned_columns(
            q_, columns, n_, q_basis_.data(), n_, sources, n_,
            coefficients.data(), q_, blas_threads_
        ));
        return projected_source_leakage_from_coefficients(
            sources, source_elements, coefficients.data(), coefficient_elements
        );
    }

    void validate_scales(const std::vector<double>& scale_x,
                         const std::vector<double>& scale_w) const {
        if (scale_x.size() != scale_w.size()) {
            throw std::runtime_error("GxE native target scale snapshots disagree");
        }
        for (size_t index = 0; index < scale_x.size(); ++index) {
            if (!std::isfinite(scale_x[index]) || !std::isfinite(scale_w[index]) ||
                scale_x[index] <= 0.0 || scale_w[index] <= 0.0) {
                throw std::runtime_error("GxE native target feature scale is invalid");
            }
        }
    }

    void scale_target_outputs(double* work_x,
                              double* work_w,
                              int rows,
                              int columns,
                              const std::vector<double>& scale_x,
                              const std::vector<double>& scale_w) const {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int column = 0; column < columns; ++column) {
            double* x = work_x + static_cast<size_t>(column) * static_cast<size_t>(rows);
            double* w = work_w + static_cast<size_t>(column) * static_cast<size_t>(rows);
            for (int row = 0; row < rows; ++row) {
                x[row] *= scale_x[static_cast<size_t>(row)];
                w[row] *= scale_w[static_cast<size_t>(row)];
            }
        }
        const size_t output_elements = checked_mul(
            static_cast<size_t>(rows), static_cast<size_t>(columns), "target output"
        );
        validate_finite_output(work_x, output_elements, "work_x");
        validate_finite_output(work_w, output_elements, "work_w");
    }

    void validate_projected_panel_handle(const ProjectedPanel& panel) const {
        if (panel.context_id_ != context_id_ || panel.rows_ != n_ ||
            panel.columns_ <= 0 ||
            panel.data_ == nullptr || panel.elements_ != checked_mul(
                2U, checked_mul(
                    static_cast<size_t>(n_), static_cast<size_t>(panel.columns_),
                    "opaque projected panel"
                ),
                "opaque projected panel"
            )) {
            throw std::runtime_error(
                "GxE native projected panel does not belong to this context"
            );
        }
    }

    nb::tuple target_projected_panel(
        int blk_start,
        int blk_end,
        nb_vec1_ro<double> scale_x,
        nb_vec1_ro<double> scale_w,
        const ProjectedPanel& sources,
        bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        validate_projected_panel_handle(sources);
        const int l = blk_end - blk_start;
        const size_t columns_size = static_cast<size_t>(sources.columns_);
        const int columns = checked_blas_dim(columns_size, "projected target columns");
        if (checked_blas_dim(scale_x.shape(0), "target scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "target scale_w") != l) {
            throw std::runtime_error("GxE native target input shape mismatch");
        }
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "target genotype");
        elements = checked_add(elements, checked_mul(2U, static_cast<size_t>(l), "target scale snapshots"), "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(l), columns_size, "target outputs"), "target outputs"), "target workspace");
        const size_t widest_panel = static_cast<size_t>(sources.columns_);
        const int widest_fused_columns = checked_blas_dim(
            checked_mul(2U, widest_panel, "fused projected target columns"),
            "fused projected target columns"
        );
        elements = checked_add(
            elements,
            checked_mul(
                2U,
                checked_mul(static_cast<size_t>(l), widest_panel, "fused target output"),
                "fused target output"
            ),
            "target workspace"
        );
        elements = checked_add(
            elements,
            partitioned_gemm_integrity_workspace_elements(
                l, widest_fused_columns, n_
            ),
            "projected target integrity workspace"
        );
        ensure_workspace(elements, "projected target block");

        const std::vector<double> scale_x_snapshot(
            scale_x.data(), scale_x.data() + static_cast<size_t>(l)
        );
        const std::vector<double> scale_w_snapshot(
            scale_w.data(), scale_w.data() + static_cast<size_t>(l)
        );
        double* work_x = nullptr;
        double* work_w = nullptr;
        auto work_x_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), columns_size, &work_x);
        auto work_w_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), columns_size, &work_w);
        int64_t missing = 0;
        {
            nb::gil_scoped_release release;
            validate_scales(scale_x_snapshot, scale_w_snapshot);
            std::unique_ptr<double[]> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            const size_t fused_work_elements = checked_mul(
                2U,
                checked_mul(static_cast<size_t>(l), widest_panel, "fused target output"),
                "fused target output"
            );
            std::unique_ptr<double[]> fused_work(
                new double[fused_work_elements]
            );
            size_t output_offset = 0;
            auto consume = [&](const ProjectedPanel& panel) {
                const int fused_columns = checked_blas_dim(
                    checked_mul(
                        2U, static_cast<size_t>(panel.columns_),
                        "fused projected target columns"
                    ),
                    "fused projected target columns"
                );
                const size_t panel_output_elements = checked_mul(
                    static_cast<size_t>(l), static_cast<size_t>(panel.columns_),
                    "projected target output"
                );
                // ProjectedPanel owns one immutable [S, E*S] allocation.  A
                // single wide product reuses the decoded G block for both X-
                // and W-left work instead of packing/reading it twice.
                const auto compute_target = [&]() {
                    return dgemm_tn_partitioned_rows(
                        l, fused_columns, n_, geno.get(), n_,
                        panel.data_.get(), n_,
                        fused_work.get(), l, blas_threads_,
                        1.0, 0.0, &panel.fingerprint_
                    );
                };
                try {
                    record_gemm_repairs(compute_target());
                } catch (const RetryableGemmInputMutation&) {
                    record_gemm_input_retry();
                    std::unique_ptr<double[]> fresh_geno;
                    std::vector<int> fresh_observed;
                    missing = decode_block(
                        blk_start, blk_end, require_missing_free,
                        fresh_geno, fresh_observed
                    );
                    geno = std::move(fresh_geno);
                    observed = std::move(fresh_observed);
                    dgemm_tn_tiled(
                        l, fused_columns, n_, geno.get(), n_,
                        panel.data_.get(), n_, fused_work.get(), l,
                        blas_threads_
                    );
                }
                std::memcpy(
                    work_x + output_offset * static_cast<size_t>(l),
                    fused_work.get(),
                    checked_mul(panel_output_elements, sizeof(double), "work_x copy")
                );
                std::memcpy(
                    work_w + output_offset * static_cast<size_t>(l),
                    fused_work.get() + panel_output_elements,
                    checked_mul(panel_output_elements, sizeof(double), "work_w copy")
                );
                output_offset += static_cast<size_t>(panel.columns_);
            };
            consume(sources);
            scale_target_outputs(
                work_x, work_w, l, columns, scale_x_snapshot, scale_w_snapshot
            );
            check_files_unchanged();
        }
        return nb::make_tuple(
            work_x_out, work_w_out, missing, sources.leakage_
        );
    }

    std::unique_lock<std::mutex> acquire_call_lock() const {
        nb::gil_scoped_release release;
        return std::unique_lock<std::mutex>(call_mutex_);
    }

    void ensure_open() const {
        if (closed_ || bed_base_ == nullptr) {
            throw std::runtime_error("GxE native context is closed");
        }
    }

    void validate_block(int blk_start, int blk_end) const {
        ensure_open();
        if (blk_start < 0 || blk_end <= blk_start || blk_end > m_total_) {
            throw std::runtime_error("GxE native BED block is outside the BIM axis");
        }
    }

    void ensure_workspace(size_t elements, const char* label) const {
        const size_t bytes = checked_mul(elements, sizeof(double), label);
        if (static_cast<uint64_t>(bytes) > max_workspace_bytes_) {
            throw std::runtime_error(
                std::string("GxE native ") + label + " requires " +
                std::to_string(bytes) + " workspace bytes, exceeding the configured limit " +
                std::to_string(max_workspace_bytes_)
            );
        }
    }

    void parse_rows(nb::object row_sel_obj) {
        if (row_sel_obj.is_none()) {
            rows_.resize(static_cast<size_t>(n_total_));
            for (int i = 0; i < n_total_; ++i) rows_[static_cast<size_t>(i)] = i;
        } else {
            nb_any_array_ro idx = nb::cast<nb_any_array_ro>(row_sel_obj);
            if (idx.ndim() != 1 || idx.shape(0) > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("GxE native row_sel must be a bounded one-dimensional integer array");
            }
            rows_.resize(idx.shape(0));
            if (idx.dtype() == nb::dtype<int32_t>()) {
                auto view = idx.view<const int32_t, nb::ndim<1>>();
                for (size_t i = 0; i < rows_.size(); ++i) rows_[i] = static_cast<int>(view(i));
            } else if (idx.dtype() == nb::dtype<int64_t>()) {
                auto view = idx.view<const int64_t, nb::ndim<1>>();
                for (size_t i = 0; i < rows_.size(); ++i) {
                    const int64_t value = view(i);
                    if (value < 0 || value > std::numeric_limits<int>::max()) {
                        throw std::runtime_error("GxE native row_sel contains an out-of-range sample index");
                    }
                    rows_[i] = static_cast<int>(value);
                }
            } else {
                throw std::runtime_error("GxE native row_sel must have dtype int32 or int64");
            }
        }
        if (rows_.size() < 3) {
            throw std::runtime_error("GxE native context requires at least three selected samples");
        }
        int previous = -1;
        for (int row : rows_) {
            if (row < 0 || row >= n_total_ || row <= previous) {
                throw std::runtime_error("GxE native row_sel must be strictly increasing and within the FAM axis");
            }
            previous = row;
        }
        n_ = checked_blas_dim(rows_.size(), "selected samples");
    }

    void copy_and_validate_design(nb_vec1_ro<double> env, nb_mat2f_ro<double> q_basis) {
        if (checked_blas_dim(env.shape(0), "environment") != n_ ||
            checked_blas_dim(q_basis.shape(0), "projection rows") != n_) {
            throw std::runtime_error("GxE native environment/projection row mismatch");
        }
        q_ = checked_blas_dim(q_basis.shape(1), "projection rank");
        if (q_ < 1 || q_ >= n_) {
            throw std::runtime_error("GxE native projection basis has invalid rank");
        }
        env_.assign(env.data(), env.data() + static_cast<size_t>(n_));
        q_basis_.assign(
            q_basis.data(),
            q_basis.data() + checked_mul(static_cast<size_t>(n_), static_cast<size_t>(q_), "projection basis")
        );
        long double env_sum = 0.0L;
        long double env_ss = 0.0L;
        for (double value : env_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("GxE native environment contains a non-finite value");
            }
            env_sum += static_cast<long double>(value);
            env_ss += static_cast<long double>(value) * static_cast<long double>(value);
        }
        environment_mean_ = static_cast<double>(env_sum / static_cast<long double>(n_));
        environment_variance_ = static_cast<double>(env_ss / static_cast<long double>(n_ - ddof_));
        if (std::abs(environment_mean_) > kStandardizedEnvTolerance ||
            !std::isfinite(environment_variance_) ||
            std::abs(environment_variance_ - 1.0) > kStandardizedEnvTolerance) {
            throw std::runtime_error("GxE native environment must be nonconstant, centered, and standardized for the configured ddof");
        }
        const int moment_rows = 4 * q_;
        feature_moment_basis_.resize(
            checked_mul(
                static_cast<size_t>(n_), static_cast<size_t>(moment_rows),
                "feature moment basis"
            )
        );
        for (int a = 0; a < q_; ++a) {
            const double* source = q_basis_.data() +
                static_cast<size_t>(a) * static_cast<size_t>(n_);
            for (int i = 0; i < n_; ++i) {
                const double environment = env_[static_cast<size_t>(i)];
                double multiplier = 1.0;
                for (int power = 0; power < 4; ++power) {
                    feature_moment_basis_[
                        static_cast<size_t>(power * q_ + a) *
                            static_cast<size_t>(n_) +
                        static_cast<size_t>(i)
                    ] = source[i] * multiplier;
                    multiplier *= environment;
                }
            }
        }
        q_gram_.assign(checked_mul(static_cast<size_t>(q_), static_cast<size_t>(q_), "Q Gram"), 0.0);
        q_e2_q_.assign(q_gram_.size(), 0.0);
        max_q_gram_error_ = 0.0;
        for (int b = 0; b < q_; ++b) {
            for (int a = 0; a < q_; ++a) {
                double gram = 0.0;
                double e2gram = 0.0;
                for (int i = 0; i < n_; ++i) {
                    const double qa = q_basis_[static_cast<size_t>(a) * static_cast<size_t>(n_) + static_cast<size_t>(i)];
                    const double qb = q_basis_[static_cast<size_t>(b) * static_cast<size_t>(n_) + static_cast<size_t>(i)];
                    if (!std::isfinite(qa) || !std::isfinite(qb)) {
                        throw std::runtime_error("GxE native projection basis contains a non-finite value");
                    }
                    gram += qa * qb;
                    e2gram += qa * qb * env_[static_cast<size_t>(i)] * env_[static_cast<size_t>(i)];
                }
                const size_t index = static_cast<size_t>(b) * static_cast<size_t>(q_) + static_cast<size_t>(a);
                q_gram_[index] = gram;
                q_e2_q_[index] = e2gram;
                max_q_gram_error_ = std::max(max_q_gram_error_, std::abs(gram - ((a == b) ? 1.0 : 0.0)));
            }
        }
        if (max_q_gram_error_ > kOrthonormalTolerance) {
            throw std::runtime_error("GxE native projection basis is not orthonormal");
        }
        std::vector<double> intercept_coeff(static_cast<size_t>(q_), 0.0);
        std::vector<double> env_coeff(static_cast<size_t>(q_), 0.0);
        const double root_n = std::sqrt(static_cast<double>(n_));
        for (int a = 0; a < q_; ++a) {
            const double* column = q_basis_.data() + static_cast<size_t>(a) * static_cast<size_t>(n_);
            for (int i = 0; i < n_; ++i) {
                intercept_coeff[static_cast<size_t>(a)] += column[i] / root_n;
                env_coeff[static_cast<size_t>(a)] += column[i] * env_[static_cast<size_t>(i)];
            }
        }
        double intercept_resid_ss = 0.0;
        double env_resid_ss = 0.0;
        for (int i = 0; i < n_; ++i) {
            double fitted_intercept = 0.0;
            double fitted_env = 0.0;
            for (int a = 0; a < q_; ++a) {
                const double value = q_basis_[static_cast<size_t>(a) * static_cast<size_t>(n_) + static_cast<size_t>(i)];
                fitted_intercept += value * intercept_coeff[static_cast<size_t>(a)];
                fitted_env += value * env_coeff[static_cast<size_t>(a)];
            }
            const double ri = 1.0 / root_n - fitted_intercept;
            const double re = env_[static_cast<size_t>(i)] - fitted_env;
            intercept_resid_ss += ri * ri;
            env_resid_ss += re * re;
        }
        if (std::sqrt(intercept_resid_ss) > 1.0e-9 ||
            std::sqrt(env_resid_ss / std::max(static_cast<double>(env_ss), std::numeric_limits<double>::min())) > 1.0e-9) {
            throw std::runtime_error("GxE native projection basis must span the intercept and environment");
        }
    }

    int64_t decode_block(int blk_start,
                         int blk_end,
                         bool require_missing_free,
                         std::unique_ptr<double[]>& geno,
                         std::vector<int>& observed) const {
        const int l = blk_end - blk_start;
        const size_t genotype_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(l), "decoded genotype"
        );
        // Every cell is assigned in the parallel decode loop. An uninitialized
        // allocation avoids a redundant serial write and gives correct NUMA
        // first-touch placement for the subsequent BLAS read.
        geno.reset(new double[genotype_elements]);
        return decode_block_into(
            blk_start, blk_end, require_missing_free, geno.get(), observed
        );
    }

    int64_t decode_block_into(int blk_start,
                              int blk_end,
                              bool require_missing_free,
                              double* geno,
                              std::vector<int>& observed,
                              double* affine_mean = nullptr,
                              double* affine_inverse_scale = nullptr) const {
        const int l = blk_end - blk_start;
        if (geno == nullptr || l <= 0) {
            throw std::runtime_error("GxE native decoded block storage is invalid");
        }
        observed.assign(static_cast<size_t>(l), 0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int j = 0; j < l; ++j) {
            const unsigned char* bytes = bed_base_ + 3 +
                static_cast<size_t>(blk_start + j) * bytes_per_snp_;
            double* column = geno +
                static_cast<size_t>(j) * static_cast<size_t>(n_);
            int nobs = 0;
            int64_t sum = 0;
            int64_t sumsq = 0;
            for (int i = 0; i < n_; ++i) {
                const int row = rows_[static_cast<size_t>(i)];
                const uint8_t bits = static_cast<uint8_t>((bytes[static_cast<size_t>(row >> 2)] >> ((row & 3) << 1)) & 0x3U);
                if (bits != 1U) {
                    const int value = (bits == 0U) ? 0 : (bits == 2U) ? 1 : 2;
                    column[i] = static_cast<double>(value);
                    ++nobs;
                    sum += value;
                    sumsq += value * value;
                } else {
                    // Three is outside the valid 0/1/2 dosage range and keeps
                    // missingness local to the already-written decode buffer.
                    column[i] = 3.0;
                }
            }
            observed[static_cast<size_t>(j)] = nobs;
            const double mean = (nobs > 0) ? static_cast<double>(sum) / static_cast<double>(nobs) : 0.0;
            double m2 = (nobs > 0)
                ? static_cast<double>(sumsq) - static_cast<double>(sum) * static_cast<double>(sum) / static_cast<double>(nobs)
                : 0.0;
            if (m2 < 0.0 && m2 > -1.0e-12) m2 = 0.0;
            const int denom = n_ - ddof_;
            const double inverse_sd = (denom > 0 && m2 > 0.0)
                ? std::sqrt(static_cast<double>(denom) / m2)
                : 1.0;
            if (affine_mean != nullptr) {
                // DirectContext standardizes the BIM A1 dosage, which is the
                // complement of the compact BED code used above.  Derive the
                // A1 mean from integer allele counts so its sealed identity is
                // independent of subtraction roundoff in 2 - compact_mean.
                affine_mean[j] = (nobs > 0)
                    ? static_cast<double>(2LL * nobs - sum)
                        / static_cast<double>(nobs)
                    : 0.0;
            }
            if (affine_inverse_scale != nullptr) {
                affine_inverse_scale[j] = inverse_sd;
            }
            for (int i = 0; i < n_; ++i) {
                const double value = column[i];
                if (value == 3.0) {
                    column[i] = 0.0;
                } else {
                    column[i] = (mean - value) * inverse_sd;
                }
            }
        }
#if defined(__linux__)
        const size_t consumed_offset = checked_add(
            3U,
            checked_mul(
                static_cast<size_t>(blk_start), bytes_per_snp_,
                "decoded BED offset"
            ),
            "decoded BED offset"
        );
        const size_t consumed_length = checked_mul(
            static_cast<size_t>(l), bytes_per_snp_, "decoded BED range"
        );
        madvise_dontneed_consumed_range(
            bed_base_, bed_size_, consumed_offset, consumed_length
        );
#endif
        int64_t missing = 0;
        for (int count : observed) missing += static_cast<int64_t>(n_ - count);
        if (require_missing_free && missing != 0) {
            throw std::runtime_error("GxE native production backend requires a missing-free selected BED block");
        }
        return missing;
    }

    void project_panel_inplace(double* panel, int columns) const {
        if (columns <= 0) return;
        std::vector<double> coefficients(
            checked_mul(static_cast<size_t>(q_), static_cast<size_t>(columns), "projection coefficients"), 0.0
        );
        record_gemm_repairs(dgemm_tn_partitioned_columns(
            q_, columns, n_, q_basis_.data(), n_, panel, n_,
            coefficients.data(), q_, blas_threads_
        ));
        record_gemm_repairs(dgemm_nn_partitioned_rows(
            n_, columns, q_, q_basis_.data(), n_, coefficients.data(), q_,
            panel, n_, blas_threads_, -1.0, 1.0
        ));
    }

    void validate_finite_output(const double* values, size_t count, const char* label) const {
        int invalid = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_) reduction(|:invalid)
#endif
        for (size_t i = 0; i < count; ++i) {
            invalid |= !std::isfinite(values[i]);
        }
        if (invalid != 0) {
            throw std::runtime_error(
                std::string("GxE native ") + label + " contains a non-finite value"
            );
        }
    }

    void record_gemm_repairs(int64_t count) const {
        if (count > 0) {
            repaired_gemm_output_columns_.fetch_add(
                count, std::memory_order_relaxed
            );
        }
    }

    void record_gemm_input_retry() const {
        retried_gemm_input_mutations_.fetch_add(
            1, std::memory_order_relaxed
        );
    }

    void check_files_unchanged() const {
#if defined(__linux__)
        ensure_open();
        if (!same_state(validate_regular_fd(bed_fd_, "BED"), bed_state_) ||
            !same_state(validate_regular_fd(bim_fd_, "BIM"), bim_state_) ||
            !same_state(validate_regular_fd(fam_fd_, "FAM"), fam_state_)) {
            throw std::runtime_error("GxE native PLINK input changed after context construction");
        }
#endif
    }

    void close_internal() noexcept {
#if defined(__linux__)
        if (bed_base_ != nullptr) {
            ::munmap(bed_base_, bed_size_);
            bed_base_ = nullptr;
        }
        for (int* descriptor : {&bed_fd_, &bim_fd_, &fam_fd_}) {
            if (*descriptor >= 0) {
                ::close(*descriptor);
                *descriptor = -1;
            }
        }
#endif
        closed_ = true;
    }

    const uint64_t context_id_;
    int ddof_ = 1;
    int decode_threads_ = 1;
    int blas_threads_ = 1;
    uint64_t max_workspace_bytes_ = 0;
    int target_panel_columns_ = 64;
    bool strict_feature_moment_verification_ = true;
    int n_total_ = 0;
    int m_total_ = 0;
    int n_ = 0;
    int q_ = 0;
    std::vector<int> rows_;
    std::vector<double> env_;
    std::vector<double> q_basis_;
    std::vector<double> feature_moment_basis_;
    std::vector<double> q_gram_;
    std::vector<double> q_e2_q_;
    double environment_mean_ = 0.0;
    double environment_variance_ = 0.0;
    double max_q_gram_error_ = 0.0;
    bool closed_ = false;
    mutable std::mutex call_mutex_;
    mutable std::atomic<int64_t> repaired_gemm_output_columns_{0};
    mutable std::atomic<int64_t> retried_gemm_input_mutations_{0};
#if defined(__linux__)
    int bed_fd_ = -1;
    int bim_fd_ = -1;
    int fam_fd_ = -1;
    FileState bed_state_{};
    FileState bim_state_{};
    FileState fam_state_{};
    unsigned char* bed_base_ = nullptr;
    size_t bed_size_ = 0;
    size_t bytes_per_snp_ = 0;
#else
    unsigned char* bed_base_ = nullptr;
#endif
};

void validate_protected_gemm_threads(int requested_threads) {
    if (requested_threads <= 0) {
        throw std::runtime_error("Protected GxE GEMM threads must be positive");
    }
#ifdef _OPENMP
    validate_configured_openmp_placement_for_entry();
    const int thread_limit = effective_openmp_capacity();
    if (requested_threads > thread_limit) {
        throw std::runtime_error(
            "Protected GxE GEMM threads exceed the OpenMP/CPU-affinity limit"
        );
    }
#else
    if (requested_threads != 1) {
        throw std::runtime_error(
            "Protected GxE GEMM threads must be one when OpenMP is unavailable"
        );
    }
#endif
}

nb::tuple standardize_genotype_block(
    nb_mat2f_rw<double> genotype,
    nb_mat2f_ro<double> missingness_targets,
    int ddof,
    bool hwe_scale,
    double eps,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        genotype.shape(0), "genotype standardization rows"
    );
    const int variants = checked_blas_dim(
        genotype.shape(1), "genotype standardization columns"
    );
    const int target_rows = checked_blas_dim(
        missingness_targets.shape(0), "missingness-target rows"
    );
    const int targets = checked_blas_dim(
        missingness_targets.shape(1), "missingness-target columns"
    );
    if (rows <= 0 || variants <= 0 || target_rows != rows || targets < 0) {
        throw std::runtime_error(
            "Genotype standardization inputs have incompatible dimensions"
        );
    }
    if (ddof < 0 || ddof >= rows) {
        throw std::runtime_error(
            "Genotype standardization ddof must be in [0, sample count)"
        );
    }
    if (!std::isfinite(eps) || eps <= 0.0) {
        throw std::runtime_error(
            "Genotype standardization epsilon must be finite and positive"
        );
    }

    int64_t* missing_counts = nullptr;
    auto missing_counts_out = make_owned_numpy_vec1<int64_t>(
        static_cast<size_t>(variants), &missing_counts
    );
    double* correlations = nullptr;
    auto correlations_out = make_owned_numpy_mat2f<double>(
        static_cast<size_t>(targets), static_cast<size_t>(variants),
        &correlations
    );
    std::vector<double> target_norms(static_cast<size_t>(targets), 0.0);
    int invalid_targets = 0;
    for (int target = 0; target < targets; ++target) {
        const double* target_column = missingness_targets.data()
            + static_cast<size_t>(target) * static_cast<size_t>(rows);
        double squared_norm = 0.0;
        for (int row = 0; row < rows; ++row) {
            const double value = target_column[row];
            invalid_targets |= !std::isfinite(value);
            squared_norm += value * value;
        }
        target_norms[static_cast<size_t>(target)] = std::sqrt(squared_norm);
    }
    if (invalid_targets != 0) {
        throw std::runtime_error(
            "Missingness diagnostic targets contain NaN or infinity"
        );
    }

    const int threads = std::max(1, std::min(requested_threads, variants));
    {
        nb::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int variant = 0; variant < variants; ++variant) {
            double* genotype_column = genotype.data()
                + static_cast<size_t>(variant) * static_cast<size_t>(rows);
            double* correlation_column = correlations
                + static_cast<size_t>(variant) * static_cast<size_t>(targets);
            std::fill(
                correlation_column,
                correlation_column + targets,
                0.0
            );
            int64_t missing = 0;
            double sum = 0.0;
            for (int row = 0; row < rows; ++row) {
                const double value = genotype_column[row];
                if (std::isnan(value)) {
                    ++missing;
                    for (int target = 0; target < targets; ++target) {
                        correlation_column[target] +=
                            missingness_targets.data()[
                                static_cast<size_t>(target)
                                    * static_cast<size_t>(rows)
                                + static_cast<size_t>(row)
                            ];
                    }
                } else {
                    sum += value;
                }
            }
            missing_counts[variant] = missing;
            const int64_t observed = static_cast<int64_t>(rows) - missing;
            const double mean = observed > 0
                ? sum / static_cast<double>(observed) : 0.0;
            double centered_ss = 0.0;
            for (int row = 0; row < rows; ++row) {
                double value = genotype_column[row];
                value = std::isnan(value) ? 0.0 : value - mean;
                genotype_column[row] = value;
                centered_ss += value * value;
            }
            const double scale = hwe_scale
                ? std::sqrt(std::max(mean * (1.0 - 0.5 * mean), 0.0))
                : std::sqrt(centered_ss / static_cast<double>(rows - ddof));
            if (std::isfinite(scale) && scale > eps) {
                const double inverse_scale = 1.0 / scale;
                for (int row = 0; row < rows; ++row) {
                    genotype_column[row] *= inverse_scale;
                }
            } else {
                std::fill(genotype_column, genotype_column + rows, 0.0);
            }

            for (int target = 0; target < targets; ++target) {
                double correlation = 0.0;
                const double target_norm =
                    target_norms[static_cast<size_t>(target)];
                if (missing > 0 && missing < rows && target_norm > 0.0) {
                    const double missing_norm = std::sqrt(
                        static_cast<double>(missing)
                        * (1.0 - static_cast<double>(missing)
                            / static_cast<double>(rows))
                    );
                    correlation = correlation_column[target]
                        / (missing_norm * target_norm);
                }
                correlation_column[target] = correlation;
            }
        }
    }
    return nb::make_tuple(missing_counts_out, correlations_out);
}

nb::tuple standardize_genotype_block_row_major(
    nb_mat2c_rw<double> genotype,
    nb_mat2f_ro<double> missingness_targets,
    int ddof,
    bool hwe_scale,
    double eps,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        genotype.shape(0), "row-major genotype standardization rows"
    );
    const int variants = checked_blas_dim(
        genotype.shape(1), "row-major genotype standardization columns"
    );
    const int target_rows = checked_blas_dim(
        missingness_targets.shape(0), "row-major missingness-target rows"
    );
    const int targets = checked_blas_dim(
        missingness_targets.shape(1), "row-major missingness-target columns"
    );
    if (rows <= 0 || variants <= 0 || target_rows != rows || targets < 0) {
        throw std::runtime_error(
            "Row-major genotype standardization inputs have incompatible dimensions"
        );
    }
    if (ddof < 0 || ddof >= rows) {
        throw std::runtime_error(
            "Row-major genotype standardization ddof must be in [0, sample count)"
        );
    }
    if (!std::isfinite(eps) || eps <= 0.0) {
        throw std::runtime_error(
            "Row-major genotype standardization epsilon must be finite and positive"
        );
    }

    int64_t* missing_counts = nullptr;
    auto missing_counts_out = make_owned_numpy_vec1<int64_t>(
        static_cast<size_t>(variants), &missing_counts
    );
    double* correlations = nullptr;
    auto correlations_out = make_owned_numpy_mat2f<double>(
        static_cast<size_t>(targets), static_cast<size_t>(variants),
        &correlations
    );
    std::vector<double> target_norms(static_cast<size_t>(targets), 0.0);
    for (int target = 0; target < targets; ++target) {
        const double* target_column = missingness_targets.data()
            + static_cast<size_t>(target) * static_cast<size_t>(rows);
        double squared_norm = 0.0;
        for (int row = 0; row < rows; ++row) {
            const double value = target_column[row];
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Row-major missingness diagnostic targets contain NaN or infinity"
                );
            }
            squared_norm += value * value;
        }
        target_norms[static_cast<size_t>(target)] = std::sqrt(squared_norm);
    }

    constexpr int kVariantTile = 32;
    const int tiles = (variants + kVariantTile - 1) / kVariantTile;
    const int threads = std::max(1, std::min(requested_threads, tiles));
    {
        nb::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int tile = 0; tile < tiles; ++tile) {
            const int first = tile * kVariantTile;
            const int count = std::min(kVariantTile, variants - first);
            int64_t missing[kVariantTile] = {};
            double sums[kVariantTile] = {};
            double centered_ss[kVariantTile] = {};
            double means[kVariantTile] = {};
            double inverse_scales[kVariantTile] = {};
            std::vector<double> target_sums(
                checked_mul(
                    static_cast<size_t>(count),
                    static_cast<size_t>(targets),
                    "row-major missingness correlations"
                ),
                0.0
            );
            for (int row = 0; row < rows; ++row) {
                const double* values = genotype.data()
                    + static_cast<size_t>(row)
                        * static_cast<size_t>(variants)
                    + static_cast<size_t>(first);
                for (int offset = 0; offset < count; ++offset) {
                    const double value = values[offset];
                    if (std::isnan(value)) {
                        ++missing[offset];
                        for (int target = 0; target < targets; ++target) {
                            target_sums[
                                static_cast<size_t>(offset)
                                    * static_cast<size_t>(targets)
                                + static_cast<size_t>(target)
                            ] += missingness_targets.data()[
                                static_cast<size_t>(target)
                                    * static_cast<size_t>(rows)
                                + static_cast<size_t>(row)
                            ];
                        }
                    } else {
                        sums[offset] += value;
                    }
                }
            }
            for (int offset = 0; offset < count; ++offset) {
                const int64_t observed =
                    static_cast<int64_t>(rows) - missing[offset];
                means[offset] = observed > 0
                    ? sums[offset] / static_cast<double>(observed) : 0.0;
            }
            for (int row = 0; row < rows; ++row) {
                double* values = genotype.data()
                    + static_cast<size_t>(row)
                        * static_cast<size_t>(variants)
                    + static_cast<size_t>(first);
                for (int offset = 0; offset < count; ++offset) {
                    double value = values[offset];
                    value = std::isnan(value) ? 0.0 : value - means[offset];
                    values[offset] = value;
                    centered_ss[offset] += value * value;
                }
            }
            for (int offset = 0; offset < count; ++offset) {
                const double scale = hwe_scale
                    ? std::sqrt(std::max(
                        means[offset] * (1.0 - 0.5 * means[offset]), 0.0
                    ))
                    : std::sqrt(
                        centered_ss[offset]
                        / static_cast<double>(rows - ddof)
                    );
                inverse_scales[offset] =
                    std::isfinite(scale) && scale > eps ? 1.0 / scale : 0.0;
            }
            for (int row = 0; row < rows; ++row) {
                double* values = genotype.data()
                    + static_cast<size_t>(row)
                        * static_cast<size_t>(variants)
                    + static_cast<size_t>(first);
                for (int offset = 0; offset < count; ++offset) {
                    values[offset] = inverse_scales[offset] != 0.0
                        ? values[offset] * inverse_scales[offset] : 0.0;
                }
            }
            for (int offset = 0; offset < count; ++offset) {
                const int variant = first + offset;
                missing_counts[variant] = missing[offset];
                for (int target = 0; target < targets; ++target) {
                    double correlation = 0.0;
                    const double target_norm =
                        target_norms[static_cast<size_t>(target)];
                    if (missing[offset] > 0 && missing[offset] < rows
                        && target_norm > 0.0) {
                        const double missing_norm = std::sqrt(
                            static_cast<double>(missing[offset])
                            * (1.0 - static_cast<double>(missing[offset])
                                / static_cast<double>(rows))
                        );
                        correlation = target_sums[
                            static_cast<size_t>(offset)
                                * static_cast<size_t>(targets)
                            + static_cast<size_t>(target)
                        ] / (missing_norm * target_norm);
                    }
                    correlations[
                        static_cast<size_t>(variant)
                            * static_cast<size_t>(targets)
                        + static_cast<size_t>(target)
                    ] = correlation;
                }
            }
        }
    }
    return nb::make_tuple(missing_counts_out, correlations_out);
}

nb_numpy_mat2f<double> fused_feature_scalar_moments(
    nb_mat2f_ro<double> genotype,
    nb_mat2f_ro<double> environments,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        genotype.shape(0), "fused feature genotype rows"
    );
    const int variants = checked_blas_dim(
        genotype.shape(1), "fused feature genotype columns"
    );
    const int environment_rows = checked_blas_dim(
        environments.shape(0), "fused feature environment rows"
    );
    const int environment_count = checked_blas_dim(
        environments.shape(1), "fused feature environment columns"
    );
    if (rows <= 0 || variants <= 0 || environment_count <= 0 ||
        environment_rows != rows) {
        throw std::runtime_error(
            "Fused GxE feature moment inputs have incompatible dimensions"
        );
    }
    const size_t output_rows_size = checked_add(
        1U,
        checked_mul(3U, static_cast<size_t>(environment_count),
                    "fused feature moment rows"),
        "fused feature moment rows"
    );
    const int output_rows = checked_blas_dim(
        output_rows_size, "fused feature moment rows"
    );
    double* output = nullptr;
    auto result = make_owned_numpy_mat2f<double>(
        output_rows_size, static_cast<size_t>(variants), &output
    );
    int invalid = 0;
    const int threads = std::max(1, std::min(requested_threads, variants));
    {
        nb::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads) \
            reduction(|:invalid)
#endif
        for (int variant = 0; variant < variants; ++variant) {
            const double* genotype_column = genotype.data()
                + static_cast<size_t>(variant) * static_cast<size_t>(rows);
            double* moment_column = output
                + static_cast<size_t>(variant) * output_rows_size;
            std::fill(moment_column, moment_column + output_rows, 0.0);
            double s0 = 0.0;
            for (int row = 0; row < rows; ++row) {
                const double value = genotype_column[row];
                const double square = value * value;
                invalid |= !std::isfinite(value) || !std::isfinite(square);
                s0 += square;
                for (int environment = 0; environment < environment_count;
                     ++environment) {
                    const double e = environments.data()[
                        static_cast<size_t>(environment)
                            * static_cast<size_t>(rows)
                        + static_cast<size_t>(row)
                    ];
                    const double e2 = e * e;
                    const size_t offset = 1U
                        + 3U * static_cast<size_t>(environment);
                    moment_column[offset] += e * square;
                    moment_column[offset + 1U] += e2 * square;
                    moment_column[offset + 2U] += e2 * e2 * square;
                    invalid |= !std::isfinite(e) || !std::isfinite(e2);
                }
            }
            moment_column[0] = s0;
        }
    }
    if (invalid != 0) {
        throw std::runtime_error(
            "Fused GxE feature moment inputs contain NaN or infinity"
        );
    }
    return result;
}

nb_numpy_mat2f<double> make_native_gemm_output_mat2f(
    size_t rows,
    size_t columns,
    NativeGemmOutputTelemetryScope& scope,
    NativeGemmOutputAllocation** allocation_out,
    double** output
) {
    const size_t elements = checked_mul(
        rows, columns, "native column-major GEMM output elements"
    );
    const size_t logical_byte_count = checked_mul(
        elements, sizeof(double), "native column-major GEMM output bytes"
    );
    auto allocation = std::make_unique<NativeGemmOutputAllocation>(
        rows,
        columns,
        "column_major",
        logical_byte_count,
        scope.request(),
        scope.evidence()
    );
    NativeGemmOutputAllocation* control = allocation.get();
    nb::capsule owner(control, [](void* pointer) noexcept {
        delete static_cast<NativeGemmOutputAllocation*>(pointer);
    });
    allocation.release();
    *allocation_out = control;
    *output = control->data();
    scope.register_output(
        control->data(), logical_byte_count, CblasColMajor
    );
    return nb_numpy_mat2f<double>(
        control->data(), {rows, columns}, owner
    );
}

nb_numpy_mat2c<double> make_native_gemm_output_mat2c(
    size_t rows,
    size_t columns,
    NativeGemmOutputTelemetryScope& scope,
    NativeGemmOutputAllocation** allocation_out,
    double** output,
    CBLAS_LAYOUT vendor_output_layout = CblasRowMajor
) {
    const size_t elements = checked_mul(
        rows, columns, "native row-major GEMM output elements"
    );
    const size_t logical_byte_count = checked_mul(
        elements, sizeof(double), "native row-major GEMM output bytes"
    );
    auto allocation = std::make_unique<NativeGemmOutputAllocation>(
        rows,
        columns,
        "row_major",
        logical_byte_count,
        scope.request(),
        scope.evidence()
    );
    NativeGemmOutputAllocation* control = allocation.get();
    nb::capsule owner(control, [](void* pointer) noexcept {
        delete static_cast<NativeGemmOutputAllocation*>(pointer);
    });
    allocation.release();
    *allocation_out = control;
    *output = control->data();
    scope.register_output(
        control->data(), logical_byte_count, vendor_output_layout
    );
    return nb_numpy_mat2c<double>(
        control->data(), {rows, columns}, owner
    );
}

// One maximum-capacity allocation per scratch role.  A positive
// ``capacity_elements`` freezes the role's physical capacity (any larger
// logical request fails closed); zero grows the single allocation to the
// largest logical shape seen so far without ever retaining two mappings.
double* prepare_reusable_native_gemm_output_mat2f(
    std::unique_ptr<NativeGemmOutputAllocation>& allocation,
    size_t rows,
    size_t columns,
    NativeGemmOutputTelemetryScope& scope,
    bool& allocated,
    size_t capacity_elements = 0
) {
    const size_t elements = checked_mul(
        rows, columns, "reusable native column-major GEMM output elements"
    );
    const size_t logical_byte_count = checked_mul(
        elements, sizeof(double),
        "reusable native column-major GEMM output bytes"
    );
    if (capacity_elements != 0 && elements > capacity_elements) {
        throw std::runtime_error(
            "Reusable native GEMM output request exceeds its admitted capacity"
        );
    }
    if (allocation != nullptr && capacity_elements == 0
        && logical_byte_count > allocation->capacity_byte_count()) {
        allocation.reset();
    }
    allocated = allocation == nullptr;
    if (allocated) {
        allocation = std::make_unique<NativeGemmOutputAllocation>(
            rows,
            columns,
            "column_major",
            logical_byte_count,
            scope.request(),
            scope.evidence(),
            capacity_elements == 0
                ? logical_byte_count
                : checked_mul(
                    capacity_elements, sizeof(double),
                    "reusable native column-major GEMM output capacity"
                )
        );
    } else {
        allocation->reuse_for_call(
            rows,
            columns,
            "column_major",
            logical_byte_count,
            scope.request(),
            scope.evidence()
        );
    }
    scope.register_output(
        allocation->data(), logical_byte_count, CblasColMajor
    );
    return allocation->data();
}

nb::tuple protected_matmul_tt_row_major_output(
    nb_mat2f_ro<double> weights,
    nb_mat2f_ro<double> genotype,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int k = checked_blas_dim(
        weights.shape(0), "protected TT source reduction"
    );
    const int m = checked_blas_dim(
        weights.shape(1), "protected TT source panel columns"
    );
    const int n = checked_blas_dim(
        genotype.shape(0), "protected TT source sample rows"
    );
    if (k <= 0 || m <= 0 || n <= 0
        || checked_blas_dim(
            genotype.shape(1), "protected TT source genotype columns"
        ) != k) {
        throw std::runtime_error(
            "Protected GxE TT source operands have incompatible dimensions"
        );
    }
    // Row-major N-by-P and column-major P-by-N have identical storage.  Return
    // the former directly so Python can retain the winning TT output without
    // a transpose or layout-conversion allocation.
    NativeGemmOutputTelemetryScope output_scope;
    NativeGemmOutputAllocation* output_allocation = nullptr;
    double* output = nullptr;
    auto result = make_native_gemm_output_mat2c(
        static_cast<size_t>(n), static_cast<size_t>(m), output_scope,
        &output_allocation, &output, CblasColMajor
    );
    int64_t repaired = 0;
    {
        nb::gil_scoped_release release;
        repaired = dgemm_tt_partitioned(
            m, n, k,
            weights.data(), k,
            genotype.data(), n,
            output, m,
            requested_threads
        );
        output_allocation->verify_after_repair();
    }
    output_scope.complete();
    return nb::make_tuple(result, repaired);
}

void project_protected_row_major_sources(
    nb_mat2c_rw<double> panel,
    nb_mat2f_ro<double> common_basis,
    nb_mat2f_ro<double> directions,
    int columns_per_environment,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        panel.shape(0), "row-major source projection rows"
    );
    const int columns = checked_blas_dim(
        panel.shape(1), "row-major source projection columns"
    );
    const int common_rows = checked_blas_dim(
        common_basis.shape(0), "row-major source common-basis rows"
    );
    const int common_rank = checked_blas_dim(
        common_basis.shape(1), "row-major source common-basis rank"
    );
    const int direction_rows = checked_blas_dim(
        directions.shape(0), "row-major source direction rows"
    );
    const int groups = checked_blas_dim(
        directions.shape(1), "row-major source direction groups"
    );
    if (rows <= 0 || columns <= 0 || groups <= 0
        || columns_per_environment <= 0
        || common_rows != rows || direction_rows != rows
        || columns != 2 * columns_per_environment * groups) {
        throw std::runtime_error(
            "Row-major protected source projection inputs have incompatible dimensions"
        );
    }
    {
        nb::gil_scoped_release release;
#if defined(GWLDCORE_GEMM_INTEGRITY)
        const MatrixFingerprint common_before = fingerprint_col_major_matrix(
            rows, common_rank, common_basis.data(), rows, requested_threads
        );
        const MatrixFingerprint directions_before = fingerprint_col_major_matrix(
            rows, groups, directions.data(), rows, requested_threads
        );
#endif
        std::vector<double> common_coefficients(
            checked_mul(
                static_cast<size_t>(common_rank),
                static_cast<size_t>(columns),
                "row-major common projection coefficients"
            )
        );
        if (common_rank > 0) {
            // A Fortran N-by-r basis has the same storage as a row-major
            // r-by-N transpose.  Stream the C-order panel by row without a
            // panel-sized layout conversion.
            dgemm_row_nn_tiled(
                common_rank, columns, rows,
                common_basis.data(), rows,
                panel.data(), columns,
                common_coefficients.data(), columns,
                requested_threads
            );
        }
        const int row_threads = std::max(
            1, std::min(requested_threads, rows)
        );
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(row_threads)
#endif
        for (int row = 0; row < rows; ++row) {
            double* panel_row = panel.data()
                + static_cast<size_t>(row) * static_cast<size_t>(columns);
            for (int column = 0; column < columns; ++column) {
                double correction = 0.0;
                for (int basis = 0; basis < common_rank; ++basis) {
                    correction += common_basis.data()[
                        static_cast<size_t>(basis) * static_cast<size_t>(rows)
                        + static_cast<size_t>(row)
                    ] * common_coefficients[
                        static_cast<size_t>(basis)
                            * static_cast<size_t>(columns)
                        + static_cast<size_t>(column)
                    ];
                }
                panel_row[column] -= correction;
            }
        }

        std::vector<double> direction_coefficients(
            static_cast<size_t>(columns), 0.0
        );
        const int group_columns = 2 * columns_per_environment;
        for (int group = 0; group < groups; ++group) {
            const int first = group * group_columns;
            dgemm_row_nn_tiled(
                1, group_columns, rows,
                directions.data()
                    + static_cast<size_t>(group) * static_cast<size_t>(rows),
                rows,
                panel.data() + static_cast<size_t>(first), columns,
                direction_coefficients.data() + static_cast<size_t>(first),
                group_columns,
                requested_threads
            );
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(row_threads)
#endif
        for (int row = 0; row < rows; ++row) {
            double* panel_row = panel.data()
                + static_cast<size_t>(row) * static_cast<size_t>(columns);
            for (int group = 0; group < groups; ++group) {
                const int first = group * group_columns;
                const int last = first + group_columns;
                const double direction = directions.data()[
                    static_cast<size_t>(group) * static_cast<size_t>(rows)
                    + static_cast<size_t>(row)
                ];
                for (int column = first; column < last; ++column) {
                    panel_row[column] -= direction
                        * direction_coefficients[static_cast<size_t>(column)];
                }
            }
        }

        std::vector<double> ones(static_cast<size_t>(rows), 1.0);
        std::vector<double> column_sums(static_cast<size_t>(columns), 0.0);
        dgemm_row_nn_tiled(
            1, columns, rows,
            ones.data(), rows,
            panel.data(), columns,
            column_sums.data(), columns,
            requested_threads
        );
        int invalid = 0;
        const double inverse_rows = 1.0 / static_cast<double>(rows);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(row_threads) \
            reduction(|:invalid)
#endif
        for (int row = 0; row < rows; ++row) {
            double* panel_row = panel.data()
                + static_cast<size_t>(row) * static_cast<size_t>(columns);
            for (int column = 0; column < columns; ++column) {
                panel_row[column] -= column_sums[static_cast<size_t>(column)]
                    * inverse_rows;
                invalid |= !std::isfinite(panel_row[column]);
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Row-major protected source projection contains NaN or infinity"
            );
        }
#if defined(GWLDCORE_GEMM_INTEGRITY)
        if (!(common_before == fingerprint_col_major_matrix(
                  rows, common_rank, common_basis.data(), rows,
                  requested_threads))
            || !(directions_before == fingerprint_col_major_matrix(
                  rows, groups, directions.data(), rows,
                  requested_threads))) {
            throw std::runtime_error(
                "Protected GxE projection basis changed during row-major projection"
            );
        }
#endif
    }
}

ProtectedRowMajorPair prepare_protected_row_major_weighted_pair(
    nb_mat2c_ro<double> right,
    nb_mat2f_ro<double> row_weights,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        right.shape(0), "protected row-major pair rows"
    );
    const int columns = checked_blas_dim(
        right.shape(1), "protected row-major pair columns"
    );
    const int weight_rows = checked_blas_dim(
        row_weights.shape(0), "protected row-major pair weight rows"
    );
    const int groups = checked_blas_dim(
        row_weights.shape(1), "protected row-major pair groups"
    );
    if (rows <= 0 || columns <= 0 || groups <= 0 || weight_rows != rows
        || columns % groups != 0) {
        throw std::runtime_error(
            "Protected GxE row-major pair inputs have incompatible dimensions"
        );
    }
    const int columns_per_group = columns / groups;
    const int sealed_columns = checked_blas_dim(
        2U * static_cast<size_t>(columns),
        "protected row-major sealed columns"
    );
    const size_t pair_elements = checked_mul(
        static_cast<size_t>(rows), static_cast<size_t>(sealed_columns),
        "protected row-major pair allocation"
    );
    ReadOnlyDoubleMapping snapshot(pair_elements);
    MatrixFingerprint pair_fingerprint{};
    {
        nb::gil_scoped_release release;
#if defined(GWLDCORE_GEMM_INTEGRITY)
        const MatrixFingerprint right_before = fingerprint_row_major_matrix(
            rows, columns, right.data(), columns, requested_threads
        );
        const MatrixFingerprint weights_before = fingerprint_col_major_matrix(
            rows, groups, row_weights.data(), rows, requested_threads
        );
#endif
        int invalid = 0;
        const int threads = std::max(1, std::min(requested_threads, rows));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads) \
            reduction(|:invalid)
#endif
        for (int row = 0; row < rows; ++row) {
            const double* source = right.data()
                + static_cast<size_t>(row) * static_cast<size_t>(columns);
            double* destination = snapshot.data()
                + static_cast<size_t>(row)
                    * static_cast<size_t>(sealed_columns);
            for (int column = 0; column < columns; ++column) {
                const int group = column / columns_per_group;
                const double value = source[column];
                const double weight = row_weights.data()[
                    static_cast<size_t>(group) * static_cast<size_t>(rows)
                    + static_cast<size_t>(row)
                ];
                destination[column] = value;
                destination[columns + column] = value * weight;
                invalid |= !std::isfinite(value)
                    || !std::isfinite(weight)
                    || !std::isfinite(destination[columns + column]);
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Protected GxE row-major pair contains NaN or infinity"
            );
        }
#if defined(GWLDCORE_GEMM_INTEGRITY)
        if (!(right_before == fingerprint_row_major_matrix(
                  rows, columns, right.data(), columns, requested_threads))
            || !(weights_before == fingerprint_col_major_matrix(
                  rows, groups, row_weights.data(), rows, requested_threads))) {
            throw std::runtime_error(
                "Protected GxE row-major pair input changed while it was sealed"
            );
        }
        pair_fingerprint = fingerprint_row_major_matrix(
            rows, sealed_columns, snapshot.data(), sealed_columns,
            requested_threads
        );
#endif
        snapshot.seal_read_only();
    }
    return ProtectedRowMajorPair(
        rows, columns, pair_elements, pair_fingerprint, std::move(snapshot)
    );
}

nb::tuple protected_matmul_row_major_tn_pair(
    nb_mat2c_ro<double> genotype,
    const ProtectedRowMajorPair& right_pair,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int k = checked_blas_dim(
        genotype.shape(0), "protected row-major TN-pair reduction"
    );
    const int m = checked_blas_dim(
        genotype.shape(1), "protected row-major TN-pair rows"
    );
    const int n = checked_blas_dim(
        2U * static_cast<size_t>(right_pair.columns()),
        "protected row-major TN-pair columns"
    );
    const size_t expected_elements = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected row-major TN-pair elements"
    );
    if (k <= 0 || m <= 0 || right_pair.rows() != k
        || right_pair.data() == nullptr
        || right_pair.elements() != expected_elements) {
        throw std::runtime_error(
            "Protected GxE row-major TN-pair operands have incompatible dimensions"
        );
    }
    NativeGemmOutputTelemetryScope output_scope;
    NativeGemmOutputAllocation* output_allocation = nullptr;
    double* output = nullptr;
    auto result = make_native_gemm_output_mat2c(
        static_cast<size_t>(m), static_cast<size_t>(n), output_scope,
        &output_allocation, &output
    );
    int64_t repaired = 0;
    {
        nb::gil_scoped_release release;
        repaired = dgemm_row_tn_partitioned(
            m, n, k,
            genotype.data(), m,
            right_pair.data(), n,
            output, n,
            requested_threads,
            right_pair.fingerprint()
        );
        output_allocation->verify_after_repair();
    }
    output_scope.complete();
    return nb::make_tuple(result, repaired);
}

ProtectedRightPair prepare_protected_row_weighted_pair(
    nb_mat2f_ro<double> right,
    nb_mat2f_ro<double> row_weights,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        right.shape(0), "protected right-pair rows"
    );
    const int columns = checked_blas_dim(
        right.shape(1), "protected right-pair columns"
    );
    const int weight_rows = checked_blas_dim(
        row_weights.shape(0), "protected right-pair weight rows"
    );
    const int groups = checked_blas_dim(
        row_weights.shape(1), "protected right-pair weight groups"
    );
    if (rows <= 0 || columns <= 0 || groups <= 0 || weight_rows != rows ||
        columns % groups != 0) {
        throw std::runtime_error(
            "Protected GxE right-pair inputs have incompatible dimensions"
        );
    }
    const int columns_per_group = columns / groups;
    const size_t panel_elements = checked_mul(
        static_cast<size_t>(rows), static_cast<size_t>(columns),
        "protected right-pair panel"
    );
    const size_t pair_elements = checked_mul(
        2U, panel_elements, "protected right-pair allocation"
    );
    ReadOnlyDoubleMapping snapshot(pair_elements);
    MatrixFingerprint pair_fingerprint{};
    {
        nb::gil_scoped_release release;
#if defined(GWLDCORE_GEMM_INTEGRITY)
        const MatrixFingerprint right_before = fingerprint_col_major_matrix(
            rows, columns, right.data(), rows, requested_threads
        );
        const MatrixFingerprint weights_before = fingerprint_col_major_matrix(
            rows, groups, row_weights.data(), rows, requested_threads
        );
#endif
        int invalid = 0;
        const int threads = std::max(
            1, std::min(requested_threads, columns)
        );
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads) \
            reduction(|:invalid)
#endif
        for (int column = 0; column < columns; ++column) {
            const int group = column / columns_per_group;
            const double* source = right.data()
                + static_cast<size_t>(column) * static_cast<size_t>(rows);
            const double* weights = row_weights.data()
                + static_cast<size_t>(group) * static_cast<size_t>(rows);
            double* copied = snapshot.data()
                + static_cast<size_t>(column) * static_cast<size_t>(rows);
            double* weighted = snapshot.data() + panel_elements
                + static_cast<size_t>(column) * static_cast<size_t>(rows);
            for (int row = 0; row < rows; ++row) {
                const double value = source[row];
                const double weight = weights[row];
                copied[row] = value;
                weighted[row] = value * weight;
                invalid |= !std::isfinite(value)
                    || !std::isfinite(weight)
                    || !std::isfinite(weighted[row]);
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Protected GxE right-pair inputs contain NaN or infinity"
            );
        }
#if defined(GWLDCORE_GEMM_INTEGRITY)
        if (!(right_before == fingerprint_col_major_matrix(
                  rows, columns, right.data(), rows, requested_threads)) ||
            !(weights_before == fingerprint_col_major_matrix(
                  rows, groups, row_weights.data(), rows, requested_threads))) {
            throw std::runtime_error(
                "Protected GxE right-pair input changed while it was sealed"
            );
        }
        pair_fingerprint = fingerprint_col_major_matrix(
            rows,
            checked_blas_dim(
                2U * static_cast<size_t>(columns),
                "protected right-pair sealed columns"
            ),
            snapshot.data(), rows, requested_threads
        );
#endif
        snapshot.seal_read_only();
    }
    return ProtectedRightPair(
        rows, columns, pair_elements, pair_fingerprint, std::move(snapshot)
    );
}

nb::tuple protected_matmul_tn_pair(
    nb_mat2f_ro<double> left,
    const ProtectedRightPair& right_pair,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int k = checked_blas_dim(left.shape(0), "protected TN-pair reduction");
    const int m = checked_blas_dim(left.shape(1), "protected TN-pair rows");
    const int n = checked_blas_dim(
        2U * static_cast<size_t>(right_pair.columns()),
        "protected TN-pair columns"
    );
    const size_t expected_elements = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected TN-pair elements"
    );
    if (k <= 0 || m <= 0 || right_pair.rows() != k ||
        right_pair.data() == nullptr ||
        right_pair.elements() != expected_elements) {
        throw std::runtime_error(
            "Protected GxE TN-pair operands have incompatible dimensions"
        );
    }
    NativeGemmOutputTelemetryScope output_scope;
    NativeGemmOutputAllocation* output_allocation = nullptr;
    double* output = nullptr;
    auto result = make_native_gemm_output_mat2f(
        static_cast<size_t>(m), static_cast<size_t>(n), output_scope,
        &output_allocation, &output
    );
    int64_t repaired = 0;
    {
        nb::gil_scoped_release release;
        repaired = dgemm_tn_partitioned_rows(
            m, n, k,
            left.data(), k,
            right_pair.data(), k,
            output, m,
            requested_threads,
            1.0, 0.0, nullptr, true
        );
        output_allocation->verify_after_repair();
    }
    output_scope.complete();
    return nb::make_tuple(result, repaired);
}

nb::tuple protected_matmul_nn(
    nb_mat2f_ro<double> left,
    nb_mat2f_ro<double> right,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    if (left.shape(1) != right.shape(0)) {
        throw std::runtime_error(
            "Protected GxE NN GEMM operands have incompatible dimensions"
        );
    }
    const int m = checked_blas_dim(left.shape(0), "protected NN rows");
    const int k = checked_blas_dim(left.shape(1), "protected NN reduction");
    const int n = checked_blas_dim(right.shape(1), "protected NN columns");
    if (m == 0 || n == 0 || k == 0) {
        throw std::runtime_error("Protected GxE NN GEMM operands must be non-empty");
    }
    NativeGemmOutputTelemetryScope output_scope;
    NativeGemmOutputAllocation* output_allocation = nullptr;
    double* output = nullptr;
    auto result = make_native_gemm_output_mat2f(
        static_cast<size_t>(m), static_cast<size_t>(n), output_scope,
        &output_allocation, &output
    );
    int64_t repaired = 0;
    {
        nb::gil_scoped_release release;
        repaired = dgemm_nn_partitioned_rows(
            m, n, k,
            left.data(), m,
            right.data(), k,
            output, m,
            requested_threads
        );
        output_allocation->verify_after_repair();
    }
    output_scope.complete();
    return nb::make_tuple(result, repaired);
}

// NumPy's Philox bit generator is used by the established Python GxE path.
// The direct context receives only the two uint64 key words produced by
// NumPy's SeedSequence during setup, then reproduces Philox4x64-10 and the
// uint8 bounded-integer stream natively.  This keeps the scientific random
// stream unchanged without materializing an M-by-B probe matrix in Python.
struct NumpyPhiloxState {
    std::array<uint64_t, 4> counter{};
    std::array<uint64_t, 2> key{};
    std::array<uint64_t, 4> buffer{};
    int buffer_position = 4;
    bool has_uint32 = false;
    uint32_t buffered_uint32 = 0;
};

std::array<uint64_t, 4> numpy_philox4x64_round(
    const std::array<uint64_t, 4>& counter,
    const std::array<uint64_t, 2>& key
) {
    constexpr uint64_t multiplier0 = 0xD2E7470EE14C6C93ULL;
    constexpr uint64_t multiplier1 = 0xCA5A826395121157ULL;
    const __uint128_t product0 = static_cast<__uint128_t>(multiplier0)
        * static_cast<__uint128_t>(counter[0]);
    const __uint128_t product1 = static_cast<__uint128_t>(multiplier1)
        * static_cast<__uint128_t>(counter[2]);
    const uint64_t lo0 = static_cast<uint64_t>(product0);
    const uint64_t hi0 = static_cast<uint64_t>(product0 >> 64U);
    const uint64_t lo1 = static_cast<uint64_t>(product1);
    const uint64_t hi1 = static_cast<uint64_t>(product1 >> 64U);
    return {
        hi1 ^ counter[1] ^ key[0], lo1,
        hi0 ^ counter[3] ^ key[1], lo0,
    };
}

std::array<uint64_t, 4> numpy_philox4x64_generate(
    std::array<uint64_t, 4> counter,
    std::array<uint64_t, 2> key
) {
    constexpr uint64_t bump0 = 0x9E3779B97F4A7C15ULL;
    constexpr uint64_t bump1 = 0xBB67AE8584CAA73BULL;
    for (int round = 0; round < 10; ++round) {
        counter = numpy_philox4x64_round(counter, key);
        if (round != 9) {
            key[0] += bump0;
            key[1] += bump1;
        }
    }
    return counter;
}

uint64_t numpy_philox_next64(NumpyPhiloxState& state) {
    if (state.buffer_position < 4) {
        return state.buffer[static_cast<size_t>(state.buffer_position++)];
    }
    for (size_t word = 0; word < state.counter.size(); ++word) {
        if (++state.counter[word] != 0) break;
    }
    state.buffer = numpy_philox4x64_generate(state.counter, state.key);
    state.buffer_position = 1;
    return state.buffer[0];
}

uint32_t numpy_philox_next32(NumpyPhiloxState& state) {
    if (state.has_uint32) {
        state.has_uint32 = false;
        return state.buffered_uint32;
    }
    const uint64_t value = numpy_philox_next64(state);
    state.has_uint32 = true;
    state.buffered_uint32 = static_cast<uint32_t>(value >> 32U);
    return static_cast<uint32_t>(value);
}

void fill_numpy_philox_rademacher(
    double* output,
    int rows,
    const uint64_t* keys,
    int probes,
    int requested_threads
) {
    if (output == nullptr || keys == nullptr || rows <= 0 || probes <= 0) {
        throw std::runtime_error("Native Philox Rademacher inputs are invalid");
    }
    int invalid = 0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(requested_threads) \
        reduction(|:invalid)
#endif
    for (int probe = 0; probe < probes; ++probe) {
        NumpyPhiloxState state;
        state.key = {
            keys[static_cast<size_t>(probe) * 2U],
            keys[static_cast<size_t>(probe) * 2U + 1U],
        };
        double* column = output
            + static_cast<size_t>(probe) * static_cast<size_t>(rows);
        uint32_t byte_buffer = 0;
        int buffered_bytes = 0;
        for (int row = 0; row < rows; ++row) {
            if (buffered_bytes == 0) {
                byte_buffer = numpy_philox_next32(state);
                buffered_bytes = 4;
            }
            const uint8_t byte = static_cast<uint8_t>(byte_buffer);
            byte_buffer >>= 8U;
            --buffered_bytes;
            // NumPy's uint8 bounded-Lemire path for [0, 2) returns the high
            // bit of each buffered byte, in little-endian byte order.
            const double value = (byte & 0x80U) != 0 ? 1.0 : -1.0;
            column[row] = value;
            invalid |= !std::isfinite(value);
        }
    }
    if (invalid != 0) {
        throw std::runtime_error("Native Philox Rademacher generation failed");
    }
}

nb_numpy_mat2f<double> numpy_philox_rademacher_block(
    nb_mat2c_ro<uint64_t> keys,
    int rows,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int probes = checked_blas_dim(
        keys.shape(0), "native Philox probe count"
    );
    if (rows <= 0 || probes <= 0 || keys.shape(1) != 2) {
        throw std::runtime_error("Native Philox key matrix must be probes-by-two");
    }
    double* output = nullptr;
    auto result = make_owned_numpy_mat2f<double>(
        static_cast<size_t>(rows), static_cast<size_t>(probes), &output
    );
    {
        nb::gil_scoped_release release;
        fill_numpy_philox_rademacher(
            output, rows, keys.data(), probes, requested_threads
        );
    }
    return result;
}

uint64_t generalized_probe_splitmix64(uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

double generalized_global_probe_sign(
    uint64_t root_seed,
    uint64_t global_variant_index,
    uint64_t global_probe_index,
    uint64_t namespace_key
) noexcept {
    constexpr uint64_t mix_variant = 0xD2B74407B1CE6E93ULL;
    constexpr uint64_t mix_probe = 0xCA5A826395121157ULL;
    constexpr uint64_t mix_root = 0x9E3779B97F4A7C15ULL;
    uint64_t state = generalized_probe_splitmix64(
        root_seed ^ namespace_key ^ mix_root
    );
    state ^= generalized_probe_splitmix64(global_variant_index + mix_variant);
    state ^= generalized_probe_splitmix64(global_probe_index + mix_probe);
    return (generalized_probe_splitmix64(state) >> 63U) != 0U ? 1.0 : -1.0;
}

nb_numpy_mat2f<double> global_variant_rademacher(
    nb_vec1_ro<int64_t> variant_indices,
    nb_vec1_ro<int64_t> probe_indices,
    uint64_t root_seed,
    uint64_t namespace_key,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int rows = checked_blas_dim(
        variant_indices.shape(0), "global variant-probe row count"
    );
    const int probes = checked_blas_dim(
        probe_indices.shape(0), "global variant-probe column count"
    );
    if (rows <= 0 || probes <= 0) {
        throw std::runtime_error(
            "Global variant probes require nonempty variant and probe axes"
        );
    }
    for (int row = 0; row < rows; ++row) {
        if (variant_indices(static_cast<size_t>(row)) < 0) {
            throw std::runtime_error(
                "Global variant indices must be nonnegative"
            );
        }
    }
    for (int probe = 0; probe < probes; ++probe) {
        if (probe_indices(static_cast<size_t>(probe)) < 0) {
            throw std::runtime_error(
                "Global probe indices must be nonnegative"
            );
        }
    }
    double* output = nullptr;
    auto result = make_owned_numpy_mat2f<double>(
        static_cast<size_t>(rows), static_cast<size_t>(probes), &output
    );
    {
        nb::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(requested_threads)
#endif
        for (int probe = 0; probe < probes; ++probe) {
            double* column = output
                + static_cast<size_t>(probe) * static_cast<size_t>(rows);
            const auto global_probe = static_cast<uint64_t>(
                probe_indices(static_cast<size_t>(probe))
            );
            for (int row = 0; row < rows; ++row) {
                column[row] = generalized_global_probe_sign(
                    root_seed,
                    static_cast<uint64_t>(
                        variant_indices(static_cast<size_t>(row))
                    ),
                    global_probe,
                    namespace_key
                );
            }
        }
    }
    return result;
}

class MultiEnvironmentKernel {
public:
    MultiEnvironmentKernel(
        nb_mat2f_ro<double> environments,
        nb_mat2f_ro<double> feature_basis,
        nb_vec1_ro<int64_t> feature_power_indices,
        nb_vec1_ro<int64_t> feature_power_offsets,
        nb_vec1_ro<int32_t> feature_ranks,
        nb_vec1_ro<double> feature_gram_values,
        nb_vec1_ro<int64_t> feature_gram_offsets,
        nb_vec1_ro<double> feature_e2_gram_values,
        nb_mat2f_ro<double> common_basis,
        nb_mat2f_ro<double> directions,
        int ddof,
        int annotation_bins,
        int requested_threads
    )
        : rows_(checked_blas_dim(environments.shape(0),
                                "multi-environment rows")),
          environments_(checked_blas_dim(environments.shape(1),
                                         "multi-environment count")),
          feature_columns_(checked_blas_dim(feature_basis.shape(1),
                                            "multi-environment feature basis")),
          common_rank_(checked_blas_dim(common_basis.shape(1),
                                        "multi-environment common rank")),
          ddof_(ddof),
          annotation_bins_(annotation_bins),
          threads_(requested_threads) {
        validate_protected_gemm_threads(threads_);
        if (rows_ <= 2 || environments_ <= 0 || feature_columns_ <= 0 ||
            common_rank_ < 0 || ddof_ < 0 || ddof_ >= rows_ ||
            annotation_bins_ <= 0 ||
            checked_blas_dim(feature_basis.shape(0),
                             "multi-environment feature rows") != rows_ ||
            checked_blas_dim(common_basis.shape(0),
                             "multi-environment common rows") != rows_ ||
            checked_blas_dim(directions.shape(0),
                             "multi-environment direction rows") != rows_ ||
            checked_blas_dim(directions.shape(1),
                             "multi-environment direction count") != environments_) {
            throw std::runtime_error(
                "Multi-environment native kernel inputs have incompatible dimensions"
            );
        }
        if (feature_ranks.shape(0) != static_cast<size_t>(environments_) ||
            feature_power_offsets.shape(0) !=
                static_cast<size_t>(4 * environments_ + 1) ||
            feature_gram_offsets.shape(0) !=
                static_cast<size_t>(environments_ + 1) ||
            feature_gram_values.shape(0) != feature_e2_gram_values.shape(0)) {
            throw std::runtime_error(
                "Multi-environment native feature metadata has incompatible dimensions"
            );
        }

        environment_values_.assign(
            environments.data(),
            environments.data() + checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(environments_),
                "multi-environment values"
            )
        );
        feature_basis_.assign(
            feature_basis.data(),
            feature_basis.data() + checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(feature_columns_),
                "multi-environment feature basis"
            )
        );
        common_basis_.assign(
            common_basis.data(),
            common_basis.data() + checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(common_rank_),
                "multi-environment common basis"
            )
        );
        directions_.assign(
            directions.data(),
            directions.data() + checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(environments_),
                "multi-environment directions"
            )
        );
        power_indices_.assign(
            feature_power_indices.data(),
            feature_power_indices.data() + feature_power_indices.shape(0)
        );
        power_offsets_.assign(
            feature_power_offsets.data(),
            feature_power_offsets.data() + feature_power_offsets.shape(0)
        );
        ranks_.assign(
            feature_ranks.data(),
            feature_ranks.data() + feature_ranks.shape(0)
        );
        gram_values_.assign(
            feature_gram_values.data(),
            feature_gram_values.data() + feature_gram_values.shape(0)
        );
        gram_offsets_.assign(
            feature_gram_offsets.data(),
            feature_gram_offsets.data() + feature_gram_offsets.shape(0)
        );
        e2_gram_values_.assign(
            feature_e2_gram_values.data(),
            feature_e2_gram_values.data() + feature_e2_gram_values.shape(0)
        );
        validate_metadata();
        prepare_packed_feature_rhs();
        mailman_worker_arenas_.resize(static_cast<size_t>(threads_));
    }

    nb::dict info() const {
        std::lock_guard<std::mutex> guard(mutex_);
        nb::dict result;
        result["schema"] = "summit.multi_environment_native_kernel.v1";
        result["schema_version"] = 1;
        result["rows"] = rows_;
        result["environment_count"] = environments_;
        result["annotation_bins"] = annotation_bins_;
        result["feature_basis_columns"] = feature_columns_;
        result["common_basis_rank"] = common_rank_;
        result["ddof"] = ddof_;
        result["threads"] = threads_;
        result["feature_calls"] = feature_calls_.load(std::memory_order_relaxed);
        result["packed_feature_calls"] =
            packed_feature_calls_.load(std::memory_order_relaxed);
        result["source_calls"] = source_calls_.load(std::memory_order_relaxed);
        result["packed_source_calls"] =
            packed_source_calls_.load(std::memory_order_relaxed);
        result["projection_calls"] = projection_calls_.load(std::memory_order_relaxed);
        result["target_calls"] = target_calls_.load(std::memory_order_relaxed);
        result["packed_target_calls"] =
            packed_target_calls_.load(std::memory_order_relaxed);
        result["normalization_calls"] =
            normalization_calls_.load(std::memory_order_relaxed);
        result["repaired_gemm_output_columns"] =
            repaired_columns_.load(std::memory_order_relaxed);
        result["checksum_recomputed_gemm_output_columns"] =
            checksum_recomputed_columns_.load(std::memory_order_relaxed);
        result["roundoff_only_gemm_output_columns"] =
            roundoff_only_columns_.load(std::memory_order_relaxed);
        result["persistent_scratch_output_allocations"] =
            scratch_output_allocations_.load(std::memory_order_relaxed);
        result["persistent_scratch_output_reuses"] =
            scratch_output_reuses_.load(std::memory_order_relaxed);
        result["source_weight_scratch_capacity_bytes"] = checked_mul(
            source_weights_scratch_.capacity(), sizeof(double),
            "multi-environment source weight scratch capacity"
        );
        result["source_annotation_scratch_capacity_bytes"] = checked_mul(
            source_sqrt_annotation_scratch_.capacity(), sizeof(double),
            "multi-environment source annotation scratch capacity"
        );
        result["execution_scratch_released"] = execution_scratch_released_;
        nb::dict roles;
        roles["source_output"] = scratch_role_to_dict(source_output_role_);
        roles["target_output"] = scratch_role_to_dict(target_output_role_);
        roles["source_weights"] = scratch_role_to_dict(source_weights_role_);
        roles["source_annotation"] =
            scratch_role_to_dict(source_annotation_role_);
        roles["feature_projected"] =
            scratch_role_to_dict(feature_projected_role_);
        roles["feature_scalar"] = scratch_role_to_dict(feature_scalar_role_);
        roles["target_mailman_output"] =
            scratch_role_to_dict(target_mailman_role_);
        roles["mailman_worker"] = mailman_worker_role_dict();
        result["scratch_roles"] = std::move(roles);
        result["mailman_plan_frozen"] = mailman_plan_frozen_;
        result["mailman_frozen_segment_size"] = mailman_frozen_segment_size_;
        result["mailman_frozen_table_size"] = mailman_frozen_table_size_;
        result["mailman_qpanel_feature"] = mailman_qpanel_feature_;
        result["mailman_qpanel_source"] = mailman_qpanel_source_;
        result["mailman_qpanel_target"] = mailman_qpanel_target_;
        result["mailman_worker_count"] = threads_;
        result["mailman_worker_capacity_bytes"] =
            (mailman_worker_table_capacity_
             + mailman_worker_segment_a_capacity_
             + mailman_worker_segment_b_capacity_) * sizeof(double);
        return result;
    }

    // Freeze exact per-role scratch capacities from the direct context's
    // admitted execution plan.  Every later request larger than its frozen
    // capacity fails closed instead of growing the allocation.
    void configure_scratch_capacities(
        size_t source_output_elements,
        size_t target_output_elements,
        size_t source_weights_elements,
        size_t source_annotation_elements,
        size_t feature_projected_elements,
        size_t feature_scalar_elements,
        size_t target_mailman_elements
    ) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (scratch_capacities_frozen_) {
            throw std::runtime_error(
                "Multi-environment kernel scratch capacities are already frozen"
            );
        }
        source_output_capacity_elements_ = source_output_elements;
        target_output_capacity_elements_ = target_output_elements;
        source_weights_capacity_elements_ = source_weights_elements;
        source_annotation_capacity_elements_ = source_annotation_elements;
        feature_projected_capacity_elements_ = feature_projected_elements;
        feature_scalar_capacity_elements_ = feature_scalar_elements;
        target_mailman_capacity_elements_ = target_mailman_elements;
        scratch_capacities_frozen_ = true;
    }

    int packed_feature_rhs_columns() const {
        return packed_feature_rhs_columns_;
    }

    // Freeze the Mailman execution configuration (segment size, q-panel
    // widths, and exact per-worker capacities) from the direct context's
    // admitted plan and allocate every worker arena eagerly so the reported
    // capacity is the truth from the first packed call.
    void configure_mailman_plan(
        int segment_size,
        int64_t table_size,
        int qpanel_feature,
        int qpanel_source,
        int qpanel_target,
        size_t worker_table_capacity_elements,
        size_t worker_segment_a_capacity_elements,
        size_t worker_segment_b_capacity_elements
    ) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (mailman_plan_frozen_) {
            throw std::runtime_error(
                "Multi-environment Mailman plan is already frozen"
            );
        }
        if (segment_size <= 0 || table_size <= 0 || qpanel_feature <= 0
            || qpanel_source <= 0 || qpanel_target <= 0) {
            throw std::runtime_error(
                "Multi-environment Mailman plan configuration is invalid"
            );
        }
        mailman_frozen_segment_size_ = segment_size;
        mailman_frozen_table_size_ = table_size;
        mailman_qpanel_feature_ = qpanel_feature;
        mailman_qpanel_source_ = qpanel_source;
        mailman_qpanel_target_ = qpanel_target;
        mailman_worker_table_capacity_ = worker_table_capacity_elements;
        mailman_worker_segment_a_capacity_ =
            worker_segment_a_capacity_elements;
        mailman_worker_segment_b_capacity_ =
            worker_segment_b_capacity_elements;
        mailman_plan_frozen_ = true;
        for (MailmanWorkerArena& arena : mailman_worker_arenas_) {
            arena.work_table.assign(worker_table_capacity_elements, 0.0);
            arena.segment_a.resize(worker_segment_a_capacity_elements);
            arena.segment_b.resize(worker_segment_b_capacity_elements);
        }
    }

    // Release every kernel-owned execution scratch role.  Idempotent;
    // records the released capacity per role and rejects any later scratch
    // request so accidental use-after-release fails clearly.
    void release_execution_scratch() const {
        std::lock_guard<std::mutex> guard(mutex_);
        release_source_scratch_unlocked();
        if (target_output_scratch_ != nullptr) {
            target_output_role_.released_bytes = std::max(
                target_output_role_.released_bytes,
                target_output_scratch_->capacity_byte_count()
            );
        }
        target_output_scratch_.reset();
        auto release_vector = [](std::vector<double>& scratch,
                                 ScratchRoleTelemetry& role) {
            role.released_bytes = std::max(
                role.released_bytes, scratch.capacity() * sizeof(double)
            );
            std::vector<double>().swap(scratch);
        };
        release_vector(feature_projected_scratch_, feature_projected_role_);
        release_vector(feature_scalar_scratch_, feature_scalar_role_);
        release_vector(target_mailman_output_scratch_, target_mailman_role_);
        mailman_worker_role_.released_bytes = std::max(
            mailman_worker_role_.released_bytes, mailman_worker_arena_bytes()
        );
        for (MailmanWorkerArena& arena : mailman_worker_arenas_) {
            std::vector<double>().swap(arena.work_table);
            std::vector<double>().swap(arena.segment_a);
            std::vector<double>().swap(arena.segment_b);
            arena.table_dirty = false;
        }
        execution_scratch_released_ = true;
    }

    void release_source_scratch() const {
        std::lock_guard<std::mutex> guard(mutex_);
        release_source_scratch_unlocked();
    }

    size_t released_scratch_bytes() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return source_output_role_.released_bytes
            + target_output_role_.released_bytes
            + source_weights_role_.released_bytes
            + source_annotation_role_.released_bytes
            + feature_projected_role_.released_bytes
            + feature_scalar_role_.released_bytes
            + target_mailman_role_.released_bytes
            + mailman_worker_role_.released_bytes;
    }

    size_t live_scratch_capacity_bytes() const {
        std::lock_guard<std::mutex> guard(mutex_);
        size_t total = mailman_worker_arena_bytes()
            + (source_weights_scratch_.capacity()
               + source_sqrt_annotation_scratch_.capacity()
               + feature_projected_scratch_.capacity()
               + feature_scalar_scratch_.capacity()
               + target_mailman_output_scratch_.capacity()) * sizeof(double);
        if (source_output_scratch_ != nullptr) {
            total += source_output_scratch_->capacity_byte_count();
        }
        if (target_output_scratch_ != nullptr) {
            total += target_output_scratch_->capacity_byte_count();
        }
        return total;
    }

    void release_source_scratch_unlocked() const {
        source_weights_role_.released_bytes = std::max(
            source_weights_role_.released_bytes,
            source_weights_scratch_.capacity() * sizeof(double)
        );
        source_annotation_role_.released_bytes = std::max(
            source_annotation_role_.released_bytes,
            source_sqrt_annotation_scratch_.capacity() * sizeof(double)
        );
        if (source_output_scratch_ != nullptr) {
            source_output_role_.released_bytes = std::max(
                source_output_role_.released_bytes,
                source_output_scratch_->capacity_byte_count()
            );
        }
        source_output_scratch_.reset();
        std::vector<double>().swap(source_weights_scratch_);
        std::vector<double>().swap(source_sqrt_annotation_scratch_);
    }

    nb::dict feature_block(nb_mat2f_ro<double> genotype,
                           double eps_var,
                           bool standardized) const {
        std::lock_guard<std::mutex> guard(mutex_);
        validate_genotype(genotype, "multi-environment feature genotype");
        if (!(eps_var > 0.0) || !std::isfinite(eps_var)) {
            throw std::runtime_error(
                "Multi-environment native feature eps_var must be positive and finite"
            );
        }
        const int variants = checked_blas_dim(
            genotype.shape(1), "multi-environment feature variants"
        );
        double* scale_x = nullptr;
        double* scale_w = nullptr;
        double* norm_x = nullptr;
        double* norm_w = nullptr;
        double* diag_x = nullptr;
        double* diag_w = nullptr;
        double* corr_xw = nullptr;
        auto scale_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &scale_x
        );
        auto scale_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &scale_w
        );
        auto norm_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &norm_x
        );
        auto norm_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &norm_w
        );
        auto diag_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &diag_x
        );
        auto diag_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &diag_w
        );
        auto corr_xw_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &corr_xw
        );
        double* max_leak_x = nullptr;
        double* max_leak_w = nullptr;
        auto max_leak_x_out = make_owned_numpy_vec1<double>(
            static_cast<size_t>(environments_), &max_leak_x
        );
        auto max_leak_w_out = make_owned_numpy_vec1<double>(
            static_cast<size_t>(environments_), &max_leak_w
        );
        std::fill(max_leak_x, max_leak_x + environments_, 0.0);
        std::fill(max_leak_w, max_leak_w + environments_, 0.0);

        NativeGemmOutputTelemetryScope output_scope;
        NativeGemmOutputAllocation* output_allocation = nullptr;
        double* projected = nullptr;
        auto projected_owner = make_native_gemm_output_mat2f(
            static_cast<size_t>(feature_columns_),
            static_cast<size_t>(variants),
            output_scope, &output_allocation, &projected
        );
        GemmIntegrityResolution integrity_resolution;
        {
            nb::gil_scoped_release release;
            const int scalar_rows = 1 + 3 * environments_;
            std::vector<double> scalar(checked_mul(
                static_cast<size_t>(scalar_rows),
                static_cast<size_t>(variants),
                "multi-environment scalar moments"
            ), 0.0);
            int invalid_input = 0;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_input)
#endif
            for (int variant = 0; variant < variants; ++variant) {
                const double* column = genotype.data() +
                    static_cast<size_t>(variant) * static_cast<size_t>(rows_);
                double* output = scalar.data() +
                    static_cast<size_t>(variant) * static_cast<size_t>(scalar_rows);
                for (int row = 0; row < rows_; ++row) {
                    const double value = column[row];
                    const double square = value * value;
                    output[0] += square;
                    invalid_input |= !std::isfinite(value) || !std::isfinite(square);
                    for (int environment = 0; environment < environments_; ++environment) {
                        const double e = environment_values_[
                            static_cast<size_t>(environment) * static_cast<size_t>(rows_) +
                            static_cast<size_t>(row)
                        ];
                        const double e2 = e * e;
                        const size_t offset = 1U + 3U * static_cast<size_t>(environment);
                        output[offset] += e * square;
                        output[offset + 1U] += e2 * square;
                        output[offset + 2U] += e2 * e2 * square;
                    }
                }
            }
            if (invalid_input != 0) {
                throw std::runtime_error(
                    "Multi-environment native feature genotype contains NaN or infinity"
                );
            }
            (void)dgemm_tn_partitioned_columns(
                feature_columns_, variants, rows_,
                feature_basis_.data(), rows_, genotype.data(), rows_,
                projected, feature_columns_, threads_, &integrity_resolution
            );
            record_integrity_resolution(integrity_resolution);
            output_allocation->verify_after_repair();
            output_scope.complete();

            finalize_feature_moments(
                scalar.data(), projected, variants, eps_var, standardized,
                scale_x, scale_w, norm_x, norm_w, diag_x, diag_w, corr_xw,
                max_leak_x, max_leak_w
            );
        }
        (void)projected_owner;
        feature_calls_.fetch_add(1, std::memory_order_relaxed);
        nb::dict result;
        result["scale_x"] = std::move(scale_x_out);
        result["scale_w"] = std::move(scale_w_out);
        result["norm_x"] = std::move(norm_x_out);
        result["norm_w"] = std::move(norm_w_out);
        result["diag_nxe_x"] = std::move(diag_x_out);
        result["diag_nxe_w"] = std::move(diag_w_out);
        result["corr_xw"] = std::move(corr_xw_out);
        result["max_projection_leakage_additive"] = std::move(max_leak_x_out);
        result["max_projection_leakage_interaction"] = std::move(max_leak_w_out);
        result["repaired_gemm_output_columns"] =
            integrity_resolution.materially_repaired_columns;
        return result;
    }

    nb::dict feature_block_packed(
        const MailmanPackedBlock& packed,
        double eps_var,
        bool standardized
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        const int variants = packed.L;
        if (packed.N != rows_ || variants <= 0 || packed.segment_size <= 0
            || packed.n_segments <= 0 || packed.table_size <= 0
            || packed.mean.size() != static_cast<size_t>(variants)
            || packed.inv_std.size() != static_cast<size_t>(variants)
            || packed.missing_rows.size() != static_cast<size_t>(variants)
            || !(eps_var > 0.0) || !std::isfinite(eps_var)) {
            throw std::runtime_error(
                "Packed multi-environment feature inputs are invalid"
            );
        }

        double* scale_x = nullptr;
        double* scale_w = nullptr;
        double* norm_x = nullptr;
        double* norm_w = nullptr;
        double* diag_x = nullptr;
        double* diag_w = nullptr;
        double* corr_xw = nullptr;
        auto scale_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &scale_x
        );
        auto scale_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &scale_w
        );
        auto norm_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &norm_x
        );
        auto norm_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &norm_w
        );
        auto diag_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &diag_x
        );
        auto diag_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &diag_w
        );
        auto corr_xw_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants), static_cast<size_t>(environments_),
            &corr_xw
        );
        double* max_leak_x = nullptr;
        double* max_leak_w = nullptr;
        auto max_leak_x_out = make_owned_numpy_vec1<double>(
            static_cast<size_t>(environments_), &max_leak_x
        );
        auto max_leak_w_out = make_owned_numpy_vec1<double>(
            static_cast<size_t>(environments_), &max_leak_w
        );

        const int scalar_rows = 1 + 3 * environments_;
        const int rhs_columns = packed_feature_rhs_columns_;
        const size_t projected_elements = checked_mul(
            static_cast<size_t>(variants),
            static_cast<size_t>(feature_columns_),
            "packed feature projections"
        );
        double* projected = prepare_vector_scratch(
            feature_projected_scratch_, feature_projected_role_,
            projected_elements, feature_projected_capacity_elements_,
            "packed feature projection scratch"
        );
        const size_t scalar_elements = checked_mul(
            static_cast<size_t>(variants), static_cast<size_t>(scalar_rows),
            "packed feature scalar moments"
        );
        double* scalar = prepare_vector_scratch(
            feature_scalar_scratch_, feature_scalar_role_,
            scalar_elements, feature_scalar_capacity_elements_,
            "packed feature scalar scratch"
        );
        {
            nb::gil_scoped_release release;
            require_frozen_mailman_geometry(packed);
            const int qpanel = mailman_plan_frozen_
                ? mailman_qpanel_feature_
                : summit::mailman::qpanel_width<double>(
                    packed.table_size, rhs_columns, packed.segment_size, 2
                );
            // Process the fused feature RHS in bounded q-panels.  Column
            // sums are independent, so panelling reorders no within-column
            // arithmetic and the moments stay bitwise identical.
            for (int q0 = 0; q0 < rhs_columns; q0 += qpanel) {
                const int q = std::min(qpanel, rhs_columns - q0);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
                {
                    const size_t table_elements = checked_mul(
                        static_cast<size_t>(packed.table_size),
                        static_cast<size_t>(q),
                        "packed feature lookup table"
                    );
                    const size_t segment_elements = checked_mul(
                        static_cast<size_t>(packed.segment_size),
                        static_cast<size_t>(q),
                        "packed feature segment output"
                    );
                    MailmanWorkerArena& arena = mailman_worker_arena(
                        table_elements, segment_elements, segment_elements,
                        true
                    );
                    double* work_table = arena.work_table.data();
                    double* linear = arena.segment_a.data();
                    double* squared = arena.segment_b.data();
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int64_t segment = 0; segment < packed.n_segments;
                         ++segment) {
                        const int base = static_cast<int>(
                            segment * static_cast<int64_t>(packed.segment_size)
                        );
                        const int actual = std::min(
                            packed.segment_size, variants - base
                        );
                        if (packed.use_u16) {
                            summit::mailman::pre_multiply_rowmajor_with_squares(
                                packed.packed16.data()
                                    + static_cast<size_t>(segment)
                                        * static_cast<size_t>(rows_),
                                actual, rows_, q,
                                packed_feature_rhs_.data() + q0, rhs_columns,
                                linear, squared, work_table
                            );
                        } else {
                            summit::mailman::pre_multiply_rowmajor_with_squares(
                                packed.packed32.data()
                                    + static_cast<size_t>(segment)
                                        * static_cast<size_t>(rows_),
                                actual, rows_, q,
                                packed_feature_rhs_.data() + q0, rhs_columns,
                                linear, squared, work_table
                            );
                        }
                        for (int local_variant = 0; local_variant < actual;
                             ++local_variant) {
                            const int variant = base + local_variant;
                            const double mean =
                                packed.mean[static_cast<size_t>(variant)];
                            const double inv_std =
                                packed.inv_std[static_cast<size_t>(variant)];
                            const double inv_var = inv_std * inv_std;
                            const double* raw_linear = linear
                                + static_cast<size_t>(local_variant)
                                    * static_cast<size_t>(q);
                            const double* raw_squared = squared
                                + static_cast<size_t>(local_variant)
                                    * static_cast<size_t>(q);
                            double* moments = projected
                                + static_cast<size_t>(variant)
                                    * static_cast<size_t>(feature_columns_);
                            double* scalars = scalar
                                + static_cast<size_t>(variant)
                                    * static_cast<size_t>(scalar_rows);
                            for (int local_column = 0; local_column < q;
                                 ++local_column) {
                                const int column = q0 + local_column;
                                long double missing_sum = 0.0L;
                                for (int missing_row : packed.missing_rows[
                                         static_cast<size_t>(variant)]) {
                                    missing_sum += packed_feature_rhs_[
                                        static_cast<size_t>(missing_row)
                                            * static_cast<size_t>(rhs_columns)
                                        + static_cast<size_t>(column)
                                    ];
                                }
                                if (column < feature_columns_) {
                                    moments[column] = inv_std * (
                                        raw_linear[local_column]
                                        - mean * packed_feature_rhs_sums_[
                                            static_cast<size_t>(column)
                                        ]
                                        + mean * static_cast<double>(missing_sum)
                                    );
                                } else {
                                    const int scalar_index =
                                        column - feature_columns_;
                                    scalars[scalar_index] = inv_var * (
                                        raw_squared[local_column]
                                        - 2.0 * mean * raw_linear[local_column]
                                        + mean * mean * (
                                            packed_feature_rhs_sums_[
                                                static_cast<size_t>(column)
                                            ]
                                            - static_cast<double>(missing_sum)
                                        )
                                    );
                                }
                            }
                        }
                    }
                }
            }
            finalize_feature_moments(
                scalar, projected, variants, eps_var,
                standardized, scale_x, scale_w, norm_x, norm_w, diag_x,
                diag_w, corr_xw, max_leak_x, max_leak_w
            );
        }
        feature_calls_.fetch_add(1, std::memory_order_relaxed);
        packed_feature_calls_.fetch_add(1, std::memory_order_relaxed);
        nb::dict result;
        result["scale_x"] = std::move(scale_x_out);
        result["scale_w"] = std::move(scale_w_out);
        result["norm_x"] = std::move(norm_x_out);
        result["norm_w"] = std::move(norm_w_out);
        result["diag_nxe_x"] = std::move(diag_x_out);
        result["diag_nxe_w"] = std::move(diag_w_out);
        result["corr_xw"] = std::move(corr_xw_out);
        result["max_projection_leakage_additive"] =
            std::move(max_leak_x_out);
        result["max_projection_leakage_interaction"] =
            std::move(max_leak_w_out);
        result["repaired_gemm_output_columns"] = 0;
        return result;
    }

    // ``annotation_is_sqrt`` marks an annotation operand that already holds
    // element-wise square roots of a constructor-validated canonical matrix
    // (the direct context's cache); the kernel then skips its own sqrt pass
    // and the redundant re-validation.  Values are bitwise identical either
    // way because sqrt of the same double is deterministic.
    int64_t source_block(
        nb_mat2f_rw<double> target,
        nb_mat2f_ro<double> genotype,
        nb_mat2f_ro<double> probes,
        nb_mat2f_ro<double> annotation,
        nb_mat2f_ro<double> scale_x,
        nb_mat2f_ro<double> scale_w,
        int environment_start,
        bool annotation_is_sqrt = false
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        validate_genotype(genotype, "multi-environment source genotype");
        const int variants = checked_blas_dim(
            genotype.shape(1), "multi-environment source variants"
        );
        const int probe_count = checked_blas_dim(
            probes.shape(1), "multi-environment source probes"
        );
        const int tile = checked_blas_dim(
            scale_x.shape(1), "multi-environment source tile"
        );
        if (probe_count <= 0 || tile <= 0 || environment_start < 0 ||
            environment_start + tile > environments_ ||
            checked_blas_dim(probes.shape(0),
                             "multi-environment source probe rows") != variants ||
            checked_blas_dim(annotation.shape(0),
                             "multi-environment source annotation rows") != variants ||
            checked_blas_dim(annotation.shape(1),
                             "multi-environment source annotations") != annotation_bins_ ||
            checked_blas_dim(scale_x.shape(0),
                             "multi-environment source scale rows") != variants ||
            scale_w.shape(0) != scale_x.shape(0) ||
            scale_w.shape(1) != scale_x.shape(1)) {
            throw std::runtime_error(
                "Multi-environment native source inputs have incompatible dimensions"
            );
        }
        const int columns = checked_blas_dim(
            checked_mul(static_cast<size_t>(annotation_bins_),
                        static_cast<size_t>(probe_count),
                        "multi-environment source columns"),
            "multi-environment source columns"
        );
        const int wide_columns = checked_blas_dim(
            checked_mul(2U * static_cast<size_t>(tile),
                        static_cast<size_t>(columns),
                        "multi-environment packed source columns"),
            "multi-environment packed source columns"
        );
        if (checked_blas_dim(target.shape(0),
                             "multi-environment source target rows") != rows_ ||
            checked_blas_dim(target.shape(1),
                             "multi-environment source target columns") != wide_columns) {
            throw std::runtime_error(
                "Multi-environment native packed source target is mis-sized"
            );
        }
        NativeGemmOutputTelemetryScope output_scope;
        require_execution_scratch_live("multi-environment source scratch");
        bool contribution_allocated = false;
        double* contribution = prepare_reusable_native_gemm_output_mat2f(
            source_output_scratch_,
            static_cast<size_t>(rows_),
            static_cast<size_t>(wide_columns),
            output_scope,
            contribution_allocated,
            admitted_allocation_capacity(
                source_output_capacity_elements_,
                "multi-environment source output scratch"
            )
        );
        record_scratch_role_use(
            source_output_role_, contribution_allocated,
            static_cast<size_t>(rows_), static_cast<size_t>(wide_columns),
            source_output_scratch_->capacity_byte_count()
        );
        (contribution_allocated ? scratch_output_allocations_
                                : scratch_output_reuses_)
            .fetch_add(1, std::memory_order_relaxed);
        GemmIntegrityResolution integrity_resolution;
        {
            nb::gil_scoped_release release;
            const size_t weight_elements = checked_mul(
                static_cast<size_t>(variants),
                static_cast<size_t>(wide_columns),
                "multi-environment source weights"
            );
            double* weights = prepare_vector_scratch(
                source_weights_scratch_, source_weights_role_,
                weight_elements, source_weights_capacity_elements_,
                "multi-environment source weight scratch"
            );
            const size_t sqrt_annotation_elements = checked_mul(
                static_cast<size_t>(variants),
                static_cast<size_t>(annotation_bins_),
                "multi-environment square-root annotations"
            );
            const double* sqrt_annotation = nullptr;
            int invalid_input = 0;
            if (annotation_is_sqrt) {
                sqrt_annotation = annotation.data();
            } else {
                double* sqrt_scratch = prepare_vector_scratch(
                    source_sqrt_annotation_scratch_, source_annotation_role_,
                    sqrt_annotation_elements,
                    source_annotation_capacity_elements_,
                    "multi-environment source annotation scratch"
                );
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_input)
#endif
                for (int64_t index = 0;
                     index < static_cast<int64_t>(sqrt_annotation_elements);
                     ++index) {
                    const double value = annotation.data()[index];
                    invalid_input |= !std::isfinite(value) || value < 0.0;
                    sqrt_scratch[static_cast<size_t>(index)] =
                        std::sqrt(std::max(0.0, value));
                }
                sqrt_annotation = sqrt_scratch;
            }
            const int64_t probe_elements =
                static_cast<int64_t>(variants) * probe_count;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_input)
#endif
            for (int64_t index = 0; index < probe_elements; ++index) {
                invalid_input |= !std::isfinite(probes.data()[index]);
            }
            const int64_t scale_elements =
                static_cast<int64_t>(variants) * tile;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_input)
#endif
            for (int64_t index = 0; index < scale_elements; ++index) {
                const double sx = scale_x.data()[index];
                const double sw = scale_w.data()[index];
                invalid_input |= !std::isfinite(sx) || !std::isfinite(sw) ||
                    sx <= 0.0 || sw <= 0.0;
            }
            if (invalid_input != 0) {
                throw std::runtime_error(
                    "Multi-environment native source inputs contain invalid values"
                );
            }
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static) \
                num_threads(threads_)
#endif
            for (int local_environment = 0; local_environment < tile;
                 ++local_environment) {
                for (int bin = 0; bin < annotation_bins_; ++bin) {
                    for (int probe = 0; probe < probe_count; ++probe) {
                        const int additive_column =
                            local_environment * 2 * columns +
                            bin * probe_count + probe;
                        const int interaction_column = additive_column + columns;
                        double* additive = weights +
                            static_cast<size_t>(additive_column) *
                                static_cast<size_t>(variants);
                        double* interaction = weights +
                            static_cast<size_t>(interaction_column) *
                                static_cast<size_t>(variants);
                        for (int variant = 0; variant < variants; ++variant) {
                            const double annotation_weight = sqrt_annotation[
                                static_cast<size_t>(bin) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            const double probe_value = probes.data()[
                                static_cast<size_t>(probe) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            const double sx = scale_x.data()[
                                static_cast<size_t>(local_environment) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            const double sw = scale_w.data()[
                                static_cast<size_t>(local_environment) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            additive[variant] = annotation_weight * probe_value * sx;
                            interaction[variant] = annotation_weight * probe_value * sw;
                        }
                    }
                }
            }
            (void)dgemm_nn_partitioned_rows(
                rows_, wide_columns, variants,
                genotype.data(), rows_, weights, variants,
                contribution, rows_, threads_, 1.0, 0.0,
                &integrity_resolution
            );
            record_integrity_resolution(integrity_resolution);
            source_output_scratch_->verify_after_repair();
            output_scope.complete();
            int invalid_output = 0;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_output)
#endif
            for (int column = 0; column < wide_columns; ++column) {
                const int local_environment = column / (2 * columns);
                const bool interaction =
                    (column % (2 * columns)) >= columns;
                const double* environment = environment_values_.data() +
                    static_cast<size_t>(environment_start + local_environment) *
                        static_cast<size_t>(rows_);
                const double* source = contribution +
                    static_cast<size_t>(column) * static_cast<size_t>(rows_);
                double* destination = target.data() +
                    static_cast<size_t>(column) * static_cast<size_t>(rows_);
                for (int row = 0; row < rows_; ++row) {
                    destination[row] += interaction
                        ? environment[row] * source[row] : source[row];
                    invalid_output |= !std::isfinite(destination[row]);
                }
            }
            if (invalid_output != 0) {
                throw std::runtime_error(
                    "Multi-environment native source accumulation is non-finite"
                );
            }
        }
        source_calls_.fetch_add(1, std::memory_order_relaxed);
        return integrity_resolution.materially_repaired_columns;
    }

    int64_t source_block_packed(
        nb_mat2f_rw<double> target,
        const MailmanPackedBlock& packed,
        nb_mat2f_ro<double> probes,
        nb_mat2f_ro<double> annotation,
        nb_mat2f_ro<double> scale_x,
        nb_mat2f_ro<double> scale_w,
        int environment_start,
        bool annotation_is_sqrt = false
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        const int variants = packed.L;
        const int probe_count = checked_blas_dim(
            probes.shape(1), "packed multi-environment source probes"
        );
        const int tile = checked_blas_dim(
            scale_x.shape(1), "packed multi-environment source tile"
        );
        if (packed.N != rows_ || variants <= 0 || packed.segment_size <= 0
            || packed.n_segments <= 0 || packed.table_size <= 0
            || packed.mean.size() != static_cast<size_t>(variants)
            || packed.inv_std.size() != static_cast<size_t>(variants)
            || probe_count <= 0 || tile <= 0 || environment_start < 0
            || environment_start + tile > environments_
            || checked_blas_dim(
                probes.shape(0), "packed multi-environment source probe rows"
            ) != variants
            || checked_blas_dim(
                annotation.shape(0),
                "packed multi-environment source annotation rows"
            ) != variants
            || checked_blas_dim(
                annotation.shape(1),
                "packed multi-environment source annotations"
            ) != annotation_bins_
            || checked_blas_dim(
                scale_x.shape(0), "packed multi-environment source scale rows"
            ) != variants
            || scale_w.shape(0) != scale_x.shape(0)
            || scale_w.shape(1) != scale_x.shape(1)) {
            throw std::runtime_error(
                "Packed multi-environment source inputs have incompatible dimensions"
            );
        }
        const int columns = checked_blas_dim(
            checked_mul(
                static_cast<size_t>(annotation_bins_),
                static_cast<size_t>(probe_count),
                "packed multi-environment source columns"
            ),
            "packed multi-environment source columns"
        );
        const int wide_columns = checked_blas_dim(
            checked_mul(
                2U * static_cast<size_t>(tile),
                static_cast<size_t>(columns),
                "packed multi-environment source width"
            ),
            "packed multi-environment source width"
        );
        if (checked_blas_dim(
                target.shape(0), "packed multi-environment source rows"
            ) != rows_
            || checked_blas_dim(
                target.shape(1), "packed multi-environment source columns"
            ) != wide_columns) {
            throw std::runtime_error(
                "Packed multi-environment source target is mis-sized"
            );
        }

        const size_t weight_elements = checked_mul(
            static_cast<size_t>(variants),
            static_cast<size_t>(wide_columns),
            "packed multi-environment source weights"
        );
        double* weights = prepare_vector_scratch(
            source_weights_scratch_, source_weights_role_,
            weight_elements, source_weights_capacity_elements_,
            "packed multi-environment source weight scratch"
        );
        std::vector<double> mean_correction(
            static_cast<size_t>(wide_columns), 0.0
        );
        int invalid_input = 0;
        {
            nb::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid_input)
#endif
            for (int variant = 0; variant < variants; ++variant) {
                const double genotype_scale =
                    packed.inv_std[static_cast<size_t>(variant)];
                const double genotype_mean =
                    packed.mean[static_cast<size_t>(variant)];
                invalid_input |= !std::isfinite(genotype_scale)
                    || genotype_scale <= 0.0
                    || !std::isfinite(genotype_mean);
                double* weight_row = weights
                    + static_cast<size_t>(variant)
                        * static_cast<size_t>(wide_columns);
                for (int local_environment = 0;
                     local_environment < tile; ++local_environment) {
                    const double sx = scale_x.data()[
                        static_cast<size_t>(local_environment)
                            * static_cast<size_t>(variants)
                        + static_cast<size_t>(variant)
                    ];
                    const double sw = scale_w.data()[
                        static_cast<size_t>(local_environment)
                            * static_cast<size_t>(variants)
                        + static_cast<size_t>(variant)
                    ];
                    invalid_input |= !std::isfinite(sx) || sx <= 0.0
                        || !std::isfinite(sw) || sw <= 0.0;
                    for (int bin = 0; bin < annotation_bins_; ++bin) {
                        const double annotation_value = annotation.data()[
                            static_cast<size_t>(bin)
                                * static_cast<size_t>(variants)
                            + static_cast<size_t>(variant)
                        ];
                        invalid_input |= !std::isfinite(annotation_value)
                            || annotation_value < 0.0;
                        const double annotation_weight = annotation_is_sqrt
                            ? annotation_value
                            : std::sqrt(std::max(0.0, annotation_value));
                        for (int probe = 0; probe < probe_count; ++probe) {
                            const double probe_value = probes.data()[
                                static_cast<size_t>(probe)
                                    * static_cast<size_t>(variants)
                                + static_cast<size_t>(variant)
                            ];
                            invalid_input |= !std::isfinite(probe_value);
                            const int additive_column =
                                local_environment * 2 * columns
                                + bin * probe_count + probe;
                            const int interaction_column =
                                additive_column + columns;
                            weight_row[additive_column] = genotype_scale
                                * annotation_weight * probe_value * sx;
                            weight_row[interaction_column] = genotype_scale
                                * annotation_weight * probe_value * sw;
                        }
                    }
                }
                (void)genotype_mean;
            }
            if (invalid_input != 0) {
                throw std::runtime_error(
                    "Packed multi-environment source inputs contain invalid values"
                );
            }
            for (int variant = 0; variant < variants; ++variant) {
                const double mean = packed.mean[static_cast<size_t>(variant)];
                const double* weight_row = weights
                    + static_cast<size_t>(variant)
                        * static_cast<size_t>(wide_columns);
#ifdef _OPENMP
#pragma omp simd
#endif
                for (int column = 0; column < wide_columns; ++column) {
                    mean_correction[static_cast<size_t>(column)] +=
                        mean * weight_row[static_cast<size_t>(column)];
                }
            }

            require_frozen_mailman_geometry(packed);
            const int qpanel = mailman_plan_frozen_
                ? mailman_qpanel_source_
                : summit::mailman::qpanel_width<double>(
                    packed.table_size, wide_columns, packed.segment_size
                );
            for (int q0 = 0; q0 < wide_columns; q0 += qpanel) {
                const int q = std::min(qpanel, wide_columns - q0);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
                {
#ifdef _OPENMP
                    const int thread = omp_get_thread_num();
                    const int thread_count = omp_get_num_threads();
#else
                    const int thread = 0;
                    const int thread_count = 1;
#endif
                    // Partition the output columns, not the sample rows.  A
                    // row partition makes every thread rebuild the same
                    // 3^segment lookup table; a column partition constructs
                    // each table cell exactly once while retaining exclusive
                    // output ownership and column-local NUMA first touch.
                    const int base_columns = q / thread_count;
                    const int remainder = q % thread_count;
                    const int thread_column_start = thread * base_columns
                        + std::min(thread, remainder);
                    const int thread_columns = base_columns
                        + (thread < remainder ? 1 : 0);
                    const size_t work_elements = checked_mul(
                        static_cast<size_t>(packed.table_size),
                        static_cast<size_t>(thread_columns),
                        "packed source lookup table"
                    );
                    MailmanWorkerArena& arena = mailman_worker_arena(
                        work_elements, 0U, 0U, false
                    );
                    arena.table_dirty = true;
                    std::vector<double>& work_table = arena.work_table;
                    for (int64_t segment = 0;
                         thread_columns > 0 && segment < packed.n_segments;
                         ++segment) {
                        const int base = static_cast<int>(
                            segment * static_cast<int64_t>(packed.segment_size)
                        );
                        const int actual = std::min(
                            packed.segment_size, variants - base
                        );
                        const auto scale = [&](int row, int local_column) {
                            const int column = q0 + thread_column_start
                                + local_column;
                            const int local_environment = column
                                / (2 * columns);
                            const bool interaction =
                                (column % (2 * columns)) >= columns;
                            return interaction
                                ? environment_values_[
                                    static_cast<size_t>(
                                        environment_start + local_environment
                                    ) * static_cast<size_t>(rows_)
                                    + static_cast<size_t>(row)
                                ]
                                : 1.0;
                        };
                        if (packed.use_u16) {
                            summit::mailman::post_multiply_colmajor_subset_transform(
                                packed.packed16.data()
                                    + static_cast<size_t>(segment)
                                        * static_cast<size_t>(rows_),
                                actual, 0, rows_, thread_columns,
                                weights
                                    + static_cast<size_t>(base)
                                        * static_cast<size_t>(wide_columns)
                                    + static_cast<size_t>(q0)
                                    + static_cast<size_t>(thread_column_start),
                                wide_columns,
                                target.data()
                                    + static_cast<size_t>(
                                        q0 + thread_column_start
                                    )
                                        * static_cast<size_t>(rows_),
                                rows_, work_table.data(), scale
                            );
                        } else {
                            summit::mailman::post_multiply_colmajor_subset_transform(
                                packed.packed32.data()
                                    + static_cast<size_t>(segment)
                                        * static_cast<size_t>(rows_),
                                actual, 0, rows_, thread_columns,
                                weights
                                    + static_cast<size_t>(base)
                                        * static_cast<size_t>(wide_columns)
                                    + static_cast<size_t>(q0)
                                    + static_cast<size_t>(thread_column_start),
                                wide_columns,
                                target.data()
                                    + static_cast<size_t>(
                                        q0 + thread_column_start
                                    )
                                        * static_cast<size_t>(rows_),
                                rows_, work_table.data(), scale
                            );
                        }
                    }
#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int local_column = 0;
                         local_column < q; ++local_column) {
                        const int column = q0 + local_column;
                        const int local_environment = column
                            / (2 * columns);
                        const bool interaction =
                            (column % (2 * columns)) >= columns;
                        double* destination = target.data()
                            + static_cast<size_t>(column)
                                * static_cast<size_t>(rows_);
                        const double correction = mean_correction[
                            static_cast<size_t>(column)
                        ];
                        const double* environment = environment_values_.data()
                            + static_cast<size_t>(
                                environment_start + local_environment
                            ) * static_cast<size_t>(rows_);
                        for (int row = 0; row < rows_; ++row) {
                            destination[row] -= correction
                                * (interaction ? environment[row] : 1.0);
                        }
                        // Missing dosages are mean-imputed to standardized
                        // zero. They are encoded as base-3 zero in the packed
                        // block, so cancel the centering term sparsely.
                        for (int variant = 0; variant < variants; ++variant) {
                            const double sparse_correction =
                                packed.mean[static_cast<size_t>(variant)]
                                * weights[
                                    static_cast<size_t>(variant)
                                        * static_cast<size_t>(wide_columns)
                                    + static_cast<size_t>(column)
                                ];
                            for (int missing_row : packed.missing_rows[
                                     static_cast<size_t>(variant)]) {
                                destination[missing_row] += sparse_correction
                                    * (interaction
                                        ? environment[missing_row] : 1.0);
                            }
                        }
                    }
                }
            }
            int invalid_output = 0;
            const int64_t output_elements =
                static_cast<int64_t>(rows_) * wide_columns;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid_output)
#endif
            for (int64_t index = 0; index < output_elements; ++index) {
                invalid_output |= !std::isfinite(target.data()[index]);
            }
            if (invalid_output != 0) {
                throw std::runtime_error(
                    "Packed multi-environment source accumulation is non-finite"
                );
            }
        }
        source_calls_.fetch_add(1, std::memory_order_relaxed);
        packed_source_calls_.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    nb_numpy_vec1<double> project_sources(
        nb_mat2f_rw<double> panel,
        int columns_per_environment,
        int environment_start
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        const int panel_rows = checked_blas_dim(
            panel.shape(0), "multi-environment projected source rows"
        );
        const int panel_columns = checked_blas_dim(
            panel.shape(1), "multi-environment projected source columns"
        );
        if (panel_rows != rows_ || columns_per_environment <= 0 ||
            panel_columns % (2 * columns_per_environment) != 0) {
            throw std::runtime_error(
                "Multi-environment native projection panel is mis-sized"
            );
        }
        const int tile = panel_columns / (2 * columns_per_environment);
        if (tile <= 0 || environment_start < 0 ||
            environment_start + tile > environments_) {
            throw std::runtime_error(
                "Multi-environment native projection tile is outside the environment axis"
            );
        }
        double* leakages = nullptr;
        auto leakage_out = make_owned_numpy_vec1<double>(
            static_cast<size_t>(tile), &leakages
        );
        std::unique_ptr<NativeGemmOutputTelemetryScope> coefficient_output_scope;
        NativeGemmOutputAllocation* coefficient_output_allocation = nullptr;
        double* coefficients = nullptr;
        std::unique_ptr<nb_numpy_mat2f<double>> coefficient_owner;
        if (common_rank_ > 0) {
            coefficient_output_scope =
                std::make_unique<NativeGemmOutputTelemetryScope>();
            coefficient_owner = std::make_unique<nb_numpy_mat2f<double>>(
                make_native_gemm_output_mat2f(
                    static_cast<size_t>(common_rank_),
                    static_cast<size_t>(panel_columns),
                    *coefficient_output_scope,
                    &coefficient_output_allocation,
                    &coefficients
                )
            );
        }
        {
            nb::gil_scoped_release release;
            if (common_rank_ > 0) {
                GemmIntegrityResolution integrity_resolution;
                (void)dgemm_tn_partitioned_columns(
                    common_rank_, panel_columns, rows_,
                    common_basis_.data(), rows_, panel.data(), rows_,
                    coefficients, common_rank_, threads_,
                    &integrity_resolution
                );
                record_integrity_resolution(integrity_resolution);
                coefficient_output_allocation->verify_after_repair();
                coefficient_output_scope->complete();
                dgemm_nn_tiled(
                    rows_, panel_columns, common_rank_,
                    common_basis_.data(), rows_, coefficients, common_rank_,
                    panel.data(), rows_, threads_, -1.0, 1.0
                );
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_)
#endif
            for (int column = 0; column < panel_columns; ++column) {
                const int local_environment =
                    column / (2 * columns_per_environment);
                const double* direction = directions_.data() +
                    static_cast<size_t>(environment_start + local_environment) *
                        static_cast<size_t>(rows_);
                double* values = panel.data() +
                    static_cast<size_t>(column) * static_cast<size_t>(rows_);
                long double direction_coefficient = 0.0L;
                long double mean = 0.0L;
                for (int row = 0; row < rows_; ++row) {
                    direction_coefficient +=
                        static_cast<long double>(direction[row]) * values[row];
                    mean += values[row];
                }
                mean /= static_cast<long double>(rows_);
                for (int row = 0; row < rows_; ++row) {
                    values[row] -= static_cast<double>(
                        direction_coefficient * direction[row] + mean
                    );
                }
            }
            for (int local_environment = 0; local_environment < tile;
                 ++local_environment) {
                const int environment = environment_start + local_environment;
                const int rank = ranks_[static_cast<size_t>(environment)];
                double maximum = 0.0;
                for (int local_column = 0;
                     local_column < 2 * columns_per_environment;
                     ++local_column) {
                    const int column =
                        local_environment * 2 * columns_per_environment +
                        local_column;
                    const double* values = panel.data() +
                        static_cast<size_t>(column) * static_cast<size_t>(rows_);
                    long double denominator = 0.0L;
                    long double leaked = 0.0L;
                    for (int row = 0; row < rows_; ++row) {
                        denominator +=
                            static_cast<long double>(values[row]) * values[row];
                    }
                    for (int a = 0; a < rank; ++a) {
                        const int64_t basis_index = power_indices_[
                            static_cast<size_t>(power_offsets_[
                                static_cast<size_t>(4 * environment)
                            ] + a)
                        ];
                        const double* basis = feature_basis_.data() +
                            static_cast<size_t>(basis_index) *
                                static_cast<size_t>(rows_);
                        long double coefficient = 0.0L;
                        for (int row = 0; row < rows_; ++row) {
                            coefficient +=
                                static_cast<long double>(basis[row]) * values[row];
                        }
                        leaked += coefficient * coefficient;
                    }
                    const double relative = denominator > 0.0L
                        ? std::sqrt(static_cast<double>(leaked / denominator))
                        : std::numeric_limits<double>::infinity();
                    if (!std::isfinite(relative)) {
                        throw std::runtime_error(
                            "Multi-environment native source leakage is non-finite"
                        );
                    }
                    maximum = std::max(maximum, relative);
                }
                if (maximum > 1.0e-9) {
                    throw std::runtime_error(
                        "Multi-environment native source projection has excessive leakage"
                    );
                }
                leakages[local_environment] = maximum;
            }
        }
        (void)coefficient_owner;
        projection_calls_.fetch_add(1, std::memory_order_relaxed);
        return leakage_out;
    }

    int64_t target_score_block(
        nb_mat2f_ro<double> genotype,
        const ProtectedRightPair& right_pair,
        nb_mat2f_ro<double> scale_x,
        nb_mat2f_ro<double> scale_w,
        nb_mat2c_rw<double> accum_xx,
        nb_mat2c_rw<double> accum_xw,
        nb_mat2c_rw<double> accum_wx,
        nb_mat2c_rw<double> accum_ww,
        int block_start,
        int total_variants,
        int probe_count,
        int environment_start
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        validate_genotype(genotype, "multi-environment target genotype");
        const int variants = checked_blas_dim(
            genotype.shape(1), "multi-environment target variants"
        );
        const int tile = checked_blas_dim(
            scale_x.shape(1), "multi-environment target tile"
        );
        if (block_start < 0 || total_variants <= 0 ||
            block_start + variants > total_variants || probe_count <= 0 ||
            tile <= 0 || environment_start < 0 ||
            environment_start + tile > environments_ ||
            checked_blas_dim(scale_x.shape(0),
                             "multi-environment target scale rows") != variants ||
            scale_w.shape(0) != scale_x.shape(0) ||
            scale_w.shape(1) != scale_x.shape(1)) {
            throw std::runtime_error(
                "Multi-environment native target scale/block metadata is invalid"
            );
        }
        const int columns = annotation_bins_ * probe_count;
        const int pair_columns = tile * 2 * columns;
        const size_t expected_pair_elements = checked_mul(
            2U,
            checked_mul(static_cast<size_t>(rows_),
                        static_cast<size_t>(pair_columns),
                        "multi-environment target pair"),
            "multi-environment target pair"
        );
        if (right_pair.rows() != rows_ ||
            right_pair.columns() != pair_columns ||
            right_pair.elements() != expected_pair_elements) {
            throw std::runtime_error(
                "Multi-environment native target pair is incompatible with the tile"
            );
        }
        for (const auto* accumulator :
             {&accum_xx, &accum_xw, &accum_wx, &accum_ww}) {
            if (checked_blas_dim(accumulator->shape(0),
                                 "multi-environment accumulator rows") !=
                    environments_ * total_variants ||
                checked_blas_dim(accumulator->shape(1),
                                 "multi-environment accumulator columns") !=
                    annotation_bins_) {
                throw std::runtime_error(
                    "Multi-environment native score accumulator is mis-sized"
                );
            }
        }
        const int output_columns = 2 * pair_columns;
        NativeGemmOutputTelemetryScope output_scope;
        require_execution_scratch_live("multi-environment target scratch");
        bool output_allocated = false;
        double* output = prepare_reusable_native_gemm_output_mat2f(
            target_output_scratch_,
            static_cast<size_t>(variants),
            static_cast<size_t>(output_columns),
            output_scope,
            output_allocated,
            admitted_allocation_capacity(
                target_output_capacity_elements_,
                "multi-environment target output scratch"
            )
        );
        record_scratch_role_use(
            target_output_role_, output_allocated,
            static_cast<size_t>(variants), static_cast<size_t>(output_columns),
            target_output_scratch_->capacity_byte_count()
        );
        (output_allocated ? scratch_output_allocations_
                          : scratch_output_reuses_)
            .fetch_add(1, std::memory_order_relaxed);
        GemmIntegrityResolution integrity_resolution;
        {
            nb::gil_scoped_release release;
            int invalid_scale = 0;
            const int64_t scale_elements =
                static_cast<int64_t>(variants) * tile;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_scale)
#endif
            for (int64_t index = 0; index < scale_elements; ++index) {
                const double sx = scale_x.data()[index];
                const double sw = scale_w.data()[index];
                invalid_scale |= !std::isfinite(sx) || !std::isfinite(sw) ||
                    sx <= 0.0 || sw <= 0.0;
            }
            if (invalid_scale != 0) {
                throw std::runtime_error(
                    "Multi-environment native target scales contain invalid values"
                );
            }
            (void)dgemm_tn_partitioned_rows(
                variants, output_columns, rows_,
                genotype.data(), rows_, right_pair.data(), rows_,
                output, variants, threads_, 1.0, 0.0, nullptr, true,
                &integrity_resolution
            );
            record_integrity_resolution(integrity_resolution);
            target_output_scratch_->verify_after_repair();
            output_scope.complete();
            int invalid_output = 0;
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static) \
                num_threads(threads_) reduction(|:invalid_output)
#endif
            for (int local_environment = 0; local_environment < tile;
                 ++local_environment) {
                for (int variant = 0; variant < variants; ++variant) {
                    const int environment = environment_start + local_environment;
                    const double sx = scale_x.data()[
                        static_cast<size_t>(local_environment) *
                            static_cast<size_t>(variants) +
                        static_cast<size_t>(variant)
                    ];
                    const double sw = scale_w.data()[
                        static_cast<size_t>(local_environment) *
                            static_cast<size_t>(variants) +
                        static_cast<size_t>(variant)
                    ];
                    const double df = static_cast<double>(
                        rows_ - ranks_[static_cast<size_t>(environment)]
                    );
                    const double score_scale = 1.0 / (df * df);
                    const size_t accumulator_row =
                        static_cast<size_t>(environment * total_variants +
                                            block_start + variant);
                    for (int bin = 0; bin < annotation_bins_; ++bin) {
                        double sum_xx = 0.0;
                        double sum_xw = 0.0;
                        double sum_wx = 0.0;
                        double sum_ww = 0.0;
                        const int source_base =
                            local_environment * 2 * columns +
                            bin * probe_count;
#ifdef _OPENMP
                        #pragma omp simd reduction(+:sum_xx,sum_xw,sum_wx,sum_ww)
#endif
                        for (int probe = 0; probe < probe_count; ++probe) {
                            const int additive_column = source_base + probe;
                            const int interaction_column =
                                source_base + columns + probe;
                            const double xx = sx * output[
                                static_cast<size_t>(additive_column) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            const double xw = sx * output[
                                static_cast<size_t>(interaction_column) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            const double wx = sw * output[
                                static_cast<size_t>(pair_columns + additive_column) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            const double ww = sw * output[
                                static_cast<size_t>(pair_columns + interaction_column) *
                                    static_cast<size_t>(variants) +
                                static_cast<size_t>(variant)
                            ];
                            sum_xx += xx * xx;
                            sum_xw += xw * xw;
                            sum_wx += wx * wx;
                            sum_ww += ww * ww;
                        }
                        const size_t index =
                            accumulator_row * static_cast<size_t>(annotation_bins_) +
                            static_cast<size_t>(bin);
                        accum_xx.data()[index] += sum_xx * score_scale;
                        accum_xw.data()[index] += sum_xw * score_scale;
                        accum_wx.data()[index] += sum_wx * score_scale;
                        accum_ww.data()[index] += sum_ww * score_scale;
                        invalid_output |=
                            !std::isfinite(accum_xx.data()[index]) ||
                            !std::isfinite(accum_xw.data()[index]) ||
                            !std::isfinite(accum_wx.data()[index]) ||
                            !std::isfinite(accum_ww.data()[index]);
                    }
                }
            }
            if (invalid_output != 0) {
                throw std::runtime_error(
                    "Multi-environment native score reduction is non-finite"
                );
            }
        }
        target_calls_.fetch_add(1, std::memory_order_relaxed);
        return integrity_resolution.materially_repaired_columns;
    }

    int64_t target_score_block_dense_pair(
        nb_mat2f_ro<double> genotype,
        nb_mat2f_ro<double> right,
        nb_mat2f_ro<double> scale_x,
        nb_mat2f_ro<double> scale_w,
        nb_mat2c_rw<double> accum_xx,
        nb_mat2c_rw<double> accum_xw,
        nb_mat2c_rw<double> accum_wx,
        nb_mat2c_rw<double> accum_ww,
        int block_start,
        int total_variants,
        int probe_count,
        int environment_start
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        validate_genotype(genotype, "dense combined target genotype");
        const int variants = checked_blas_dim(
            genotype.shape(1), "dense combined target variants"
        );
        const int tile = checked_blas_dim(
            scale_x.shape(1), "dense combined target tile"
        );
        const int columns = checked_blas_dim(
            checked_mul(
                static_cast<size_t>(annotation_bins_),
                static_cast<size_t>(probe_count),
                "dense combined target columns"
            ),
            "dense combined target columns"
        );
        const int pair_columns = checked_blas_dim(
            checked_mul(
                2U * static_cast<size_t>(tile),
                static_cast<size_t>(columns),
                "dense combined target pair columns"
            ),
            "dense combined target pair columns"
        );
        const int right_columns = checked_blas_dim(
            2U * static_cast<size_t>(pair_columns),
            "dense combined target right columns"
        );
        if (block_start < 0 || total_variants <= 0
            || block_start + variants > total_variants || probe_count <= 0
            || tile <= 0 || environment_start < 0
            || environment_start + tile > environments_
            || checked_blas_dim(
                scale_x.shape(0), "dense combined target scale rows"
            ) != variants
            || scale_w.shape(0) != scale_x.shape(0)
            || scale_w.shape(1) != scale_x.shape(1)
            || checked_blas_dim(
                right.shape(0), "dense combined target right rows"
            ) != rows_
            || checked_blas_dim(
                right.shape(1), "dense combined target right columns"
            ) != right_columns) {
            throw std::runtime_error(
                "Dense combined target metadata is incompatible"
            );
        }
        for (const auto* accumulator :
             {&accum_xx, &accum_xw, &accum_wx, &accum_ww}) {
            if (checked_blas_dim(
                    accumulator->shape(0),
                    "dense combined target accumulator rows"
                ) != environments_ * total_variants
                || checked_blas_dim(
                    accumulator->shape(1),
                    "dense combined target accumulator columns"
                ) != annotation_bins_) {
                throw std::runtime_error(
                    "Dense combined target accumulator is mis-sized"
                );
            }
        }

        NativeGemmOutputTelemetryScope output_scope;
        require_execution_scratch_live("dense combined target scratch");
        bool output_allocated = false;
        double* output = prepare_reusable_native_gemm_output_mat2f(
            target_output_scratch_,
            static_cast<size_t>(variants),
            static_cast<size_t>(right_columns),
            output_scope, output_allocated,
            admitted_allocation_capacity(
                target_output_capacity_elements_,
                "dense combined target output scratch"
            )
        );
        record_scratch_role_use(
            target_output_role_, output_allocated,
            static_cast<size_t>(variants), static_cast<size_t>(right_columns),
            target_output_scratch_->capacity_byte_count()
        );
        (output_allocated ? scratch_output_allocations_
                          : scratch_output_reuses_)
            .fetch_add(1, std::memory_order_relaxed);
        GemmIntegrityResolution integrity_resolution;
        {
            nb::gil_scoped_release release;
            int invalid_scale = 0;
            const int64_t scale_elements =
                static_cast<int64_t>(variants) * tile;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid_scale)
#endif
            for (int64_t index = 0; index < scale_elements; ++index) {
                const double sx = scale_x.data()[index];
                const double sw = scale_w.data()[index];
                invalid_scale |= !std::isfinite(sx) || !std::isfinite(sw)
                    || sx <= 0.0 || sw <= 0.0;
            }
            if (invalid_scale != 0) {
                throw std::runtime_error(
                    "Dense combined target scale is invalid"
                );
            }
            (void)dgemm_tn_partitioned_rows(
                variants, right_columns, rows_,
                genotype.data(), rows_, right.data(), rows_,
                output, variants, threads_, 1.0, 0.0, nullptr, true,
                &integrity_resolution
            );
            record_integrity_resolution(integrity_resolution);
            target_output_scratch_->verify_after_repair();
            output_scope.complete();

            int invalid_output = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_) \
    reduction(|:invalid_output)
#endif
            for (int local_environment = 0;
                 local_environment < tile; ++local_environment) {
                for (int variant = 0; variant < variants; ++variant) {
                    const int environment =
                        environment_start + local_environment;
                    const double sx = scale_x.data()[
                        static_cast<size_t>(local_environment)
                            * static_cast<size_t>(variants)
                        + static_cast<size_t>(variant)
                    ];
                    const double sw = scale_w.data()[
                        static_cast<size_t>(local_environment)
                            * static_cast<size_t>(variants)
                        + static_cast<size_t>(variant)
                    ];
                    const double df = static_cast<double>(
                        rows_ - ranks_[static_cast<size_t>(environment)]
                    );
                    const double score_scale = 1.0 / (df * df);
                    const size_t accumulator_row = static_cast<size_t>(
                        environment * total_variants + block_start + variant
                    );
                    for (int bin = 0; bin < annotation_bins_; ++bin) {
                        double sum_xx = 0.0;
                        double sum_xw = 0.0;
                        double sum_wx = 0.0;
                        double sum_ww = 0.0;
                        const int source_base =
                            local_environment * 2 * columns
                            + bin * probe_count;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum_xx,sum_xw,sum_wx,sum_ww)
#endif
                        for (int probe = 0; probe < probe_count; ++probe) {
                            const int additive_column = source_base + probe;
                            const int interaction_column =
                                source_base + columns + probe;
                            const double xx = sx * output[
                                static_cast<size_t>(additive_column)
                                    * static_cast<size_t>(variants)
                                + static_cast<size_t>(variant)
                            ];
                            const double xw = sx * output[
                                static_cast<size_t>(interaction_column)
                                    * static_cast<size_t>(variants)
                                + static_cast<size_t>(variant)
                            ];
                            const double wx = sw * output[
                                static_cast<size_t>(
                                    pair_columns + additive_column
                                ) * static_cast<size_t>(variants)
                                + static_cast<size_t>(variant)
                            ];
                            const double ww = sw * output[
                                static_cast<size_t>(
                                    pair_columns + interaction_column
                                ) * static_cast<size_t>(variants)
                                + static_cast<size_t>(variant)
                            ];
                            sum_xx += xx * xx;
                            sum_xw += xw * xw;
                            sum_wx += wx * wx;
                            sum_ww += ww * ww;
                        }
                        const size_t index = accumulator_row
                            * static_cast<size_t>(annotation_bins_)
                            + static_cast<size_t>(bin);
                        accum_xx.data()[index] += sum_xx * score_scale;
                        accum_xw.data()[index] += sum_xw * score_scale;
                        accum_wx.data()[index] += sum_wx * score_scale;
                        accum_ww.data()[index] += sum_ww * score_scale;
                        invalid_output |=
                            !std::isfinite(accum_xx.data()[index])
                            || !std::isfinite(accum_xw.data()[index])
                            || !std::isfinite(accum_wx.data()[index])
                            || !std::isfinite(accum_ww.data()[index]);
                    }
                }
            }
            if (invalid_output != 0) {
                throw std::runtime_error(
                    "Dense combined target reduction is non-finite"
                );
            }
        }
        target_calls_.fetch_add(1, std::memory_order_relaxed);
        return integrity_resolution.materially_repaired_columns;
    }

    int64_t target_score_block_packed(
        const MailmanPackedBlock& packed,
        nb_mat2f_ro<double> source_panel,
        nb_mat2f_ro<double> scale_x,
        nb_mat2f_ro<double> scale_w,
        nb_mat2c_rw<double> accum_xx,
        nb_mat2c_rw<double> accum_xw,
        nb_mat2c_rw<double> accum_wx,
        nb_mat2c_rw<double> accum_ww,
        int block_start,
        int total_variants,
        int probe_count,
        int environment_start
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        const int variants = packed.L;
        const int tile = checked_blas_dim(
            scale_x.shape(1), "packed multi-environment target tile"
        );
        if (packed.N != rows_ || variants <= 0 || packed.segment_size <= 0
            || packed.n_segments <= 0 || packed.table_size <= 0
            || packed.mean.size() != static_cast<size_t>(variants)
            || packed.inv_std.size() != static_cast<size_t>(variants)
            || block_start < 0 || total_variants <= 0
            || block_start + variants > total_variants || probe_count <= 0
            || tile <= 0 || environment_start < 0
            || environment_start + tile > environments_
            || checked_blas_dim(
                scale_x.shape(0), "packed multi-environment target scale rows"
            ) != variants
            || scale_w.shape(0) != scale_x.shape(0)
            || scale_w.shape(1) != scale_x.shape(1)) {
            throw std::runtime_error(
                "Packed multi-environment target metadata is invalid"
            );
        }
        const int columns = checked_blas_dim(
            checked_mul(
                static_cast<size_t>(annotation_bins_),
                static_cast<size_t>(probe_count),
                "packed multi-environment target columns"
            ),
            "packed multi-environment target columns"
        );
        const int pair_columns = checked_blas_dim(
            checked_mul(
                2U * static_cast<size_t>(tile),
                static_cast<size_t>(columns),
                "packed multi-environment source columns"
            ),
            "packed multi-environment source columns"
        );
        if (checked_blas_dim(
                source_panel.shape(0), "packed target source rows"
            ) != rows_
            || checked_blas_dim(
                source_panel.shape(1), "packed target source columns"
            ) != pair_columns) {
            throw std::runtime_error(
                "Packed multi-environment source panel is mis-sized"
            );
        }
        for (const auto* accumulator :
             {&accum_xx, &accum_xw, &accum_wx, &accum_ww}) {
            if (checked_blas_dim(
                    accumulator->shape(0), "packed target accumulator rows"
                ) != environments_ * total_variants
                || checked_blas_dim(
                    accumulator->shape(1), "packed target accumulator columns"
                ) != annotation_bins_) {
                throw std::runtime_error(
                    "Packed multi-environment score accumulator is mis-sized"
                );
            }
        }

        const int output_columns = 2 * pair_columns;
        const size_t output_elements = checked_mul(
            static_cast<size_t>(variants),
            static_cast<size_t>(output_columns),
            "packed multi-environment target output"
        );
        const bool target_mailman_allocated =
            target_mailman_output_scratch_.capacity() < output_elements
            || (scratch_capacities_frozen_
                && target_mailman_output_scratch_.capacity()
                    < target_mailman_capacity_elements_);
        double* output = prepare_vector_scratch(
            target_mailman_output_scratch_, target_mailman_role_,
            output_elements, target_mailman_capacity_elements_,
            "packed multi-environment target output scratch"
        );
        (target_mailman_allocated ? scratch_output_allocations_
                                  : scratch_output_reuses_)
            .fetch_add(1, std::memory_order_relaxed);
        require_frozen_mailman_geometry(packed);
        const int qpanel = mailman_plan_frozen_
            ? mailman_qpanel_target_
            : summit::mailman::qpanel_width<double>(
                packed.table_size, output_columns, packed.segment_size
            );
        std::vector<double> rhs_sums(static_cast<size_t>(qpanel), 0.0);
        int invalid = 0;
        {
            nb::gil_scoped_release release;
            for (int q0 = 0; q0 < output_columns; q0 += qpanel) {
                const int q = std::min(qpanel, output_columns - q0);
                const auto rhs_value = [&](int row, int local_column) {
                    const int virtual_column = q0 + local_column;
                    const bool weighted = virtual_column >= pair_columns;
                    const int source_column = weighted
                        ? virtual_column - pair_columns
                        : virtual_column;
                    const int local_environment = source_column
                        / (2 * columns);
                    double value = source_panel.data()[
                        static_cast<size_t>(source_column)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ];
                    if (weighted) {
                        value *= environment_values_[
                            static_cast<size_t>(
                                environment_start + local_environment
                            ) * static_cast<size_t>(rows_)
                            + static_cast<size_t>(row)
                        ];
                    }
                    return value;
                };
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid)
#endif
                for (int local_column = 0;
                     local_column < q; ++local_column) {
                    long double total = 0.0L;
                    for (int row = 0; row < rows_; ++row) {
                        total += rhs_value(row, local_column);
                    }
                    rhs_sums[static_cast<size_t>(local_column)] =
                        static_cast<double>(total);
                    invalid |= !std::isfinite(
                        rhs_sums[static_cast<size_t>(local_column)]
                    );
                }
                if (invalid != 0) {
                    throw std::runtime_error(
                        "Packed multi-environment virtual target RHS is non-finite"
                    );
                }

#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
                {
                    const size_t table_elements = checked_mul(
                        static_cast<size_t>(packed.table_size),
                        static_cast<size_t>(q),
                        "packed target lookup table"
                    );
                    const size_t segment_elements = checked_mul(
                        static_cast<size_t>(packed.segment_size),
                        static_cast<size_t>(q),
                        "packed target segment output"
                    );
                    MailmanWorkerArena& worker = mailman_worker_arena(
                        table_elements, segment_elements, 0U, true
                    );
                    std::vector<double>& work_table = worker.work_table;
                    std::vector<double>& raw_segment = worker.segment_a;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int64_t segment = 0;
                         segment < packed.n_segments; ++segment) {
                        const int base = static_cast<int>(
                            segment * static_cast<int64_t>(packed.segment_size)
                        );
                        const int actual = std::min(
                            packed.segment_size, variants - base
                        );
                        if (packed.use_u16) {
                            summit::mailman::pre_multiply_generated(
                                packed.packed16.data()
                                    + static_cast<size_t>(segment)
                                        * static_cast<size_t>(rows_),
                                actual, rows_, q, rhs_value,
                                raw_segment.data(), work_table.data()
                            );
                        } else {
                            summit::mailman::pre_multiply_generated(
                                packed.packed32.data()
                                    + static_cast<size_t>(segment)
                                        * static_cast<size_t>(rows_),
                                actual, rows_, q, rhs_value,
                                raw_segment.data(), work_table.data()
                            );
                        }
                        for (int local_variant = 0;
                             local_variant < actual; ++local_variant) {
                            const int variant = base + local_variant;
                            const double mean = packed.mean[
                                static_cast<size_t>(variant)
                            ];
                            const double inv_std = packed.inv_std[
                                static_cast<size_t>(variant)
                            ];
                            const double* raw = raw_segment.data()
                                + static_cast<size_t>(local_variant)
                                    * static_cast<size_t>(q);
                            double* destination = output
                                + static_cast<size_t>(variant)
                                    * static_cast<size_t>(output_columns)
                                + static_cast<size_t>(q0);
                            for (int local_column = 0;
                                 local_column < q; ++local_column) {
                                long double missing_rhs = 0.0L;
                                for (int missing_row : packed.missing_rows[
                                         static_cast<size_t>(variant)]) {
                                    missing_rhs += rhs_value(
                                        missing_row, local_column
                                    );
                                }
                                destination[local_column] = inv_std * (
                                    raw[local_column]
                                    - mean * rhs_sums[
                                        static_cast<size_t>(local_column)
                                    ]
                                    + mean * static_cast<double>(missing_rhs)
                                );
                            }
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_) \
    reduction(|:invalid)
#endif
            for (int local_environment = 0;
                 local_environment < tile; ++local_environment) {
                for (int variant = 0; variant < variants; ++variant) {
                    const int environment =
                        environment_start + local_environment;
                    const double sx = scale_x.data()[
                        static_cast<size_t>(local_environment)
                            * static_cast<size_t>(variants)
                        + static_cast<size_t>(variant)
                    ];
                    const double sw = scale_w.data()[
                        static_cast<size_t>(local_environment)
                            * static_cast<size_t>(variants)
                        + static_cast<size_t>(variant)
                    ];
                    invalid |= !std::isfinite(sx) || sx <= 0.0
                        || !std::isfinite(sw) || sw <= 0.0;
                    const double score_scale = 1.0 / (
                        static_cast<double>(
                            rows_ - ranks_[static_cast<size_t>(environment)]
                        )
                        * static_cast<double>(
                            rows_ - ranks_[static_cast<size_t>(environment)]
                        )
                    );
                    const size_t accumulator_row = static_cast<size_t>(
                        environment * total_variants + block_start + variant
                    );
                    const double* variant_output = output
                        + static_cast<size_t>(variant)
                            * static_cast<size_t>(output_columns);
                    for (int bin = 0; bin < annotation_bins_; ++bin) {
                        double sum_xx = 0.0;
                        double sum_xw = 0.0;
                        double sum_wx = 0.0;
                        double sum_ww = 0.0;
                        const int source_base =
                            local_environment * 2 * columns
                            + bin * probe_count;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum_xx,sum_xw,sum_wx,sum_ww)
#endif
                        for (int probe = 0; probe < probe_count; ++probe) {
                            const int additive_column = source_base + probe;
                            const int interaction_column =
                                source_base + columns + probe;
                            const double xx = sx
                                * variant_output[additive_column];
                            const double xw = sx
                                * variant_output[interaction_column];
                            const double wx = sw
                                * variant_output[
                                    pair_columns + additive_column
                                ];
                            const double ww = sw
                                * variant_output[
                                    pair_columns + interaction_column
                                ];
                            sum_xx += xx * xx;
                            sum_xw += xw * xw;
                            sum_wx += wx * wx;
                            sum_ww += ww * ww;
                        }
                        const size_t index = accumulator_row
                            * static_cast<size_t>(annotation_bins_)
                            + static_cast<size_t>(bin);
                        accum_xx.data()[index] += sum_xx * score_scale;
                        accum_xw.data()[index] += sum_xw * score_scale;
                        accum_wx.data()[index] += sum_wx * score_scale;
                        accum_ww.data()[index] += sum_ww * score_scale;
                        invalid |= !std::isfinite(accum_xx.data()[index])
                            || !std::isfinite(accum_xw.data()[index])
                            || !std::isfinite(accum_wx.data()[index])
                            || !std::isfinite(accum_ww.data()[index]);
                    }
                }
            }
            if (invalid != 0) {
                throw std::runtime_error(
                    "Packed multi-environment target reduction is non-finite"
                );
            }
        }
        target_calls_.fetch_add(1, std::memory_order_relaxed);
        packed_target_calls_.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    void normalize_scores(
        nb_mat2c_rw<double> accum_xx,
        nb_mat2c_rw<double> accum_xw,
        nb_mat2c_rw<double> accum_wx,
        nb_mat2c_rw<double> accum_ww,
        int probe_count
    ) const {
        std::lock_guard<std::mutex> guard(mutex_);
        if (probe_count <= 0) {
            throw std::runtime_error(
                "Multi-environment native score normalization requires probes"
            );
        }
        const int rows = checked_blas_dim(
            accum_xx.shape(0), "multi-environment normalized score rows"
        );
        const int columns = checked_blas_dim(
            accum_xx.shape(1), "multi-environment normalized score columns"
        );
        if (rows <= 0 || rows % environments_ != 0 ||
            columns != annotation_bins_) {
            throw std::runtime_error(
                "Multi-environment native score normalization shape is invalid"
            );
        }
        for (const auto* accumulator :
             {&accum_xw, &accum_wx, &accum_ww}) {
            if (checked_blas_dim(
                    accumulator->shape(0),
                    "multi-environment normalized score rows"
                ) != rows ||
                checked_blas_dim(
                    accumulator->shape(1),
                    "multi-environment normalized score columns"
                ) != columns) {
                throw std::runtime_error(
                    "Multi-environment native score accumulators disagree"
                );
            }
        }
        const int64_t elements =
            static_cast<int64_t>(rows) * static_cast<int64_t>(columns);
        const double scale = 1.0 / static_cast<double>(probe_count);
        int invalid_output = 0;
        {
            nb::gil_scoped_release release;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(|:invalid_output)
#endif
            for (int64_t index = 0; index < elements; ++index) {
                accum_xx.data()[index] *= scale;
                accum_xw.data()[index] *= scale;
                accum_wx.data()[index] *= scale;
                accum_ww.data()[index] *= scale;
                invalid_output |= !std::isfinite(accum_xx.data()[index]) ||
                    !std::isfinite(accum_xw.data()[index]) ||
                    !std::isfinite(accum_wx.data()[index]) ||
                    !std::isfinite(accum_ww.data()[index]);
            }
        }
        if (invalid_output != 0) {
            throw std::runtime_error(
                "Multi-environment native normalized scores are non-finite"
            );
        }
        normalization_calls_.fetch_add(1, std::memory_order_relaxed);
    }

private:
    void finalize_feature_moments(
        const double* scalar, const double* projected, int variants,
        double eps_var, bool standardized,
        double* scale_x, double* scale_w, double* norm_x, double* norm_w,
        double* diag_x, double* diag_w, double* corr_xw,
        double* max_leak_x, double* max_leak_w
    ) const {
        const int scalar_rows = 1 + 3 * environments_;
        int invalid_feature = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid_feature)
#endif
        for (int environment = 0; environment < environments_; ++environment) {
            const int rank = ranks_[static_cast<size_t>(environment)];
            const int df = rows_ - rank;
            const int64_t gram_offset =
                gram_offsets_[static_cast<size_t>(environment)];
            double local_max_x = 0.0;
            double local_max_w = 0.0;
            for (int variant = 0; variant < variants; ++variant) {
                const double* scalars = scalar
                    + static_cast<size_t>(variant)
                        * static_cast<size_t>(scalar_rows);
                const double* moments = projected
                    + static_cast<size_t>(variant)
                        * static_cast<size_t>(feature_columns_);
                const double s0 = scalars[0];
                const double s1 = scalars[1 + 3 * environment];
                const double s2 = scalars[2 + 3 * environment];
                const double s4 = scalars[3 + 3 * environment];
                long double u00 = 0.0L, u11 = 0.0L, u01 = 0.0L;
                long double u0u2 = 0.0L, u1u3 = 0.0L;
                long double u0e2u0 = 0.0L, u1e2u1 = 0.0L;
                long double leak_x_sq = 0.0L, leak_w_sq = 0.0L;
                for (int a = 0; a < rank; ++a) {
                    const double u0 = feature_moment(moments, environment, 0, a);
                    const double u1 = feature_moment(moments, environment, 1, a);
                    const double u2 = feature_moment(moments, environment, 2, a);
                    const double u3 = feature_moment(moments, environment, 3, a);
                    u00 += static_cast<long double>(u0) * u0;
                    u11 += static_cast<long double>(u1) * u1;
                    u01 += static_cast<long double>(u0) * u1;
                    u0u2 += static_cast<long double>(u0) * u2;
                    u1u3 += static_cast<long double>(u1) * u3;
                    long double e2u0 = 0.0L, e2u1 = 0.0L;
                    long double gu0 = 0.0L, gu1 = 0.0L;
                    for (int b = 0; b < rank; ++b) {
                        const size_t matrix_index =
                            static_cast<size_t>(gram_offset)
                            + static_cast<size_t>(b) * static_cast<size_t>(rank)
                            + static_cast<size_t>(a);
                        const double v0 = feature_moment(moments, environment, 0, b);
                        const double v1 = feature_moment(moments, environment, 1, b);
                        gu0 += static_cast<long double>(gram_values_[matrix_index]) * v0;
                        gu1 += static_cast<long double>(gram_values_[matrix_index]) * v1;
                        e2u0 += static_cast<long double>(e2_gram_values_[matrix_index]) * v0;
                        e2u1 += static_cast<long double>(e2_gram_values_[matrix_index]) * v1;
                    }
                    u0e2u0 += static_cast<long double>(u0) * e2u0;
                    u1e2u1 += static_cast<long double>(u1) * e2u1;
                    const long double dx = static_cast<long double>(u0) - gu0;
                    const long double dw = static_cast<long double>(u1) - gu1;
                    leak_x_sq += dx * dx;
                    leak_w_sq += dw * dw;
                }
                const double ssx = s0 - static_cast<double>(u00);
                const double ssw = s2 - static_cast<double>(u11);
                const double varx = ssx / static_cast<double>(df);
                const double varw = ssw / static_cast<double>(df);
                if (!std::isfinite(varx) || !std::isfinite(varw)
                    || varx <= eps_var || varw <= eps_var) {
                    invalid_feature = 1;
                    continue;
                }
                const double sx = standardized ? 1.0 / std::sqrt(varx) : 1.0;
                const double sw = standardized ? 1.0 / std::sqrt(varw) : 1.0;
                const size_t output_index =
                    static_cast<size_t>(environment) * static_cast<size_t>(variants)
                    + static_cast<size_t>(variant);
                scale_x[output_index] = sx;
                scale_w[output_index] = sw;
                norm_x[output_index] = sx * sx * ssx / static_cast<double>(df);
                norm_w[output_index] = sw * sw * ssw / static_cast<double>(df);
                diag_x[output_index] = sx * sx
                    * (s2 - 2.0 * static_cast<double>(u0u2)
                       + static_cast<double>(u0e2u0))
                    / static_cast<double>(df);
                diag_w[output_index] = sw * sw
                    * (s4 - 2.0 * static_cast<double>(u1u3)
                       + static_cast<double>(u1e2u1))
                    / static_cast<double>(df);
                corr_xw[output_index] = sx * sw
                    * (s1 - static_cast<double>(u01))
                    / static_cast<double>(df);
                local_max_x = std::max(
                    local_max_x,
                    std::sqrt(
                        std::max(0.0, static_cast<double>(leak_x_sq))
                        / std::max(ssx, std::numeric_limits<double>::min())
                    )
                );
                local_max_w = std::max(
                    local_max_w,
                    std::sqrt(
                        std::max(0.0, static_cast<double>(leak_w_sq))
                        / std::max(ssw, std::numeric_limits<double>::min())
                    )
                );
                if (!std::isfinite(scale_x[output_index])
                    || !std::isfinite(scale_w[output_index])
                    || !std::isfinite(norm_x[output_index])
                    || !std::isfinite(norm_w[output_index])
                    || !std::isfinite(diag_x[output_index])
                    || !std::isfinite(diag_w[output_index])
                    || !std::isfinite(corr_xw[output_index])) {
                    invalid_feature = 1;
                }
            }
            max_leak_x[environment] = local_max_x;
            max_leak_w[environment] = local_max_w;
        }
        if (invalid_feature != 0) {
            throw std::runtime_error(
                "Multi-environment native projected feature has zero, invalid, or non-finite moments"
            );
        }
    }

    void prepare_packed_feature_rhs() {
        const int scalar_rows = 1 + 3 * environments_;
        packed_feature_rhs_columns_ = checked_blas_dim(
            checked_add(
                static_cast<size_t>(feature_columns_),
                static_cast<size_t>(scalar_rows),
                "packed feature RHS columns"
            ),
            "packed feature RHS columns"
        );
        packed_feature_rhs_.resize(checked_mul(
            static_cast<size_t>(rows_),
            static_cast<size_t>(packed_feature_rhs_columns_),
            "persistent packed feature RHS"
        ));
        packed_feature_rhs_sums_.assign(
            static_cast<size_t>(packed_feature_rhs_columns_), 0.0
        );
        int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid)
#endif
        for (int row = 0; row < rows_; ++row) {
            double* rhs_row = packed_feature_rhs_.data()
                + static_cast<size_t>(row)
                    * static_cast<size_t>(packed_feature_rhs_columns_);
            for (int column = 0; column < feature_columns_; ++column) {
                const double value = feature_basis_[
                    static_cast<size_t>(column) * static_cast<size_t>(rows_)
                    + static_cast<size_t>(row)
                ];
                rhs_row[column] = value;
                invalid |= !std::isfinite(value);
            }
            rhs_row[feature_columns_] = 1.0;
            for (int environment = 0; environment < environments_;
                 ++environment) {
                const double e = environment_values_[
                    static_cast<size_t>(environment)
                        * static_cast<size_t>(rows_)
                    + static_cast<size_t>(row)
                ];
                const double e2 = e * e;
                const int offset = feature_columns_ + 1 + 3 * environment;
                rhs_row[offset] = e;
                rhs_row[offset + 1] = e2;
                rhs_row[offset + 2] = e2 * e2;
                invalid |= !std::isfinite(e) || !std::isfinite(e2)
                    || !std::isfinite(e2 * e2);
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Packed multi-environment feature RHS is non-finite"
            );
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
        for (int column = 0; column < packed_feature_rhs_columns_; ++column) {
            long double total = 0.0L;
            for (int row = 0; row < rows_; ++row) {
                total += packed_feature_rhs_[
                    static_cast<size_t>(row)
                        * static_cast<size_t>(packed_feature_rhs_columns_)
                    + static_cast<size_t>(column)
                ];
            }
            packed_feature_rhs_sums_[static_cast<size_t>(column)] =
                static_cast<double>(total);
        }
    }

    void validate_metadata() {
        if (power_offsets_.front() != 0 ||
            power_offsets_.back() != static_cast<int64_t>(power_indices_.size()) ||
            gram_offsets_.front() != 0 ||
            gram_offsets_.back() != static_cast<int64_t>(gram_values_.size())) {
            throw std::runtime_error(
                "Multi-environment native feature offsets do not span their buffers"
            );
        }
        int64_t previous_power = 0;
        int64_t previous_gram = 0;
        for (int environment = 0; environment < environments_; ++environment) {
            const int rank = ranks_[static_cast<size_t>(environment)];
            if (rank <= 0 || rank >= rows_ || rows_ - rank <= 0) {
                throw std::runtime_error(
                    "Multi-environment native feature rank is invalid"
                );
            }
            for (int power = 0; power < 4; ++power) {
                const size_t offset_index =
                    static_cast<size_t>(4 * environment + power);
                const int64_t begin = power_offsets_[offset_index];
                const int64_t end = power_offsets_[offset_index + 1U];
                if (begin != previous_power || end - begin != rank) {
                    throw std::runtime_error(
                        "Multi-environment native power-index ranges are malformed"
                    );
                }
                for (int64_t index = begin; index < end; ++index) {
                    if (power_indices_[static_cast<size_t>(index)] < 0 ||
                        power_indices_[static_cast<size_t>(index)] >= feature_columns_) {
                        throw std::runtime_error(
                            "Multi-environment native power index is out of range"
                        );
                    }
                }
                previous_power = end;
            }
            const int64_t gram_begin = gram_offsets_[
                static_cast<size_t>(environment)
            ];
            const int64_t gram_end = gram_offsets_[
                static_cast<size_t>(environment + 1)
            ];
            if (gram_begin != previous_gram ||
                gram_end - gram_begin != static_cast<int64_t>(rank) * rank) {
                throw std::runtime_error(
                    "Multi-environment native Gram ranges are malformed"
                );
            }
            previous_gram = gram_end;
        }
        for (double value : environment_values_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Multi-environment native environment contains NaN or infinity"
                );
            }
        }
        for (double value : feature_basis_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Multi-environment native feature basis contains NaN or infinity"
                );
            }
        }
        for (double value : common_basis_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Multi-environment native common basis contains NaN or infinity"
                );
            }
        }
        for (double value : directions_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Multi-environment native direction contains NaN or infinity"
                );
            }
        }
        for (double value : gram_values_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Multi-environment native Gram matrix contains NaN or infinity"
                );
            }
        }
        for (double value : e2_gram_values_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Multi-environment native weighted Gram matrix contains NaN or infinity"
                );
            }
        }
    }

    void validate_genotype(nb_mat2f_ro<double> genotype,
                           const char* label) const {
        if (checked_blas_dim(genotype.shape(0), label) != rows_ ||
            checked_blas_dim(genotype.shape(1), label) <= 0) {
            throw std::runtime_error(
                "Multi-environment native genotype block has incompatible dimensions"
            );
        }
    }

    double feature_moment(const double* projected_column,
                          int environment,
                          int power,
                          int rank_index) const {
        const size_t range = static_cast<size_t>(4 * environment + power);
        const int64_t begin = power_offsets_[range];
        const int64_t basis_index = power_indices_[
            static_cast<size_t>(begin + rank_index)
        ];
        return projected_column[static_cast<size_t>(basis_index)];
    }

    void record_repairs(int64_t repaired) const {
        if (repaired > 0) {
            repaired_columns_.fetch_add(repaired, std::memory_order_relaxed);
        }
    }

    void record_integrity_resolution(
        const GemmIntegrityResolution& resolution
    ) const {
        record_repairs(resolution.materially_repaired_columns);
        if (resolution.checksum_recomputed_columns > 0) {
            checksum_recomputed_columns_.fetch_add(
                resolution.checksum_recomputed_columns,
                std::memory_order_relaxed
            );
        }
        const int64_t roundoff = resolution.roundoff_only_columns();
        if (roundoff > 0) {
            roundoff_only_columns_.fetch_add(
                roundoff, std::memory_order_relaxed
            );
        }
    }

    int rows_ = 0;
    int environments_ = 0;
    int feature_columns_ = 0;
    int common_rank_ = 0;
    int ddof_ = 1;
    int annotation_bins_ = 0;
    int threads_ = 1;
    std::vector<double> environment_values_;
    std::vector<double> feature_basis_;
    std::vector<double> common_basis_;
    std::vector<double> directions_;
    std::vector<int64_t> power_indices_;
    std::vector<int64_t> power_offsets_;
    std::vector<int32_t> ranks_;
    std::vector<double> gram_values_;
    std::vector<int64_t> gram_offsets_;
    std::vector<double> e2_gram_values_;
    int packed_feature_rhs_columns_ = 0;
    std::vector<double> packed_feature_rhs_;
    std::vector<double> packed_feature_rhs_sums_;
    // One maximum-capacity allocation per execution-scratch role.  In the
    // frozen direct-context plan the capacities below are configured before
    // the first call and any larger request fails closed; a standalone
    // kernel (no frozen plan) grows the single allocation instead, never
    // retaining a second mapping for another exact shape.
    mutable std::unique_ptr<NativeGemmOutputAllocation>
        source_output_scratch_;
    mutable std::unique_ptr<NativeGemmOutputAllocation>
        target_output_scratch_;
    mutable std::vector<double> source_weights_scratch_;
    mutable std::vector<double> source_sqrt_annotation_scratch_;
    mutable std::vector<double> feature_projected_scratch_;
    mutable std::vector<double> feature_scalar_scratch_;
    mutable std::vector<double> target_mailman_output_scratch_;
    size_t source_output_capacity_elements_ = 0;
    size_t target_output_capacity_elements_ = 0;
    size_t source_weights_capacity_elements_ = 0;
    size_t source_annotation_capacity_elements_ = 0;
    size_t feature_projected_capacity_elements_ = 0;
    size_t feature_scalar_capacity_elements_ = 0;
    size_t target_mailman_capacity_elements_ = 0;
    bool scratch_capacities_frozen_ = false;
    mutable bool execution_scratch_released_ = false;

    struct ScratchRoleTelemetry {
        int64_t allocations = 0;
        int64_t reuses = 0;
        size_t high_water_rows = 0;
        size_t high_water_columns = 0;
        size_t capacity_bytes = 0;
        size_t released_bytes = 0;
    };

    // Context-owned per-worker Mailman scratch: one arena per OpenMP worker
    // slot, replacing persistent function-local thread_local vectors that
    // were invisible to the planner and never releasable.  ``segment_a``
    // serves the plain segment output (source/target kernels) and the linear
    // half of the feature kernel; ``segment_b`` is the squared half.
    struct MailmanWorkerArena {
        std::vector<double> work_table;
        std::vector<double> segment_a;
        std::vector<double> segment_b;
        // The post-multiply kernel leaves table cells populated; the
        // pre-multiply kernels require an all-zero table on entry.
        bool table_dirty = false;
    };
    mutable std::vector<MailmanWorkerArena> mailman_worker_arenas_;
    mutable ScratchRoleTelemetry mailman_worker_role_;
    bool mailman_plan_frozen_ = false;
    int mailman_frozen_segment_size_ = 0;
    int64_t mailman_frozen_table_size_ = 0;
    int mailman_qpanel_feature_ = 0;
    int mailman_qpanel_source_ = 0;
    int mailman_qpanel_target_ = 0;
    size_t mailman_worker_table_capacity_ = 0;
    size_t mailman_worker_segment_a_capacity_ = 0;
    size_t mailman_worker_segment_b_capacity_ = 0;

    MailmanWorkerArena& mailman_worker_arena(
        size_t table_elements,
        size_t segment_a_elements,
        size_t segment_b_elements,
        bool needs_zeroed_table
    ) const {
#ifdef _OPENMP
        const int worker = omp_get_thread_num();
        if (omp_get_num_threads() > threads_) {
            throw std::runtime_error(
                "Mailman worker team exceeds the planned worker count"
            );
        }
#else
        const int worker = 0;
#endif
        if (mailman_worker_arenas_.size()
                != static_cast<size_t>(threads_)) {
            throw std::runtime_error(
                "Mailman worker arenas were not prepared for this team"
            );
        }
        MailmanWorkerArena& arena =
            mailman_worker_arenas_[static_cast<size_t>(worker)];
        if (mailman_plan_frozen_
            && (table_elements > mailman_worker_table_capacity_
                || segment_a_elements > mailman_worker_segment_a_capacity_
                || segment_b_elements > mailman_worker_segment_b_capacity_)) {
            throw std::runtime_error(
                "Mailman worker scratch exceeds its admitted frozen capacity"
            );
        }
        const size_t table_target = mailman_plan_frozen_
            ? mailman_worker_table_capacity_ : table_elements;
        const size_t a_target = mailman_plan_frozen_
            ? mailman_worker_segment_a_capacity_ : segment_a_elements;
        const size_t b_target = mailman_plan_frozen_
            ? mailman_worker_segment_b_capacity_ : segment_b_elements;
        if (arena.work_table.size() < table_target) {
            arena.work_table.assign(table_target, 0.0);
            arena.table_dirty = false;
        }
        if (arena.segment_a.size() < a_target) {
            arena.segment_a.resize(a_target);
        }
        if (arena.segment_b.size() < b_target) {
            arena.segment_b.resize(b_target);
        }
        if (needs_zeroed_table && arena.table_dirty) {
            std::fill(arena.work_table.begin(), arena.work_table.end(), 0.0);
            arena.table_dirty = false;
        }
        return arena;
    }

    size_t mailman_worker_arena_bytes() const {
        size_t total = 0;
        for (const MailmanWorkerArena& arena : mailman_worker_arenas_) {
            total += (arena.work_table.capacity()
                      + arena.segment_a.capacity()
                      + arena.segment_b.capacity()) * sizeof(double);
        }
        return total;
    }

    nb::dict mailman_worker_role_dict() const {
        int64_t allocated_workers = 0;
        size_t high_water = 0;
        for (const MailmanWorkerArena& arena : mailman_worker_arenas_) {
            const size_t bytes = (arena.work_table.capacity()
                                  + arena.segment_a.capacity()
                                  + arena.segment_b.capacity())
                * sizeof(double);
            if (bytes > 0) ++allocated_workers;
            high_water = std::max(high_water, bytes);
        }
        nb::dict result;
        result["allocations"] = allocated_workers;
        result["reuses"] = mailman_worker_role_.reuses;
        result["high_water_rows"] = high_water;
        result["high_water_columns"] = 1;
        result["capacity_bytes"] = mailman_worker_arena_bytes();
        result["released_bytes"] = mailman_worker_role_.released_bytes;
        return result;
    }
    mutable ScratchRoleTelemetry source_output_role_;
    mutable ScratchRoleTelemetry target_output_role_;
    mutable ScratchRoleTelemetry source_weights_role_;
    mutable ScratchRoleTelemetry source_annotation_role_;
    mutable ScratchRoleTelemetry feature_projected_role_;
    mutable ScratchRoleTelemetry feature_scalar_role_;
    mutable ScratchRoleTelemetry target_mailman_role_;

    void require_frozen_mailman_geometry(
        const MailmanPackedBlock& packed
    ) const {
        if (!mailman_plan_frozen_) return;
        if (packed.segment_size != mailman_frozen_segment_size_
            || packed.table_size != mailman_frozen_table_size_) {
            throw std::runtime_error(
                "Packed Mailman block geometry disagrees with the frozen "
                "plan; an unmodeled SUMMIT_MAILMAN_* override cannot alter "
                "a protected execution"
            );
        }
    }

    size_t admitted_allocation_capacity(
        size_t capacity_elements,
        const char* description
    ) const {
        if (scratch_capacities_frozen_ && capacity_elements == 0) {
            throw std::runtime_error(
                std::string(description)
                + " has no admitted capacity in the frozen execution plan"
            );
        }
        return capacity_elements;
    }

    void require_execution_scratch_live(const char* role) const {
        if (execution_scratch_released_) {
            throw std::runtime_error(
                std::string(role)
                + " was requested after execution scratch release"
            );
        }
    }

    static void record_scratch_role_use(
        ScratchRoleTelemetry& role,
        bool allocated,
        size_t rows,
        size_t columns,
        size_t capacity_bytes
    ) {
        if (allocated) {
            ++role.allocations;
        } else {
            ++role.reuses;
        }
        role.high_water_rows = std::max(role.high_water_rows, rows);
        role.high_water_columns = std::max(role.high_water_columns, columns);
        role.capacity_bytes = std::max(role.capacity_bytes, capacity_bytes);
    }

    // Vector-backed scratch roles share one growth policy: with a frozen
    // capacity the single reservation happens once and any larger request
    // fails closed; without one (standalone kernel) the vector may grow but
    // never holds a second allocation for another exact shape.
    double* prepare_vector_scratch(
        std::vector<double>& scratch,
        ScratchRoleTelemetry& role,
        size_t elements,
        size_t frozen_capacity_elements,
        const char* description
    ) const {
        require_execution_scratch_live(description);
        if (scratch_capacities_frozen_) {
            if (elements > frozen_capacity_elements) {
                throw std::runtime_error(
                    std::string(description)
                    + " exceeds its admitted frozen capacity"
                );
            }
            if (scratch.capacity() < frozen_capacity_elements) {
                scratch.reserve(frozen_capacity_elements);
                ++role.allocations;
            } else {
                ++role.reuses;
            }
        } else {
            if (scratch.capacity() < elements) {
                ++role.allocations;
            } else {
                ++role.reuses;
            }
        }
        if (scratch.size() < elements) {
            scratch.resize(elements);
        }
        role.high_water_rows = std::max(role.high_water_rows, elements);
        role.high_water_columns = 1;
        role.capacity_bytes = std::max(
            role.capacity_bytes, scratch.capacity() * sizeof(double)
        );
        return scratch.data();
    }

    static nb::dict scratch_role_to_dict(const ScratchRoleTelemetry& role) {
        nb::dict result;
        result["allocations"] = role.allocations;
        result["reuses"] = role.reuses;
        result["high_water_rows"] = role.high_water_rows;
        result["high_water_columns"] = role.high_water_columns;
        result["capacity_bytes"] = role.capacity_bytes;
        result["released_bytes"] = role.released_bytes;
        return result;
    }
    mutable std::mutex mutex_;
    mutable std::atomic<int64_t> feature_calls_{0};
    mutable std::atomic<int64_t> packed_feature_calls_{0};
    mutable std::atomic<int64_t> source_calls_{0};
    mutable std::atomic<int64_t> packed_source_calls_{0};
    mutable std::atomic<int64_t> projection_calls_{0};
    mutable std::atomic<int64_t> target_calls_{0};
    mutable std::atomic<int64_t> packed_target_calls_{0};
    mutable std::atomic<int64_t> normalization_calls_{0};
    mutable std::atomic<int64_t> repaired_columns_{0};
    mutable std::atomic<int64_t> checksum_recomputed_columns_{0};
    mutable std::atomic<int64_t> roundoff_only_columns_{0};
    mutable std::atomic<int64_t> scratch_output_allocations_{0};
    mutable std::atomic<int64_t> scratch_output_reuses_{0};
};

class MultiEnvironmentDirectContext {
public:
    MultiEnvironmentDirectContext(
        int bed_descriptor,
        int bim_descriptor,
        int fam_descriptor,
        nb::object row_sel,
        nb_vec1_ro<double> reader_environment,
        nb_mat2f_ro<double> reader_q_basis,
        int decode_threads,
        uint64_t max_workspace_bytes,
        nb_mat2f_ro<double> environments,
        nb_mat2f_ro<double> feature_basis,
        nb_vec1_ro<int64_t> feature_power_indices,
        nb_vec1_ro<int64_t> feature_power_offsets,
        nb_vec1_ro<int32_t> feature_ranks,
        nb_vec1_ro<double> feature_gram_values,
        nb_vec1_ro<int64_t> feature_gram_offsets,
        nb_vec1_ro<double> feature_e2_gram_values,
        nb_mat2f_ro<double> common_basis,
        nb_mat2f_ro<double> directions,
        nb_mat2c_ro<double> annotations,
        nb_vec1_ro<double> annotation_masses,
        nb_mat2c_ro<int64_t> blocks,
        nb_mat2c_ro<int64_t> environment_tiles,
        nb_mat2c_ro<int64_t> probe_tiles,
        nb_mat2c_ro<uint64_t> philox_keys,
        int ddof,
        int total_probes,
        double eps_var,
        bool standardized,
        bool dense_blas_hybrid,
        int requested_threads
    )
        : reader_(std::make_unique<DirectContext>(
              bed_descriptor, bim_descriptor, fam_descriptor,
              std::move(row_sel), ddof, reader_environment, reader_q_basis,
              decode_threads, max_workspace_bytes, 1, false,
              requested_threads
          )),
          kernel_(
              environments, feature_basis, feature_power_indices,
              feature_power_offsets, feature_ranks, feature_gram_values,
              feature_gram_offsets, feature_e2_gram_values, common_basis,
              directions, ddof,
              checked_blas_dim(annotations.shape(1),
                               "direct multi-environment annotation bins"),
              requested_threads
          ),
          rows_(reader_->n_),
          variants_(reader_->m_total_),
          environments_(checked_blas_dim(
              environments.shape(1), "direct multi-environment count"
          )),
          annotation_bins_(checked_blas_dim(
              annotations.shape(1), "direct multi-environment annotation bins"
          )),
          feature_columns_(checked_blas_dim(
              feature_basis.shape(1),
              "direct multi-environment feature columns"
          )),
          common_rank_(checked_blas_dim(
              common_basis.shape(1), "direct multi-environment common rank"
          )),
          total_probes_(total_probes),
          eps_var_(eps_var),
          standardized_(standardized),
          dense_blas_hybrid_(dense_blas_hybrid),
          threads_(requested_threads),
          decode_threads_(decode_threads) {
        numa_request_ = native_numa_contract_request();
        if (rows_ != checked_blas_dim(
                environments.shape(0), "direct multi-environment rows") ||
            variants_ != checked_blas_dim(
                annotations.shape(0), "direct multi-environment variants") ||
            annotation_bins_ <= 0 || environments_ <= 0 ||
            total_probes_ <= 0 || !(eps_var_ > 0.0) ||
            !std::isfinite(eps_var_)) {
            throw std::runtime_error(
                "Direct multi-environment execution metadata is inconsistent"
            );
        }
        if (annotation_masses.shape(0) !=
                static_cast<size_t>(annotation_bins_)) {
            throw std::runtime_error(
                "Direct multi-environment annotation masses are mis-sized"
            );
        }
        annotation_values_.assign(
            annotations.data(),
            annotations.data() + checked_mul(
                static_cast<size_t>(variants_),
                static_cast<size_t>(annotation_bins_),
                "direct multi-environment annotations"
            )
        );
        annotation_masses_.assign(
            annotation_masses.data(),
            annotation_masses.data() + annotation_masses.shape(0)
        );
        environment_values_.assign(
            environments.data(),
            environments.data() + checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(environments_),
                "direct multi-environment values"
            )
        );
        environment_target_norms_.assign(
            static_cast<size_t>(environments_), 0.0
        );
        for (int environment = 0; environment < environments_; ++environment) {
            long double squared_norm = 0.0L;
            for (int row = 0; row < rows_; ++row) {
                const double value = environment_values_[
                    static_cast<size_t>(environment)
                        * static_cast<size_t>(rows_)
                    + static_cast<size_t>(row)
                ];
                squared_norm += static_cast<long double>(value) * value;
            }
            environment_target_norms_[static_cast<size_t>(environment)] =
                std::sqrt(static_cast<double>(squared_norm));
        }
        parse_ranges(blocks, variants_, "genotype block", blocks_);
        parse_ranges(
            environment_tiles, environments_, "environment tile",
            environment_tiles_
        );
        parse_probe_tiles(probe_tiles);
        const size_t expected_key_rows = checked_mul(
            blocks_.size(), static_cast<size_t>(total_probes_),
            "direct multi-environment Philox keys"
        );
        if (philox_keys.shape(0) != expected_key_rows ||
            philox_keys.shape(1) != 2) {
            throw std::runtime_error(
                "Direct multi-environment Philox key table is mis-sized"
            );
        }
        philox_keys_.assign(
            philox_keys.data(),
            philox_keys.data() + checked_mul(
                expected_key_rows, 2U,
                "direct multi-environment Philox keys"
            )
        );
        for (double value : annotation_values_) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error(
                    "Direct multi-environment annotations must be finite and nonnegative"
                );
            }
        }
        for (double value : annotation_masses_) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::runtime_error(
                    "Direct multi-environment annotation masses must be positive"
                );
            }
        }
        sqrt_annotation_values_.resize(annotation_values_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
        for (int64_t index = 0;
             index < static_cast<int64_t>(annotation_values_.size());
             ++index) {
            sqrt_annotation_values_[static_cast<size_t>(index)] =
                std::sqrt(annotation_values_[static_cast<size_t>(index)]);
        }
        // The supplied masses stay numerically authoritative (they are the
        // published binary64 pairwise sums), but they must be consistent
        // with the copied canonical annotation matrix: verify each against
        // a long-double column sum under a documented tolerance of
        // 64*eps*(long-double magnitude sum) + DBL_MIN, which admits only
        // summation-order differences.  A direct caller can therefore no
        // longer pair valid annotations with unrelated masses.
        for (int bin = 0; bin < annotation_bins_; ++bin) {
            long double derived = 0.0L;
            long double magnitude = 0.0L;
            for (int64_t variant = 0; variant < variants_; ++variant) {
                const long double value = annotation_values_[
                    static_cast<size_t>(variant)
                        * static_cast<size_t>(annotation_bins_)
                    + static_cast<size_t>(bin)
                ];
                derived += value;
                magnitude += value < 0.0L ? -value : value;
            }
            const long double supplied = annotation_masses_[
                static_cast<size_t>(bin)
            ];
            const long double tolerance = 64.0L
                * static_cast<long double>(
                    std::numeric_limits<double>::epsilon()
                ) * magnitude
                + static_cast<long double>(
                    std::numeric_limits<double>::min()
                );
            const long double difference = supplied > derived
                ? supplied - derived : derived - supplied;
            if (difference > tolerance) {
                throw std::runtime_error(
                    "Direct multi-environment annotation masses are "
                    "inconsistent with the copied annotation matrix"
                );
            }
        }
        const size_t tile_products = checked_mul(
            environment_tiles_.size(), probe_tiles_.size(),
            "direct multi-environment tile products"
        );
        fused_two_pass_execution_ = environment_tiles_.size() == 1U
            && requested_probe_tile_count_ == 1U;
        if (!dense_blas_hybrid_
            && total_probes_ > kMultiEnvironmentMailmanMaximumProbes) {
            throw std::runtime_error(
                "Packed Mailman GxE execution is restricted to at most 10 probes"
            );
        }
        planned_genotype_passes_ = fused_two_pass_execution_
            ? 2U
            : checked_add(
                1U,
                checked_mul(2U, tile_products,
                            "direct multi-environment genotype passes"),
                "direct multi-environment genotype passes"
            );
        // The packed fallback allocates an output only for common-covariate
        // projection. The dense hybrid allocates one feature and one source
        // output per block, one projection output per tile, and one combined
        // [S,e*S] target output per block/tile.
        const size_t output_calls = dense_blas_hybrid_
            ? checked_add(
                checked_mul(
                    blocks_.size(),
                    checked_add(
                        1U,
                        checked_mul(2U, tile_products,
                                    "dense direct output calls"),
                        "dense direct output calls"
                    ),
                    "dense direct output calls"
                ),
                common_rank_ > 0 ? tile_products : 0U,
                "dense direct output calls"
            )
            : (common_rank_ > 0 ? tile_products : 0U);
        planned_output_calls_ = output_calls;
        if (planned_output_calls_ > kNativeGemmOutputEvidenceCapacity ||
            planned_output_calls_ > kGemmTelemetryCapacity) {
            throw std::runtime_error(
                "Direct multi-environment plan exceeds native evidence capacity"
            );
        }
        // One feature record per block, one source and one target record
        // per block per tile product, and up to two projection records
        // (projection + normal-equation update) per tile product.
        planned_semantic_call_maximum_ = checked_add(
            checked_mul(
                blocks_.size(),
                checked_add(
                    1U,
                    checked_mul(2U, tile_products,
                                "direct semantic record maximum"),
                    "direct semantic record maximum"
                ),
                "direct semantic record maximum"
            ),
            checked_mul(2U, tile_products, "direct semantic record maximum"),
            "direct semantic record maximum"
        );

        // Freeze one maximum-capacity allocation per execution-scratch role
        // from the admitted plan.  These formulas must match the Python
        // complete-process memory planner exactly; the frozen-plan info()
        // reconciliation fails closed on any disagreement.
        int max_block_width = 0;
        for (const auto& [start, stop] : blocks_) {
            max_block_width = std::max(max_block_width, stop - start);
        }
        int max_environment_tile = 0;
        for (const auto& [start, stop] : environment_tiles_) {
            max_environment_tile = std::max(max_environment_tile, stop - start);
        }
        max_genotype_block_width_ = max_block_width;
        const size_t wide_max = checked_mul(
            checked_mul(
                2U * static_cast<size_t>(max_environment_tile),
                static_cast<size_t>(annotation_bins_),
                "direct multi-environment wide capacity"
            ),
            static_cast<size_t>(maximum_execution_probe_chunk_width_),
            "direct multi-environment wide capacity"
        );
        decoded_scratch_capacity_bytes_ = dense_blas_hybrid_
            ? checked_mul(
                checked_mul(static_cast<size_t>(rows_),
                            static_cast<size_t>(max_block_width),
                            "direct decoded scratch capacity"),
                sizeof(double), "direct decoded scratch capacity"
            )
            : 0U;
        const size_t source_output_elements = dense_blas_hybrid_
            ? checked_mul(static_cast<size_t>(rows_), wide_max,
                          "direct source output capacity")
            : 0U;
        const size_t target_output_elements = dense_blas_hybrid_
            ? checked_mul(
                static_cast<size_t>(max_block_width),
                checked_mul(2U, wide_max, "direct target output capacity"),
                "direct target output capacity"
            )
            : 0U;
        source_output_scratch_capacity_bytes_ =
            source_output_elements * sizeof(double);
        target_output_scratch_capacity_bytes_ =
            target_output_elements * sizeof(double);
        kernel_.configure_scratch_capacities(
            source_output_elements,
            target_output_elements,
            checked_mul(static_cast<size_t>(max_block_width), wide_max,
                        "direct source weight capacity"),
            checked_mul(static_cast<size_t>(max_block_width),
                        static_cast<size_t>(annotation_bins_),
                        "direct source annotation capacity"),
            checked_mul(static_cast<size_t>(max_block_width),
                        static_cast<size_t>(feature_columns_),
                        "direct feature projection capacity"),
            checked_mul(static_cast<size_t>(max_block_width),
                        static_cast<size_t>(1 + 3 * environments_),
                        "direct feature scalar capacity"),
            dense_blas_hybrid_
                ? 0U
                : checked_mul(
                    static_cast<size_t>(max_block_width),
                    checked_mul(2U, wide_max,
                                "direct packed target output capacity"),
                    "direct packed target output capacity"
                )
        );

        // Freeze the Mailman execution configuration once: segment size and
        // q-panel widths come from the environment as read here, are
        // mirrored by the Python planner, and are enforced against every
        // packed block so an unmodeled override can never change a
        // protected execution.  A dense context freezes zero worker
        // capacity so any packed call fails closed.
        const int frozen_segment =
            compute_mailman_segment_size_optimized(rows_);
        const int64_t frozen_table =
            compute_mailman_table_size(frozen_segment);
        const int feature_rhs_columns = kernel_.packed_feature_rhs_columns();
        const int wide_max_int = checked_blas_dim(
            wide_max, "direct Mailman wide capacity"
        );
        const int target_max_int = checked_blas_dim(
            checked_mul(2U, wide_max, "direct Mailman target capacity"),
            "direct Mailman target capacity"
        );
        const int qpanel_feature = summit::mailman::qpanel_width<double>(
            frozen_table, feature_rhs_columns, frozen_segment, 2
        );
        const int qpanel_source = summit::mailman::qpanel_width<double>(
            frozen_table, wide_max_int, frozen_segment
        );
        const int qpanel_target = summit::mailman::qpanel_width<double>(
            frozen_table, target_max_int, frozen_segment
        );
        size_t worker_table_capacity = 0;
        size_t worker_segment_a_capacity = 0;
        size_t worker_segment_b_capacity = 0;
        if (!dense_blas_hybrid_) {
            const size_t feature_panel = static_cast<size_t>(
                std::min(qpanel_feature, feature_rhs_columns)
            );
            const size_t source_panel = static_cast<size_t>(
                std::min(qpanel_source, wide_max_int)
            );
            const size_t target_panel = static_cast<size_t>(
                std::min(qpanel_target, target_max_int)
            );
            worker_table_capacity = checked_mul(
                static_cast<size_t>(frozen_table),
                std::max(feature_panel,
                         std::max(source_panel, target_panel)),
                "direct Mailman worker table capacity"
            );
            worker_segment_a_capacity = checked_mul(
                static_cast<size_t>(frozen_segment),
                std::max(feature_panel, target_panel),
                "direct Mailman worker segment capacity"
            );
            worker_segment_b_capacity = checked_mul(
                static_cast<size_t>(frozen_segment), feature_panel,
                "direct Mailman worker segment capacity"
            );
        }
        kernel_.configure_mailman_plan(
            frozen_segment, frozen_table,
            qpanel_feature, qpanel_source, qpanel_target,
            worker_table_capacity,
            worker_segment_a_capacity,
            worker_segment_b_capacity
        );
        mailman_frozen_segment_size_ = frozen_segment;
        mailman_frozen_table_size_ = frozen_table;
        mailman_qpanel_feature_ = qpanel_feature;
        mailman_qpanel_source_ = qpanel_source;
        mailman_qpanel_target_ = qpanel_target;
        mailman_worker_scratch_capacity_bytes_per_worker_ =
            (worker_table_capacity + worker_segment_a_capacity
             + worker_segment_b_capacity) * sizeof(double);
    }

    MultiEnvironmentDirectContext(const MultiEnvironmentDirectContext&) = delete;
    MultiEnvironmentDirectContext& operator=(
        const MultiEnvironmentDirectContext&
    ) = delete;

    nb::dict info() const {
        std::lock_guard<std::mutex> guard(mutex_);
        nb::dict result;
        result["schema"] = "summit.multi_environment_direct_context.v3";
        result["schema_version"] = 1;
        result["rows"] = rows_;
        result["variants"] = variants_;
        result["environment_count"] = environments_;
        result["annotation_bins"] = annotation_bins_;
        result["probe_count"] = total_probes_;
        result["mailman_maximum_probe_count"] =
            kMultiEnvironmentMailmanMaximumProbes;
        result["mailman_probe_count_eligible"] =
            total_probes_ <= kMultiEnvironmentMailmanMaximumProbes;
        result["block_count"] = blocks_.size();
        result["environment_tile_count"] = environment_tiles_.size();
        result["probe_tile_count"] = requested_probe_tile_count_;
        result["execution_probe_chunk_count"] = probe_tiles_.size();
        result["maximum_execution_probe_chunk_width"] =
            maximum_execution_probe_chunk_width_;
        result["fused_two_pass_execution"] = fused_two_pass_execution_;
        result["planned_output_calls"] = planned_output_calls_;
        result["planned_semantic_call_maximum"] =
            planned_semantic_call_maximum_;
        result["planned_genotype_passes"] = planned_genotype_passes_;
        result["decode_threads"] = decode_threads_;
        result["threads"] = threads_;
        result["descriptor_owned_bed"] = true;
        result["native_probe_generation"] = true;
        result["execution_kernel"] = dense_blas_hybrid_
            ? "dense_private_blas_streamed_pair"
            : "packed_mailman_low_memory_fallback";
        result["packed_genotype_mailman"] = !dense_blas_hybrid_;
        result["packed_feature_moments"] = !dense_blas_hybrid_;
        result["packed_source_direct_accumulation"] = !dense_blas_hybrid_;
        result["virtual_environment_weighted_target_rhs"] =
            !dense_blas_hybrid_;
        result["materialized_target_rhs"] = dense_blas_hybrid_;
        result["dense_genotype_feature_decode"] = dense_blas_hybrid_;
        result["dense_genotype_target_decode"] = dense_blas_hybrid_;
        result["persistent_dense_decode_scratch"] = dense_blas_hybrid_;
        result["persistent_environment_weighted_target_panel"] =
            dense_blas_hybrid_;
        result["source_panel_released_before_target"] = false;
        result["source_panel_reused_as_protected_pair_first_half"] = false;
        result["contracted_numa_decode"] =
            dense_blas_hybrid_ && numa_request_.required;
        result["contracted_packed_source_panel_numa"] =
            numa_request_.required;
        result["max_genotype_block_width"] = max_genotype_block_width_;
        result["mailman_frozen_segment_size"] = mailman_frozen_segment_size_;
        result["mailman_frozen_table_size"] = mailman_frozen_table_size_;
        result["mailman_qpanel_feature"] = mailman_qpanel_feature_;
        result["mailman_qpanel_source"] = mailman_qpanel_source_;
        result["mailman_qpanel_target"] = mailman_qpanel_target_;
        result["mailman_worker_scratch_capacity_bytes_per_worker"] =
            mailman_worker_scratch_capacity_bytes_per_worker_;
        result["decoded_scratch_capacity_bytes"] =
            decoded_scratch_capacity_bytes_;
        result["source_output_scratch_capacity_bytes"] =
            source_output_scratch_capacity_bytes_;
        result["target_output_scratch_capacity_bytes"] =
            target_output_scratch_capacity_bytes_;
        result["single_use"] = true;
        result["completed"] = completed_;
        result["execution_scratch_released"] = execution_scratch_released_;
        result["decoded_scratch_released_bytes"] =
            decoded_scratch_released_bytes_;
        return result;
    }

    // Release every execution-only native mapping after the single-use run
    // has completed and its results were detached into separately owned
    // arrays.  Idempotent and safe during ordinary exception cleanup; any
    // later scratch request fails clearly.
    nb::dict release_execution_scratch() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!completed_) {
            throw std::runtime_error(
                "Execution scratch can be released only after the "
                "single-use direct context completed its run"
            );
        }
        const size_t live_before = kernel_.live_scratch_capacity_bytes()
            + (dense_decode_state_.bound_storage != nullptr
               || dense_decode_state_.legacy_storage != nullptr
               ? decoded_scratch_capacity_bytes_ : 0U);
        if (dense_decode_state_.bound_storage != nullptr) {
            decoded_scratch_released_bytes_ = std::max(
                decoded_scratch_released_bytes_,
                dense_decode_state_.bound_storage->capacity_byte_count()
            );
        } else if (dense_decode_state_.legacy_storage != nullptr) {
            decoded_scratch_released_bytes_ = std::max(
                decoded_scratch_released_bytes_,
                decoded_scratch_capacity_bytes_
            );
        }
        dense_decode_state_.legacy_storage.reset();
        dense_decode_state_.bound_storage.reset();
        dense_decode_state_.evidence.reset();
        std::vector<int>().swap(dense_decode_state_.observed);
        kernel_.release_execution_scratch();
        execution_scratch_released_ = true;
        nb::dict evidence;
        evidence["released"] = true;
        evidence["live_scratch_capacity_bytes_before"] = live_before;
        evidence["live_scratch_capacity_bytes_after"] =
            kernel_.live_scratch_capacity_bytes();
        evidence["decoded_scratch_released_bytes"] =
            decoded_scratch_released_bytes_;
        evidence["kernel_scratch_released_bytes"] =
            kernel_.released_scratch_bytes();
        return evidence;
    }

    nb::dict run() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (completed_) {
            throw std::runtime_error(
                "Direct multi-environment context is single-use"
            );
        }
        reader_->ensure_open();
        reader_->check_files_unchanged();

        double* scale_x = nullptr;
        double* scale_w = nullptr;
        double* norm_x = nullptr;
        double* norm_w = nullptr;
        double* diag_x = nullptr;
        double* diag_w = nullptr;
        double* corr_xw = nullptr;
        auto scale_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &scale_x
        );
        auto scale_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &scale_w
        );
        auto norm_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &norm_x
        );
        auto norm_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &norm_w
        );
        auto diag_x_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &diag_x
        );
        auto diag_w_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &diag_w
        );
        auto corr_xw_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &corr_xw
        );
        std::vector<double> max_leak_x(static_cast<size_t>(environments_), 0.0);
        std::vector<double> max_leak_w(static_cast<size_t>(environments_), 0.0);
        std::vector<int64_t> missing_counts(
            static_cast<size_t>(variants_), 0
        );
        double* missing_correlations = nullptr;
        auto missing_correlations_out = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(variants_),
            static_cast<size_t>(environments_), &missing_correlations
        );
        std::fill(
            missing_correlations,
            missing_correlations
                + checked_mul(
                    static_cast<size_t>(variants_),
                    static_cast<size_t>(environments_),
                    "direct missingness correlations"
                ),
            0.0
        );

        const size_t accumulator_rows = checked_mul(
            static_cast<size_t>(environments_),
            static_cast<size_t>(variants_),
            "direct multi-environment accumulator rows"
        );
        double* accum_xx = nullptr;
        double* accum_xw = nullptr;
        double* accum_wx = nullptr;
        double* accum_ww = nullptr;
        auto accum_xx_out = make_owned_numpy_mat2c<double>(
            accumulator_rows, static_cast<size_t>(annotation_bins_), &accum_xx
        );
        auto accum_xw_out = make_owned_numpy_mat2c<double>(
            accumulator_rows, static_cast<size_t>(annotation_bins_), &accum_xw
        );
        auto accum_wx_out = make_owned_numpy_mat2c<double>(
            accumulator_rows, static_cast<size_t>(annotation_bins_), &accum_wx
        );
        auto accum_ww_out = make_owned_numpy_mat2c<double>(
            accumulator_rows, static_cast<size_t>(annotation_bins_), &accum_ww
        );
        const size_t accumulator_elements = checked_mul(
            accumulator_rows, static_cast<size_t>(annotation_bins_),
            "direct multi-environment accumulators"
        );
        std::fill(accum_xx, accum_xx + accumulator_elements, 0.0);
        std::fill(accum_xw, accum_xw + accumulator_elements, 0.0);
        std::fill(accum_wx, accum_wx + accumulator_elements, 0.0);
        std::fill(accum_ww, accum_ww + accumulator_elements, 0.0);

        const int families = 2 * annotation_bins_;
        std::vector<double> population_square_sums;
        std::vector<double> population_same_probe;
        if (total_probes_ >= 2) {
            population_square_sums.assign(
                checked_mul(
                    checked_mul(
                        static_cast<size_t>(environments_),
                        static_cast<size_t>(rows_),
                        "direct population rows"
                    ),
                    static_cast<size_t>(families),
                    "direct population square sums"
                ),
                0.0
            );
            population_same_probe.assign(
                checked_mul(
                    checked_mul(
                        static_cast<size_t>(environments_),
                        static_cast<size_t>(families),
                        "direct population families"
                    ),
                    static_cast<size_t>(families),
                    "direct population same-probe products"
                ),
                0.0
            );
        }

        nb::list calls;
        nb::list numa_decode_records;
        nb::list source_panel_numa_records;
        const bool fused = fused_two_pass_execution_;
        struct TileState {
            int environment_start = 0;
            int environment_stop = 0;
            int probe_start = 0;
            int probe_count = 0;
            int columns = 0;
            int wide = 0;
            std::unique_ptr<PackedSourcePanel> source;
            std::unique_ptr<nb_numpy_mat2f<double>> panel;
        };
        auto decode_dense_block = [&](int start, int stop,
                                      bool collect_missingness) {
            const int width = stop - start;
            DenseDecodeState& state = dense_decode_state_;
            int64_t missing = 0;
            double* data = decode_context_block(
                start, stop, state.legacy_storage, state.bound_storage,
                state.evidence, state.observed, missing, numa_decode_records
            );
            if (state.observed.size() != static_cast<size_t>(width)) {
                throw std::runtime_error(
                    "Reusable dense decode lost observation counts"
                );
            }
            for (int variant = 0; variant < width; ++variant) {
                missing_counts[static_cast<size_t>(start + variant)] =
                    static_cast<int64_t>(
                        rows_ - state.observed[static_cast<size_t>(variant)]
                    );
            }
            if (collect_missingness && missing != 0) {
                collect_missingness_correlations(
                    start, stop, missing_counts, missing_correlations
                );
            }
            nb::capsule owner(data, [](void*) noexcept {});
            return nb_numpy_mat2f<double>(
                data,
                {static_cast<size_t>(rows_), static_cast<size_t>(width)},
                owner
            );
        };
        std::vector<TileState> fused_states;
        if (fused) {
            const auto [environment_start, environment_stop] =
                environment_tiles_[0];
            fused_states.reserve(probe_tiles_.size());
            for (const auto [probe_start, probe_count] : probe_tiles_) {
                TileState state;
                state.environment_start = environment_start;
                state.environment_stop = environment_stop;
                state.probe_start = probe_start;
                state.probe_count = probe_count;
                state.columns = annotation_bins_ * probe_count;
                state.wide = (environment_stop - environment_start)
                    * 2 * state.columns;
                state.source = std::make_unique<PackedSourcePanel>(
                    rows_, dense_blas_hybrid_ ? 2 * state.wide : state.wide,
                    numa_request_,
                    dense_blas_hybrid_
                        ? "persistent_source_and_environment_weighted_target_pair"
                        : "persistent_packed_source_panel",
                    dense_blas_hybrid_
                        ? "after_environment_weighting_before_target_scoring"
                        : "after_projection_before_target_scoring"
                );
                state.panel = std::make_unique<nb_numpy_mat2f<double>>(
                    dense_blas_hybrid_
                        ? state.source->writable_column_view(0, state.wide)
                        : state.source->writable_view()
                );
                fused_states.push_back(std::move(state));
            }
        }

        // Reuse one packed-block owner across every genotype pass.  Packing a
        // new block resizes its vectors in place, preserving the largest
        // capacities instead of returning a multi-megabyte workspace to the
        // allocator after every block.
        MailmanPackedBlock packed_scratch;
        for (size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
            const auto [start, stop] = blocks_[block_index];
            const auto wall_start = std::chrono::steady_clock::now();
            const std::clock_t cpu_start = std::clock();
            nb::dict feature;
            std::unique_ptr<nb_numpy_mat2f<double>> dense_block;
            if (dense_blas_hybrid_) {
                dense_block = std::make_unique<nb_numpy_mat2f<double>>(
                    decode_dense_block(start, stop, true)
                );
                feature = kernel_.feature_block(
                    readonly_f_view(*dense_block), eps_var_, standardized_
                );
            } else {
                pack_context_block(start, stop, packed_scratch);
                int64_t missing = 0;
                for (int count : packed_scratch.observed) {
                    missing += static_cast<int64_t>(rows_ - count);
                }
                for (int variant = 0; variant < stop - start; ++variant) {
                    missing_counts[static_cast<size_t>(start + variant)] =
                        static_cast<int64_t>(
                            rows_ - packed_scratch.observed[
                                static_cast<size_t>(variant)
                            ]
                        );
                }
                if (missing != 0) {
                    collect_missingness_correlations(
                        start, stop, missing_counts, missing_correlations
                    );
                }
                feature = kernel_.feature_block_packed(
                    packed_scratch, eps_var_, standardized_
                );
            }
            const int64_t feature_repairs = nb::cast<int64_t>(
                feature["repaired_gemm_output_columns"]
            );
            append_call(
                calls,
                dense_blas_hybrid_ ? "multi_feature" : "multi_feature_mailman",
                feature_basis_columns(), stop - start,
                rows_, "T", rows_, rows_, feature_basis_columns(),
                start, stop, -1, -1, -1, -1, dense_blas_hybrid_,
                elapsed_wall(wall_start), elapsed_cpu(cpu_start),
                1.0, 0.0, feature_repairs
            );
            copy_feature_block(
                feature, start, stop, scale_x, scale_w, norm_x, norm_w,
                diag_x, diag_w, corr_xw, max_leak_x, max_leak_w
            );
            if (fused) {
                for (TileState& state : fused_states) {
                    if (dense_blas_hybrid_) {
                        accumulate_dense_source_for_block(
                            *state.panel, *dense_block, block_index,
                            start, stop, state.environment_start,
                            state.environment_stop, state.probe_start,
                            state.probe_count, scale_x, scale_w, calls
                        );
                    } else {
                        accumulate_source_for_block(
                            *state.panel, packed_scratch, block_index,
                            start, stop, state.environment_start,
                            state.environment_stop, state.probe_start,
                            state.probe_count, scale_x, scale_w, calls
                        );
                    }
                }
            }
        }

        auto finalize_panel = [&](TileState& state) {
            if (state.panel == nullptr || state.source == nullptr) {
                throw std::runtime_error(
                    "Direct multi-environment source tile is absent"
                );
            }
            const auto projection_wall = std::chrono::steady_clock::now();
            const std::clock_t projection_cpu = std::clock();
            (void)kernel_.project_sources(
                writable_f_view(*state.panel), state.columns,
                state.environment_start
            );
            if (common_rank_ > 0) {
                append_call(
                    calls, "multi_projection", common_rank_, state.wide,
                    rows_, "T", rows_, rows_, common_rank_, -1, -1,
                    state.environment_start, state.environment_stop,
                    state.probe_start, state.probe_count, true,
                    elapsed_wall(projection_wall), elapsed_cpu(projection_cpu)
                );
                append_call(
                    calls, "nn_update", rows_, state.wide, common_rank_, "N",
                    rows_, common_rank_, rows_, -1, -1,
                    state.environment_start, state.environment_stop,
                    state.probe_start, state.probe_count, false, 0.0, 0.0,
                    -1.0, 1.0
                );
            }
            if (total_probes_ >= 2) {
                accumulate_population(
                    state.panel->data(), state.environment_start,
                    state.environment_stop, state.probe_count,
                    population_square_sums, population_same_probe
                );
            }
            if (dense_blas_hybrid_) {
                auto weighted_panel = state.source->writable_column_view(
                    state.wide, state.wide
                );
                fill_environment_weighted_panel(
                    *state.panel, weighted_panel,
                    state.environment_start, state.environment_stop,
                    state.columns
                );
            }
            state.panel.reset();
            state.source->seal();
            nb::dict source_evidence = state.source->numa_evidence();
            source_evidence["environment_start"] = state.environment_start;
            source_evidence["environment_stop"] = state.environment_stop;
            source_evidence["probe_start"] = state.probe_start;
            source_evidence["probe_count"] = state.probe_count;
            source_panel_numa_records.append(std::move(source_evidence));
        };

        auto score_target_block = [&](TileState& state,
                                      const MailmanPackedBlock& packed,
                                      int start, int stop) {
            if (state.source == nullptr) {
                throw std::runtime_error(
                    "Direct multi-environment packed source tile is absent"
                );
            }
            const int tile = state.environment_stop - state.environment_start;
            double* block_scale_x = nullptr;
            double* block_scale_w = nullptr;
            auto block_scale_x_owner = make_owned_numpy_mat2f<double>(
                static_cast<size_t>(stop - start), static_cast<size_t>(tile),
                &block_scale_x
            );
            auto block_scale_w_owner = make_owned_numpy_mat2f<double>(
                static_cast<size_t>(stop - start), static_cast<size_t>(tile),
                &block_scale_w
            );
            copy_scale_tile(
                scale_x, scale_w, start, stop,
                state.environment_start, state.environment_stop,
                block_scale_x, block_scale_w
            );
            const auto target_wall = std::chrono::steady_clock::now();
            const std::clock_t target_cpu = std::clock();
            auto source_view = state.source->readonly_view();
            const int64_t target_repairs = kernel_.target_score_block_packed(
                packed, readonly_f_view(source_view),
                readonly_f_view(block_scale_x_owner),
                readonly_f_view(block_scale_w_owner),
                writable_c_view(accum_xx_out),
                writable_c_view(accum_xw_out),
                writable_c_view(accum_wx_out),
                writable_c_view(accum_ww_out),
                start, variants_, state.probe_count, state.environment_start
            );
            append_call(
                calls, "multi_target_mailman", stop - start,
                4 * tile * state.columns, rows_, "T", rows_, rows_,
                stop - start, start, stop,
                state.environment_start, state.environment_stop,
                state.probe_start, state.probe_count, false,
                elapsed_wall(target_wall), elapsed_cpu(target_cpu),
                1.0, 0.0, target_repairs
            );
        };

        auto score_dense_target_block = [&](TileState& state,
                                             nb_numpy_mat2f<double>& genotype,
                                             int start, int stop) {
            if (state.source == nullptr) {
                throw std::runtime_error(
                    "Dense direct target pair is absent"
                );
            }
            const int tile = state.environment_stop - state.environment_start;
            double* block_scale_x = nullptr;
            double* block_scale_w = nullptr;
            auto block_scale_x_owner = make_owned_numpy_mat2f<double>(
                static_cast<size_t>(stop - start), static_cast<size_t>(tile),
                &block_scale_x
            );
            auto block_scale_w_owner = make_owned_numpy_mat2f<double>(
                static_cast<size_t>(stop - start), static_cast<size_t>(tile),
                &block_scale_w
            );
            copy_scale_tile(
                scale_x, scale_w, start, stop,
                state.environment_start, state.environment_stop,
                block_scale_x, block_scale_w
            );
            auto source_view = state.source->readonly_view();
            const auto wall_start = std::chrono::steady_clock::now();
            const std::clock_t cpu_start = std::clock();
            const int64_t repairs = kernel_.target_score_block_dense_pair(
                readonly_f_view(genotype), readonly_f_view(source_view),
                readonly_f_view(block_scale_x_owner),
                readonly_f_view(block_scale_w_owner),
                writable_c_view(accum_xx_out),
                writable_c_view(accum_xw_out),
                writable_c_view(accum_wx_out),
                writable_c_view(accum_ww_out),
                start, variants_, state.probe_count, state.environment_start
            );
            append_call(
                calls, "multi_target_pair", stop - start, 2 * state.wide,
                rows_, "T", rows_, rows_, stop - start, start, stop,
                state.environment_start, state.environment_stop,
                state.probe_start, state.probe_count, true,
                elapsed_wall(wall_start), elapsed_cpu(cpu_start),
                1.0, 0.0, repairs
            );
        };

        if (fused) {
            for (TileState& state : fused_states) finalize_panel(state);
            kernel_.release_source_scratch();
            for (size_t block_index = 0;
                 block_index < blocks_.size(); ++block_index) {
                const auto [start, stop] = blocks_[block_index];
                if (dense_blas_hybrid_) {
                    auto dense_block = decode_dense_block(start, stop, false);
                    for (TileState& state : fused_states) {
                        score_dense_target_block(
                            state, dense_block, start, stop
                        );
                    }
                } else {
                    pack_context_block(start, stop, packed_scratch);
                    for (TileState& state : fused_states) {
                        score_target_block(
                            state, packed_scratch, start, stop
                        );
                    }
                }
            }
        } else {
            for (const auto [probe_start, probe_count] : probe_tiles_) {
                for (const auto [environment_start, environment_stop] :
                     environment_tiles_) {
                    TileState state;
                    state.environment_start = environment_start;
                    state.environment_stop = environment_stop;
                    state.probe_start = probe_start;
                    state.probe_count = probe_count;
                    state.columns = annotation_bins_ * probe_count;
                    state.wide = (environment_stop - environment_start)
                        * 2 * state.columns;
                    state.source = std::make_unique<PackedSourcePanel>(
                        rows_, dense_blas_hybrid_ ? 2 * state.wide : state.wide,
                        numa_request_,
                        dense_blas_hybrid_
                            ? "persistent_source_and_environment_weighted_target_pair"
                            : "persistent_packed_source_panel",
                        dense_blas_hybrid_
                            ? "after_environment_weighting_before_target_scoring"
                            : "after_projection_before_target_scoring"
                    );
                    state.panel =
                        std::make_unique<nb_numpy_mat2f<double>>(
                            dense_blas_hybrid_
                                ? state.source->writable_column_view(
                                    0, state.wide
                                )
                                : state.source->writable_view()
                        );
                    for (size_t block_index = 0;
                        block_index < blocks_.size(); ++block_index) {
                        const auto [start, stop] = blocks_[block_index];
                        if (dense_blas_hybrid_) {
                            auto dense_block = decode_dense_block(
                                start, stop, false
                            );
                            accumulate_dense_source_for_block(
                                *state.panel, dense_block, block_index,
                                start, stop, environment_start, environment_stop,
                                probe_start, probe_count,
                                scale_x, scale_w, calls
                            );
                        } else {
                            pack_context_block(start, stop, packed_scratch);
                            accumulate_source_for_block(
                                *state.panel, packed_scratch, block_index,
                                start, stop, environment_start, environment_stop,
                                probe_start, probe_count,
                                scale_x, scale_w, calls
                            );
                        }
                    }
                    finalize_panel(state);
                    kernel_.release_source_scratch();
                    for (size_t block_index = 0;
                        block_index < blocks_.size(); ++block_index) {
                        const auto [start, stop] = blocks_[block_index];
                        if (dense_blas_hybrid_) {
                            auto dense_block = decode_dense_block(
                                start, stop, false
                            );
                            score_dense_target_block(
                                state, dense_block, start, stop
                            );
                        } else {
                            pack_context_block(start, stop, packed_scratch);
                            score_target_block(
                                state, packed_scratch, start, stop
                            );
                        }
                    }
                }
            }
        }

        kernel_.normalize_scores(
            writable_c_view(accum_xx_out), writable_c_view(accum_xw_out),
            writable_c_view(accum_wx_out), writable_c_view(accum_ww_out),
            total_probes_
        );
        reader_->check_files_unchanged();
        if (nb::len(calls) > planned_semantic_call_maximum_) {
            throw std::runtime_error(
                "Direct multi-environment execution produced more semantic "
                "call records than its frozen plan admits"
            );
        }
        completed_ = true;

        nb::dict scores;
        scores["xx"] = std::move(accum_xx_out);
        scores["xw"] = std::move(accum_xw_out);
        scores["wx"] = std::move(accum_wx_out);
        scores["ww"] = std::move(accum_ww_out);
        nb::dict features;
        features["scale_x"] = std::move(scale_x_out);
        features["scale_w"] = std::move(scale_w_out);
        features["norm_x"] = std::move(norm_x_out);
        features["norm_w"] = std::move(norm_w_out);
        features["diag_nxe_x"] = std::move(diag_x_out);
        features["diag_nxe_w"] = std::move(diag_w_out);
        features["corr_xw"] = std::move(corr_xw_out);
        features["max_projection_leakage_additive"] = max_leak_x;
        features["max_projection_leakage_interaction"] = max_leak_w;

        nb::dict result;
        result["features"] = std::move(features);
        result["scores"] = std::move(scores);
        result["missing_counts"] = missing_counts;
        result["missing_environment_correlations"] =
            std::move(missing_correlations_out);
        result["calls"] = std::move(calls);
        result["numa_bound_decode_records"] = std::move(numa_decode_records);
        result["packed_source_panel_numa_records"] =
            std::move(source_panel_numa_records);
        result["observed_genotype_block_reads"] = observed_block_reads_;
        result["decoded_scratch_allocations"] = decoded_scratch_allocations_;
        result["decoded_scratch_reuses"] = decoded_scratch_reuses_;
        result["population_same_individual_products"] =
            finalize_population(
                population_square_sums, population_same_probe
            );
        result["kernel_info"] = kernel_.info();
        result["context_info"] = info_unlocked();
        return result;
    }

private:
    static nb::dict numa_decode_record_to_dict(
        const NativeGemmOutputNumaEvidenceData& evidence,
        int block_start,
        int block_stop
    ) {
        if (!evidence.applicable || !evidence.contract_required
            || !evidence.complete) {
            throw std::runtime_error(
                "Contracted native BED decode evidence is incomplete"
            );
        }
        nb::list selected_nodes;
        for (int node : evidence.selected_nodes) selected_nodes.append(node);

        nb::dict allocation;
        allocation["schema"] = "summit.numa_bound_anonymous_buffer.v1";
        // schema_version 2 adds capacity_byte_count: the decode scratch is
        // one maximum-capacity mapping reused by every block width.
        allocation["schema_version"] = 2;
        allocation["byte_count"] = evidence.logical_byte_count;
        allocation["capacity_byte_count"] = evidence.capacity_byte_count;
        allocation["mapping_bytes"] = evidence.mapping_bytes;
        allocation["page_size"] = evidence.page_size;
        allocation["page_count"] = evidence.mapping_page_count;
        allocation["selected_nodes"] = selected_nodes;
        allocation["policy_mode"] = "bind_static_nodes";
        allocation["page_aligned_mapping"] = evidence.page_aligned_mapping;
        allocation["bound_before_first_touch"] =
            evidence.bound_before_first_touch;
        allocation["live_owner_policy_verified"] =
            evidence.pre_touch_live_owner_policy_verified;
        allocation["range_policy_verified"] =
            evidence.pre_touch_range_policy_verified;
        allocation["page_migration_requested"] = false;
        allocation["placement_repair_performed"] = false;
        allocation["post_decode_complete_page_query"] = false;

        nb::dict verification;
        verification["schema"] = "summit.numa_bound_anonymous_buffer.v1";
        verification["schema_version"] = 2;
        verification["byte_count"] = evidence.logical_byte_count;
        verification["capacity_byte_count"] = evidence.capacity_byte_count;
        verification["mapping_bytes"] = evidence.mapping_bytes;
        verification["page_size"] = evidence.page_size;
        verification["page_count"] = evidence.mapping_page_count;
        verification["selected_nodes"] = selected_nodes;
        verification["policy_mode"] = "bind_static_nodes";
        verification["page_aligned_mapping"] = evidence.page_aligned_mapping;
        verification["bound_before_first_touch"] =
            evidence.bound_before_first_touch;
        verification["live_owner_policy_verified"] =
            evidence.post_repair_live_owner_policy_verified;
        verification["range_policy_verified"] =
            evidence.post_repair_range_policy_verified;
        verification["page_migration_requested"] = false;
        verification["placement_repair_performed"] = false;
        verification["post_decode_complete_page_query"] =
            evidence.post_repair_complete_page_query;
        verification["post_decode_strict_policy_verified"] =
            evidence.post_repair_strict_policy_verified;
        verification["queried_pages"] = evidence.queried_pages;
        verification["resolved_pages"] = evidence.resolved_pages;
        verification["query_chunks"] = evidence.query_chunks;
        verification["query_chunk_page_limit"] =
            kNativeGemmOutputQueryChunkPages;
        nb::dict node_histogram;
        for (const auto& item : evidence.node_histogram) {
            const std::string key = std::to_string(item.first);
            node_histogram[key.c_str()] = item.second;
        }
        verification["node_histogram"] = std::move(node_histogram);
        verification["ordered_status_sha256"] =
            evidence.ordered_status_sha256;
        verification["ordered_status_encoding"] =
            "native_32bit_signed_little";
        verification["complete"] = evidence.complete;

        nb::list genotype_block;
        genotype_block.append(block_start);
        genotype_block.append(block_stop);
        nb::dict result;
        result["genotype_block"] = std::move(genotype_block);
        result["memory_order"] = "F";
        result["decoder"] = "gxeldcore.DirectContext.decode_block";
        result["allocation"] = std::move(allocation);
        result["bound_mapping_preserved_after_standardization"] = true;
        result["verification_stage"] =
            "post_standardization_pre_return";
        result["verification"] = std::move(verification);
        return result;
    }

    double* decode_context_block(
        int block_start,
        int block_stop,
        std::unique_ptr<double[]>& legacy_storage,
        std::unique_ptr<NativeGemmOutputAllocation>& bound_storage,
        std::shared_ptr<SharedNativeGemmOutputNumaEvidence>& bound_evidence,
        std::vector<int>& observed,
        int64_t& missing,
        nb::list& records
    ) const {
        ++observed_block_reads_;
        if (!numa_request_.required) {
            const size_t elements = checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(block_stop - block_start),
                "reusable direct BED decode"
            );
            if (elements * sizeof(double) > decoded_scratch_capacity_bytes_) {
                throw std::runtime_error(
                    "Direct BED decode exceeds its admitted scratch capacity"
                );
            }
            if (legacy_storage == nullptr) {
                legacy_storage = std::make_unique<double[]>(
                    decoded_scratch_capacity_bytes_ / sizeof(double)
                );
                ++decoded_scratch_allocations_;
            } else {
                ++decoded_scratch_reuses_;
            }
            missing = reader_->decode_block_into(
                block_start, block_stop, false,
                legacy_storage.get(), observed
            );
            return legacy_storage.get();
        }

        const size_t rows = static_cast<size_t>(rows_);
        const size_t columns = static_cast<size_t>(block_stop - block_start);
        const size_t logical_bytes = checked_mul(
            checked_mul(rows, columns, "direct bounded BED decode"),
            sizeof(double), "direct bounded BED decode"
        );
        NativeGemmOutputNumaEvidenceData initial;
        initial.applicable = true;
        initial.contract_required = true;
        bound_evidence =
            std::make_shared<SharedNativeGemmOutputNumaEvidence>(
                std::move(initial)
            );
        if (logical_bytes > decoded_scratch_capacity_bytes_) {
            throw std::runtime_error(
                "Direct bounded BED decode exceeds its admitted scratch capacity"
            );
        }
        if (bound_storage == nullptr) {
            bound_storage = std::make_unique<NativeGemmOutputAllocation>(
                rows, columns, "column_major", logical_bytes,
                numa_request_, bound_evidence,
                decoded_scratch_capacity_bytes_
            );
            ++decoded_scratch_allocations_;
        } else {
            bound_storage->reuse_for_call(
                rows, columns, "column_major", logical_bytes,
                numa_request_, bound_evidence
            );
            ++decoded_scratch_reuses_;
        }
        missing = reader_->decode_block_into(
            block_start, block_stop, false, bound_storage->data(), observed
        );
        reader_->check_files_unchanged();
        bound_storage->verify_after_repair();
        records.append(numa_decode_record_to_dict(
            bound_evidence->snapshot(), block_start, block_stop
        ));
        return bound_storage->data();
    }

    void pack_context_block(
        int block_start,
        int block_stop,
        MailmanPackedBlock& packed
    ) const {
        ++observed_block_reads_;
        read_block_mailman_mean_memory(
            reader_->bed_base_, reader_->bed_size_, reader_->n_total_,
            reader_->bytes_per_snp_, block_start, block_stop,
            reader_->rows_, reader_->ddof_, packed
        );
        if (packed.N != rows_ || packed.L != block_stop - block_start
            || packed.observed.size()
                != static_cast<size_t>(block_stop - block_start)) {
            throw std::runtime_error(
                "Descriptor-owned packed BED block is inconsistent"
            );
        }
        reader_->check_files_unchanged();
#if defined(__linux__)
        const size_t offset = checked_add(
            3U,
            checked_mul(
                static_cast<size_t>(block_start), reader_->bytes_per_snp_,
                "packed BED consumed offset"
            ),
            "packed BED consumed offset"
        );
        const size_t length = checked_mul(
            static_cast<size_t>(block_stop - block_start),
            reader_->bytes_per_snp_, "packed BED consumed range"
        );
        madvise_dontneed_consumed_range(
            reader_->bed_base_, reader_->bed_size_, offset, length
        );
#endif
    }

    double* unpack_context_block(
        int block_start,
        int block_stop,
        const MailmanPackedBlock& packed,
        std::unique_ptr<double[]>& legacy_storage,
        std::unique_ptr<NativeGemmOutputAllocation>& bound_storage,
        std::shared_ptr<SharedNativeGemmOutputNumaEvidence>& bound_evidence,
        nb::list& records
    ) const {
        const size_t rows = static_cast<size_t>(rows_);
        const size_t columns = static_cast<size_t>(block_stop - block_start);
        double* destination = nullptr;
        if (!numa_request_.required) {
            legacy_storage = std::make_unique<double[]>(
                checked_mul(rows, columns, "unpacked genotype block")
            );
            destination = legacy_storage.get();
        } else {
            const size_t logical_bytes = checked_mul(
                checked_mul(rows, columns, "packed bounded BED decode"),
                sizeof(double), "packed bounded BED decode"
            );
            NativeGemmOutputNumaEvidenceData initial;
            initial.applicable = true;
            initial.contract_required = true;
            bound_evidence =
                std::make_shared<SharedNativeGemmOutputNumaEvidence>(
                    std::move(initial)
                );
            bound_storage = std::make_unique<NativeGemmOutputAllocation>(
                rows, columns, "column_major", logical_bytes,
                numa_request_, bound_evidence
            );
            destination = bound_storage->data();
        }

        int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(decode_threads_) \
    reduction(|:invalid)
#endif
        for (int64_t segment = 0; segment < packed.n_segments; ++segment) {
            const int base = static_cast<int>(
                segment * static_cast<int64_t>(packed.segment_size)
            );
            const int actual = std::min(
                packed.segment_size, packed.L - base
            );
            int64_t divisor = compute_mailman_table_size(actual);
            for (int local_variant = 0;
                 local_variant < actual; ++local_variant) {
                divisor /= 3;
                const int variant = base + local_variant;
                const double mean = packed.mean[
                    static_cast<size_t>(variant)
                ];
                const double inv_std = packed.inv_std[
                    static_cast<size_t>(variant)
                ];
                double* column = destination
                    + static_cast<size_t>(variant) * rows;
                if (packed.use_u16) {
                    const uint16_t* codes = packed.packed16.data()
                        + static_cast<size_t>(segment) * rows;
                    for (int row = 0; row < rows_; ++row) {
                        const int genotype = static_cast<int>(
                            (codes[static_cast<size_t>(row)] / divisor) % 3
                        );
                        column[row] = (static_cast<double>(genotype) - mean)
                            * inv_std;
                        invalid |= !std::isfinite(column[row]);
                    }
                } else {
                    const uint32_t* codes = packed.packed32.data()
                        + static_cast<size_t>(segment) * rows;
                    for (int row = 0; row < rows_; ++row) {
                        const int genotype = static_cast<int>(
                            (codes[static_cast<size_t>(row)] / divisor) % 3
                        );
                        column[row] = (static_cast<double>(genotype) - mean)
                            * inv_std;
                        invalid |= !std::isfinite(column[row]);
                    }
                }
                for (int missing_row : packed.missing_rows[
                         static_cast<size_t>(variant)]) {
                    if (missing_row < 0 || missing_row >= rows_) {
                        invalid = 1;
                    } else {
                        column[missing_row] = 0.0;
                    }
                }
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Descriptor-owned packed BED expansion is non-finite"
            );
        }
        if (bound_storage != nullptr) {
            bound_storage->verify_after_repair();
            nb::dict record = numa_decode_record_to_dict(
                bound_evidence->snapshot(), block_start, block_stop
            );
            record["decoder"] =
                "gxeldcore.MailmanPackedBlock.expand_standardized";
            records.append(std::move(record));
        }
        return destination;
    }

    static void parse_ranges(
        nb_mat2c_ro<int64_t> raw,
        int axis_size,
        const char* label,
        std::vector<std::pair<int, int>>& output
    ) {
        if (raw.shape(1) != 2 || raw.shape(0) == 0) {
            throw std::runtime_error(std::string("Direct ") + label
                                     + " plan must be nonempty N-by-two");
        }
        int previous = 0;
        for (size_t index = 0; index < raw.shape(0); ++index) {
            const int64_t begin64 = raw.data()[index * 2U];
            const int64_t end64 = raw.data()[index * 2U + 1U];
            if (begin64 != previous || begin64 < 0 || end64 <= begin64 ||
                end64 > axis_size || end64 > std::numeric_limits<int>::max()) {
                throw std::runtime_error(std::string("Direct ") + label
                                         + " plan is not a contiguous partition");
            }
            output.emplace_back(
                static_cast<int>(begin64), static_cast<int>(end64)
            );
            previous = static_cast<int>(end64);
        }
        if (previous != axis_size) {
            throw std::runtime_error(std::string("Direct ") + label
                                     + " plan does not cover its axis");
        }
    }

    void parse_probe_tiles(nb_mat2c_ro<int64_t> raw) {
        if (raw.shape(1) != 2 || raw.shape(0) == 0) {
            throw std::runtime_error(
                "Direct probe tile plan must be nonempty N-by-two"
            );
        }
        requested_probe_tile_count_ = raw.shape(0);
        int previous = 0;
        for (size_t index = 0; index < raw.shape(0); ++index) {
            const int64_t start64 = raw.data()[index * 2U];
            const int64_t count64 = raw.data()[index * 2U + 1U];
            const int64_t stop64 = start64 + count64;
            if (start64 != previous || start64 < 0 || count64 <= 0
                || stop64 > total_probes_
                || stop64 > std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "Direct probe tile plan is not a contiguous partition"
                );
            }
            probe_tiles_.emplace_back(
                static_cast<int>(start64), static_cast<int>(count64)
            );
            maximum_execution_probe_chunk_width_ = std::max(
                maximum_execution_probe_chunk_width_,
                static_cast<int>(count64)
            );
            previous = static_cast<int>(stop64);
        }
        if (previous != total_probes_) {
            throw std::runtime_error(
                "Direct probe tile plan does not cover its axis"
            );
        }
    }

    static nb_mat2f_ro<double> readonly_f_view(
        const nb_numpy_mat2f<double>& value
    ) {
        return nb_mat2f_ro<double>(
            value.data(), {value.shape(0), value.shape(1)}
        );
    }

    static nb_mat2f_rw<double> writable_f_view(
        nb_numpy_mat2f<double>& value
    ) {
        return nb_mat2f_rw<double>(
            value.data(), {value.shape(0), value.shape(1)}
        );
    }

    static nb_mat2c_rw<double> writable_c_view(
        nb_numpy_mat2c<double>& value
    ) {
        return nb_mat2c_rw<double>(
            value.data(), {value.shape(0), value.shape(1)}
        );
    }

    int feature_basis_columns() const noexcept {
        return feature_columns_;
    }

    static double elapsed_wall(
        const std::chrono::steady_clock::time_point& start
    ) {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
    }

    static double elapsed_cpu(std::clock_t start) {
        return static_cast<double>(std::clock() - start)
            / static_cast<double>(CLOCKS_PER_SEC);
    }

    static void append_call(
        nb::list& calls,
        const char* operation,
        int m,
        int n,
        int k,
        const char* transpose_a,
        int lda,
        int ldb,
        int ldc,
        int block_start,
        int block_stop,
        int environment_start,
        int environment_stop,
        int probe_start,
        int probe_count,
        bool output_expected,
        double wall_seconds,
        double process_cpu_seconds,
        double alpha = 1.0,
        double beta = 0.0,
        int64_t materially_repaired_columns = 0
    ) {
        nb::dict fallback;
        fallback["operation"] = operation;
        fallback["arithmetic_dtype"] = "float64";
        fallback["actual_left_storage_dtype"] = "float64";
        fallback["actual_right_storage_dtype"] = "float64";
        fallback["actual_output_storage_dtype"] = "float64";
        fallback["layout"] = "column_major";
        fallback["transpose_a"] = transpose_a;
        fallback["transpose_b"] = "N";
        fallback["m"] = m;
        fallback["n"] = n;
        fallback["k"] = k;
        fallback["lda"] = lda;
        fallback["ldb"] = ldb;
        fallback["ldc"] = ldc;
        fallback["alpha"] = alpha;
        fallback["beta"] = beta;
        fallback["wall_seconds"] = wall_seconds;
        fallback["process_cpu_seconds"] = process_cpu_seconds;
        fallback["omp_in_parallel"] = nb::none();
        fallback["completed"] = true;
        nb::dict semantic;
        if (block_start >= 0) {
            semantic["genotype_block"] = nb::make_tuple(
                block_start, block_stop
            );
            semantic["genotype_block_width"] = block_stop - block_start;
        }
        if (environment_start >= 0) {
            semantic["environment_tile"] = nb::make_tuple(
                environment_start, environment_stop
            );
        }
        if (probe_start >= 0) {
            semantic["probe_tile"] = nb::make_tuple(
                probe_start, probe_count
            );
        }
        if (std::strcmp(operation, "multi_feature") == 0
            || std::strcmp(operation, "multi_feature_mailman") == 0) {
            semantic["phase"] = "feature_moments";
        } else if (std::strcmp(operation, "multi_source") == 0
                   || std::strcmp(operation, "multi_source_mailman") == 0) {
            semantic["phase"] = "source_gemm";
        } else if (std::strcmp(operation, "multi_target_score") == 0
                   || std::strcmp(operation, "multi_target_mailman") == 0
                   || std::strcmp(operation, "multi_target_pair") == 0
                   || std::strcmp(operation, "multi_target_unweighted") == 0
                   || std::strcmp(operation, "multi_target_weighted") == 0) {
            semantic["phase"] = "target_gemm";
        } else {
            semantic["phase"] = "projection_context_correction";
        }
        semantic["materially_repaired_columns"] =
            materially_repaired_columns;
        nb::dict record;
        record["fallback"] = std::move(fallback);
        record["semantic"] = std::move(semantic);
        record["output_expected"] = output_expected;
        calls.append(std::move(record));
    }

    void collect_missingness_correlations(
        int start,
        int stop,
        const std::vector<int64_t>& missing_counts,
        double* correlations
    ) const {
        const int width = stop - start;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int local_variant = 0; local_variant < width; ++local_variant) {
            const int variant = start + local_variant;
            const int64_t missing = missing_counts[static_cast<size_t>(variant)];
            if (missing <= 0 || missing >= rows_) continue;
            const unsigned char* bytes = reader_->bed_base_ + 3
                + static_cast<size_t>(variant) * reader_->bytes_per_snp_;
            const double missing_norm = std::sqrt(
                static_cast<double>(missing)
                * (1.0 - static_cast<double>(missing)
                    / static_cast<double>(rows_))
            );
            for (int environment = 0; environment < environments_; ++environment) {
                long double total = 0.0L;
                for (int row = 0; row < rows_; ++row) {
                    const int source_row = reader_->rows_[static_cast<size_t>(row)];
                    const uint8_t bits = static_cast<uint8_t>(
                        (bytes[static_cast<size_t>(source_row >> 2)]
                            >> ((source_row & 3) << 1)) & 0x3U
                    );
                    if (bits == 1U) {
                        total += environment_values_[
                            static_cast<size_t>(environment)
                                * static_cast<size_t>(rows_)
                            + static_cast<size_t>(row)
                        ];
                    }
                }
                const double target_norm = environment_target_norms_[
                    static_cast<size_t>(environment)
                ];
                correlations[
                    static_cast<size_t>(environment)
                        * static_cast<size_t>(variants_)
                    + static_cast<size_t>(variant)
                ] = target_norm > 0.0
                    ? static_cast<double>(total)
                        / (missing_norm * target_norm)
                    : 0.0;
            }
        }
    }

    void copy_feature_block(
        const nb::dict& feature,
        int start,
        int stop,
        double* scale_x,
        double* scale_w,
        double* norm_x,
        double* norm_w,
        double* diag_x,
        double* diag_w,
        double* corr_xw,
        std::vector<double>& max_leak_x,
        std::vector<double>& max_leak_w
    ) const {
        const int width = stop - start;
        const auto copy_matrix = [&](const char* key, double* destination) {
            const auto source = nb::cast<nb_mat2f_ro<double>>(feature[key]);
            if (source.shape(0) != static_cast<size_t>(width) ||
                source.shape(1) != static_cast<size_t>(environments_)) {
                throw std::runtime_error(
                    "Direct multi-environment feature output is mis-sized"
                );
            }
            for (int environment = 0; environment < environments_; ++environment) {
                std::memcpy(
                    destination
                        + static_cast<size_t>(environment)
                            * static_cast<size_t>(variants_)
                        + static_cast<size_t>(start),
                    source.data()
                        + static_cast<size_t>(environment)
                            * static_cast<size_t>(width),
                    static_cast<size_t>(width) * sizeof(double)
                );
            }
        };
        copy_matrix("scale_x", scale_x);
        copy_matrix("scale_w", scale_w);
        copy_matrix("norm_x", norm_x);
        copy_matrix("norm_w", norm_w);
        copy_matrix("diag_nxe_x", diag_x);
        copy_matrix("diag_nxe_w", diag_w);
        copy_matrix("corr_xw", corr_xw);
        const auto leak_x = nb::cast<nb_vec1_ro<double>>(
            feature["max_projection_leakage_additive"]
        );
        const auto leak_w = nb::cast<nb_vec1_ro<double>>(
            feature["max_projection_leakage_interaction"]
        );
        for (int environment = 0; environment < environments_; ++environment) {
            max_leak_x[static_cast<size_t>(environment)] = std::max(
                max_leak_x[static_cast<size_t>(environment)],
                leak_x.data()[environment]
            );
            max_leak_w[static_cast<size_t>(environment)] = std::max(
                max_leak_w[static_cast<size_t>(environment)],
                leak_w.data()[environment]
            );
        }
    }

    void copy_scale_tile(
        const double* scale_x,
        const double* scale_w,
        int start,
        int stop,
        int environment_start,
        int environment_stop,
        double* block_scale_x,
        double* block_scale_w
    ) const {
        const int width = stop - start;
        for (int environment = environment_start;
             environment < environment_stop; ++environment) {
            const size_t source = static_cast<size_t>(environment)
                * static_cast<size_t>(variants_) + static_cast<size_t>(start);
            const size_t destination =
                static_cast<size_t>(environment - environment_start)
                    * static_cast<size_t>(width);
            std::memcpy(
                block_scale_x + destination, scale_x + source,
                static_cast<size_t>(width) * sizeof(double)
            );
            std::memcpy(
                block_scale_w + destination, scale_w + source,
                static_cast<size_t>(width) * sizeof(double)
            );
        }
    }

    void fill_environment_weighted_panel(
        const nb_numpy_mat2f<double>& source,
        nb_numpy_mat2f<double>& weighted,
        int environment_start,
        int environment_stop,
        int columns_per_environment_family
    ) const {
        const int tile = environment_stop - environment_start;
        const int expected_columns = 2 * tile
            * columns_per_environment_family;
        if (tile <= 0 || environment_start < 0
            || environment_stop > environments_
            || source.shape(0) != static_cast<size_t>(rows_)
            || weighted.shape(0) != source.shape(0)
            || source.shape(1) != static_cast<size_t>(expected_columns)
            || weighted.shape(1) != source.shape(1)) {
            throw std::runtime_error(
                "Dense environment-weighted target panel is mis-sized"
            );
        }
        int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    reduction(|:invalid)
#endif
        for (int column = 0; column < expected_columns; ++column) {
            const int local_environment = column
                / (2 * columns_per_environment_family);
            const double* environment = environment_values_.data()
                + static_cast<size_t>(
                    environment_start + local_environment
                ) * static_cast<size_t>(rows_);
            const double* input = source.data()
                + static_cast<size_t>(column) * static_cast<size_t>(rows_);
            double* output = weighted.data()
                + static_cast<size_t>(column) * static_cast<size_t>(rows_);
            for (int row = 0; row < rows_; ++row) {
                output[row] = input[row] * environment[row];
                invalid |= !std::isfinite(output[row]);
            }
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Dense environment-weighted target panel is non-finite"
            );
        }
    }

    void accumulate_dense_source_for_block(
        nb_numpy_mat2f<double>& panel,
        nb_numpy_mat2f<double>& genotype,
        size_t block_index,
        int start,
        int stop,
        int environment_start,
        int environment_stop,
        int probe_start,
        int probe_count,
        const double* scale_x,
        const double* scale_w,
        nb::list& calls
    ) {
        const int width = stop - start;
        double* probes = nullptr;
        auto probe_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width), static_cast<size_t>(probe_count),
            &probes
        );
        const uint64_t* keys = philox_keys_.data()
            + checked_mul(
                checked_add(
                    checked_mul(
                        block_index, static_cast<size_t>(total_probes_),
                        "dense direct Philox block offset"
                    ),
                    static_cast<size_t>(probe_start),
                    "dense direct Philox probe offset"
                ),
                2U, "dense direct Philox key offset"
            );
        {
            nb::gil_scoped_release release;
            fill_numpy_philox_rademacher(
                probes, width, keys, probe_count, threads_
            );
        }
        double* annotation = nullptr;
        auto annotation_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width),
            static_cast<size_t>(annotation_bins_), &annotation
        );
        for (int bin = 0; bin < annotation_bins_; ++bin) {
            for (int variant = 0; variant < width; ++variant) {
                annotation[
                    static_cast<size_t>(bin) * static_cast<size_t>(width)
                    + static_cast<size_t>(variant)
                ] = sqrt_annotation_values_[
                    static_cast<size_t>(start + variant)
                        * static_cast<size_t>(annotation_bins_)
                    + static_cast<size_t>(bin)
                ];
            }
        }
        const int tile = environment_stop - environment_start;
        double* block_scale_x = nullptr;
        double* block_scale_w = nullptr;
        auto block_scale_x_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width), static_cast<size_t>(tile),
            &block_scale_x
        );
        auto block_scale_w_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width), static_cast<size_t>(tile),
            &block_scale_w
        );
        copy_scale_tile(
            scale_x, scale_w, start, stop,
            environment_start, environment_stop,
            block_scale_x, block_scale_w
        );
        const auto wall_start = std::chrono::steady_clock::now();
        const std::clock_t cpu_start = std::clock();
        const int64_t source_repairs = kernel_.source_block(
            writable_f_view(panel), readonly_f_view(genotype),
            readonly_f_view(probe_owner), readonly_f_view(annotation_owner),
            readonly_f_view(block_scale_x_owner),
            readonly_f_view(block_scale_w_owner), environment_start,
            true
        );
        append_call(
            calls, "multi_source", rows_, panel.shape(1), width, "N",
            rows_, width, rows_, start, stop,
            environment_start, environment_stop,
            probe_start, probe_count, true,
            elapsed_wall(wall_start), elapsed_cpu(cpu_start),
            1.0, 0.0, source_repairs
        );
    }

    void accumulate_source_for_block(
        nb_numpy_mat2f<double>& panel,
        const MailmanPackedBlock& packed,
        size_t block_index,
        int start,
        int stop,
        int environment_start,
        int environment_stop,
        int probe_start,
        int probe_count,
        const double* scale_x,
        const double* scale_w,
        nb::list& calls
    ) {
        const int width = stop - start;
        if (packed.N != rows_ || packed.L != width) {
            throw std::runtime_error(
                "Packed source block disagrees with the genotype range"
            );
        }
        double* probes = nullptr;
        auto probe_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width), static_cast<size_t>(probe_count),
            &probes
        );
        const uint64_t* keys = philox_keys_.data()
            + checked_mul(
                checked_add(
                    checked_mul(
                        block_index, static_cast<size_t>(total_probes_),
                        "direct Philox block offset"
                    ),
                    static_cast<size_t>(probe_start),
                    "direct Philox probe offset"
                ),
                2U, "direct Philox key offset"
            );
        {
            nb::gil_scoped_release release;
            fill_numpy_philox_rademacher(
                probes, width, keys, probe_count, threads_
            );
        }
        double* annotation = nullptr;
        auto annotation_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width),
            static_cast<size_t>(annotation_bins_), &annotation
        );
        for (int bin = 0; bin < annotation_bins_; ++bin) {
            for (int variant = 0; variant < width; ++variant) {
                annotation[
                    static_cast<size_t>(bin) * static_cast<size_t>(width)
                    + static_cast<size_t>(variant)
                ] = sqrt_annotation_values_[
                    static_cast<size_t>(start + variant)
                        * static_cast<size_t>(annotation_bins_)
                    + static_cast<size_t>(bin)
                ];
            }
        }
        const int tile = environment_stop - environment_start;
        double* block_scale_x = nullptr;
        double* block_scale_w = nullptr;
        auto block_scale_x_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width), static_cast<size_t>(tile),
            &block_scale_x
        );
        auto block_scale_w_owner = make_owned_numpy_mat2f<double>(
            static_cast<size_t>(width), static_cast<size_t>(tile),
            &block_scale_w
        );
        copy_scale_tile(
            scale_x, scale_w, start, stop,
            environment_start, environment_stop,
            block_scale_x, block_scale_w
        );
        const auto wall_start = std::chrono::steady_clock::now();
        const std::clock_t cpu_start = std::clock();
        const int64_t source_repairs = kernel_.source_block_packed(
            writable_f_view(panel), packed,
            readonly_f_view(probe_owner), readonly_f_view(annotation_owner),
            readonly_f_view(block_scale_x_owner),
            readonly_f_view(block_scale_w_owner), environment_start,
            true
        );
        append_call(
            calls, "multi_source_mailman", rows_, panel.shape(1), width, "N",
            rows_, width, rows_, start, stop,
            environment_start, environment_stop,
            probe_start, probe_count, false,
            elapsed_wall(wall_start), elapsed_cpu(cpu_start),
            1.0, 0.0, source_repairs
        );
    }

    void accumulate_population(
        const double* panel,
        int environment_start,
        int environment_stop,
        int probe_count,
        std::vector<double>& square_sums,
        std::vector<double>& same_probe
    ) const {
        const int columns = annotation_bins_ * probe_count;
        constexpr int probe_chunk = 8;
        for (int environment = environment_start;
             environment < environment_stop; ++environment) {
            const int local_environment = environment - environment_start;
            double* environment_square = square_sums.data()
                + static_cast<size_t>(environment)
                    * static_cast<size_t>(rows_)
                    * static_cast<size_t>(families_count());
            double* environment_same = same_probe.data()
                + static_cast<size_t>(environment)
                    * static_cast<size_t>(families_count())
                    * static_cast<size_t>(families_count());
            // Parallelism assigns whole output coordinates to threads while
            // keeping every coordinate's accumulation order identical to the
            // serial loop (probe chunks ascending, rows and probes ascending
            // inside each chunk), so the parallel result is bitwise equal to
            // the serial one; only independent coordinates run concurrently.
            const int family_pairs =
                families_count() * (families_count() + 1) / 2;
            for (int first_probe = 0; first_probe < probe_count;
                 first_probe += probe_chunk) {
                const int stop_probe = std::min(
                    probe_count, first_probe + probe_chunk
                );
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_)
#endif
                for (int left = 0; left < families_count(); ++left) {
                    for (int row = 0; row < rows_; ++row) {
                        long double sum = 0.0L;
                        for (int probe = first_probe;
                             probe < stop_probe; ++probe) {
                            const int column = local_environment * 2 * columns
                                + left * probe_count + probe;
                            const double value = panel[
                                static_cast<size_t>(column)
                                    * static_cast<size_t>(rows_)
                                + static_cast<size_t>(row)
                            ];
                            sum += static_cast<long double>(value) * value;
                        }
                        environment_square[
                            static_cast<size_t>(row)
                                * static_cast<size_t>(families_count())
                            + static_cast<size_t>(left)
                        ] += static_cast<double>(sum);
                    }
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int pair = 0; pair < family_pairs; ++pair) {
                    // Unrank the upper-triangular (left, right >= left) pair.
                    int left = 0;
                    int remaining = pair;
                    while (remaining >= families_count() - left) {
                        remaining -= families_count() - left;
                        ++left;
                    }
                    const int right = left + remaining;
                    long double total = 0.0L;
                    for (int row = 0; row < rows_; ++row) {
                        for (int probe = first_probe;
                             probe < stop_probe; ++probe) {
                            const int left_column =
                                local_environment * 2 * columns
                                + left * probe_count + probe;
                            const int right_column =
                                local_environment * 2 * columns
                                + right * probe_count + probe;
                            const double lv = panel[
                                static_cast<size_t>(left_column)
                                    * static_cast<size_t>(rows_)
                                + static_cast<size_t>(row)
                            ];
                            const double rv = panel[
                                static_cast<size_t>(right_column)
                                    * static_cast<size_t>(rows_)
                                + static_cast<size_t>(row)
                            ];
                            total += static_cast<long double>(lv) * lv
                                * static_cast<long double>(rv) * rv;
                        }
                    }
                    environment_same[
                        static_cast<size_t>(left)
                            * static_cast<size_t>(families_count())
                        + static_cast<size_t>(right)
                    ] += static_cast<double>(total);
                    if (right != left) {
                        environment_same[
                            static_cast<size_t>(right)
                                * static_cast<size_t>(families_count())
                            + static_cast<size_t>(left)
                        ] += static_cast<double>(total);
                    }
                }
            }
        }
    }

    int families_count() const noexcept { return 2 * annotation_bins_; }

    nb::object finalize_population(
        const std::vector<double>& square_sums,
        const std::vector<double>& same_probe
    ) const {
        if (total_probes_ < 2) return nb::none();
        double* output = nullptr;
        auto result = make_owned_numpy_mat2c<double>(
            checked_mul(
                static_cast<size_t>(environments_),
                static_cast<size_t>(families_count()),
                "direct population output rows"
            ),
            static_cast<size_t>(families_count()), &output
        );
        for (int environment = 0; environment < environments_; ++environment) {
            const double* environment_square = square_sums.data()
                + static_cast<size_t>(environment)
                    * static_cast<size_t>(rows_)
                    * static_cast<size_t>(families_count());
            const double* environment_same = same_probe.data()
                + static_cast<size_t>(environment)
                    * static_cast<size_t>(families_count())
                    * static_cast<size_t>(families_count());
            for (int left = 0; left < families_count(); ++left) {
                const double left_mass = annotation_masses_[
                    static_cast<size_t>(left % annotation_bins_)
                ];
                for (int right = left; right < families_count(); ++right) {
                    const double right_mass = annotation_masses_[
                        static_cast<size_t>(right % annotation_bins_)
                    ];
                    long double total = 0.0L;
                    for (int row = 0; row < rows_; ++row) {
                        total += static_cast<long double>(environment_square[
                            static_cast<size_t>(row)
                                * static_cast<size_t>(families_count())
                            + static_cast<size_t>(left)
                        ]) * environment_square[
                            static_cast<size_t>(row)
                                * static_cast<size_t>(families_count())
                            + static_cast<size_t>(right)
                        ];
                    }
                    const double value = (
                        static_cast<double>(total)
                        - environment_same[
                            static_cast<size_t>(left)
                                * static_cast<size_t>(families_count())
                            + static_cast<size_t>(right)
                        ]
                    ) / (
                        static_cast<double>(total_probes_)
                        * static_cast<double>(total_probes_ - 1)
                        * left_mass * right_mass
                    );
                    const size_t base = static_cast<size_t>(environment)
                        * static_cast<size_t>(families_count())
                        * static_cast<size_t>(families_count());
                    output[base
                        + static_cast<size_t>(left)
                            * static_cast<size_t>(families_count())
                        + static_cast<size_t>(right)] = value;
                    output[base
                        + static_cast<size_t>(right)
                            * static_cast<size_t>(families_count())
                        + static_cast<size_t>(left)] = value;
                }
            }
        }
        return nb::cast(std::move(result));
    }

    nb::dict info_unlocked() const {
        nb::dict result;
        result["schema"] = "summit.multi_environment_direct_context.v3";
        result["schema_version"] = 1;
        result["rows"] = rows_;
        result["variants"] = variants_;
        result["environment_count"] = environments_;
        result["annotation_bins"] = annotation_bins_;
        result["probe_count"] = total_probes_;
        result["mailman_maximum_probe_count"] =
            kMultiEnvironmentMailmanMaximumProbes;
        result["mailman_probe_count_eligible"] =
            total_probes_ <= kMultiEnvironmentMailmanMaximumProbes;
        result["block_count"] = blocks_.size();
        result["environment_tile_count"] = environment_tiles_.size();
        result["probe_tile_count"] = requested_probe_tile_count_;
        result["execution_probe_chunk_count"] = probe_tiles_.size();
        result["maximum_execution_probe_chunk_width"] =
            maximum_execution_probe_chunk_width_;
        result["fused_two_pass_execution"] = fused_two_pass_execution_;
        result["planned_output_calls"] = planned_output_calls_;
        result["planned_semantic_call_maximum"] =
            planned_semantic_call_maximum_;
        result["planned_genotype_passes"] = planned_genotype_passes_;
        result["decode_threads"] = decode_threads_;
        result["threads"] = threads_;
        result["descriptor_owned_bed"] = true;
        result["native_probe_generation"] = true;
        result["execution_kernel"] = dense_blas_hybrid_
            ? "dense_private_blas_streamed_pair"
            : "packed_mailman_low_memory_fallback";
        result["packed_genotype_mailman"] = !dense_blas_hybrid_;
        result["packed_feature_moments"] = !dense_blas_hybrid_;
        result["packed_source_direct_accumulation"] = !dense_blas_hybrid_;
        result["virtual_environment_weighted_target_rhs"] =
            !dense_blas_hybrid_;
        result["materialized_target_rhs"] = dense_blas_hybrid_;
        result["dense_genotype_feature_decode"] = dense_blas_hybrid_;
        result["dense_genotype_target_decode"] = dense_blas_hybrid_;
        result["persistent_dense_decode_scratch"] = dense_blas_hybrid_;
        result["persistent_environment_weighted_target_panel"] =
            dense_blas_hybrid_;
        result["source_panel_released_before_target"] = false;
        result["source_panel_reused_as_protected_pair_first_half"] = false;
        result["contracted_numa_decode"] =
            dense_blas_hybrid_ && numa_request_.required;
        result["contracted_packed_source_panel_numa"] =
            numa_request_.required;
        result["max_genotype_block_width"] = max_genotype_block_width_;
        result["mailman_frozen_segment_size"] = mailman_frozen_segment_size_;
        result["mailman_frozen_table_size"] = mailman_frozen_table_size_;
        result["mailman_qpanel_feature"] = mailman_qpanel_feature_;
        result["mailman_qpanel_source"] = mailman_qpanel_source_;
        result["mailman_qpanel_target"] = mailman_qpanel_target_;
        result["mailman_worker_scratch_capacity_bytes_per_worker"] =
            mailman_worker_scratch_capacity_bytes_per_worker_;
        result["decoded_scratch_capacity_bytes"] =
            decoded_scratch_capacity_bytes_;
        result["source_output_scratch_capacity_bytes"] =
            source_output_scratch_capacity_bytes_;
        result["target_output_scratch_capacity_bytes"] =
            target_output_scratch_capacity_bytes_;
        result["single_use"] = true;
        result["completed"] = completed_;
        result["execution_scratch_released"] = execution_scratch_released_;
        result["decoded_scratch_released_bytes"] =
            decoded_scratch_released_bytes_;
        return result;
    }

    struct DenseDecodeState {
        std::unique_ptr<double[]> legacy_storage;
        std::unique_ptr<NativeGemmOutputAllocation> bound_storage;
        std::shared_ptr<SharedNativeGemmOutputNumaEvidence> evidence;
        std::vector<int> observed;
    };

    std::unique_ptr<DirectContext> reader_;
    MultiEnvironmentKernel kernel_;
    // Element-wise square roots of the canonical annotation matrix, computed
    // once at construction (audit Priority 1.6) and charged in the plan's
    // context copies; both source paths consume it instead of re-deriving
    // square roots per block/tile call.
    std::vector<double> sqrt_annotation_values_;
    mutable DenseDecodeState dense_decode_state_;
    mutable int64_t decoded_scratch_allocations_ = 0;
    mutable int64_t decoded_scratch_reuses_ = 0;
    bool execution_scratch_released_ = false;
    size_t decoded_scratch_released_bytes_ = 0;
    size_t planned_semantic_call_maximum_ = 0;
    int max_genotype_block_width_ = 0;
    int mailman_frozen_segment_size_ = 0;
    int64_t mailman_frozen_table_size_ = 0;
    int mailman_qpanel_feature_ = 0;
    int mailman_qpanel_source_ = 0;
    int mailman_qpanel_target_ = 0;
    size_t mailman_worker_scratch_capacity_bytes_per_worker_ = 0;
    size_t decoded_scratch_capacity_bytes_ = 0;
    size_t source_output_scratch_capacity_bytes_ = 0;
    size_t target_output_scratch_capacity_bytes_ = 0;
    int rows_ = 0;
    int variants_ = 0;
    int environments_ = 0;
    int annotation_bins_ = 0;
    int feature_columns_ = 0;
    int common_rank_ = 0;
    int total_probes_ = 0;
    double eps_var_ = 0.0;
    bool standardized_ = true;
    bool dense_blas_hybrid_ = false;
    int threads_ = 1;
    int decode_threads_ = 1;
    size_t planned_output_calls_ = 0;
    size_t planned_genotype_passes_ = 0;
    size_t requested_probe_tile_count_ = 0;
    int maximum_execution_probe_chunk_width_ = 0;
    bool fused_two_pass_execution_ = false;
    mutable size_t observed_block_reads_ = 0;
    bool completed_ = false;
    std::vector<double> annotation_values_;
    std::vector<double> annotation_masses_;
    std::vector<double> environment_values_;
    std::vector<double> environment_target_norms_;
    std::vector<uint64_t> philox_keys_;
    std::vector<std::pair<int, int>> blocks_;
    std::vector<std::pair<int, int>> environment_tiles_;
    std::vector<std::pair<int, int>> probe_tiles_;
    NativeNumaContractRequest numa_request_;
    mutable std::mutex mutex_;
};

#if defined(GWLDCORE_GEMM_INTEGRITY)
nb::dict matrix_fingerprint_to_diagnostic_dict(
    const MatrixFingerprint& fingerprint
) {
    nb::dict result;
    // Decimal strings retain every uint64 bit when the mapping is serialized
    // through JSON consumers that cannot represent 64-bit integers exactly.
    result["xor_hash_uint64"] = std::to_string(fingerprint.xor_hash);
    result["sum_hash_uint64"] = std::to_string(fingerprint.sum_hash);
    return result;
}

nb_numpy_mat2f<double> make_owned_diagnostic_mat2f(
    size_t rows, size_t columns, double** output
) {
    const size_t elements = checked_mul(
        rows, columns, "NN integrity diagnostic result matrix"
    );
    const size_t allocated_elements = std::max<size_t>(1U, elements);
    void* storage = nullptr;
    if (posix_memalign(
            &storage, 64,
            checked_mul(
                allocated_elements, sizeof(double),
                "NN integrity diagnostic result storage"
            )) != 0) {
        throw std::bad_alloc();
    }
    *output = static_cast<double*>(storage);
    nb::capsule owner(storage, [](void* pointer) noexcept {
        std::free(pointer);
    });
    return nb_numpy_mat2f<double>(
        static_cast<double*>(storage), {rows, columns}, owner
    );
}

nb::dict nn_integrity_diagnostic_to_dict(
    const NnIntegrityDiagnostic& diagnostic
) {
    nb::dict result;
    result["schema_version"] = 1;
    result["integrity_check_eligible"] =
        diagnostic.integrity_check_eligible;
    result["diagnostic_executed"] = diagnostic.diagnostic_executed;
    result["classification"] = diagnostic.classification;
    result["m"] = diagnostic.m;
    result["n"] = diagnostic.n;
    result["k"] = diagnostic.k;
    result["minimum_vendor_flops"] = kCheckedGemmMinimumFlops;
    result["long_double_reference_accumulator"] = "C++ long double";
    result["long_double_reference_array_storage"] =
        "binary64_cast_with_full_precision_decimal_companion";
    nb::dict floating_point_capabilities;
    nb::dict long_double_capabilities;
    long_double_capabilities["sizeof_bytes"] = sizeof(long double);
    long_double_capabilities["digits"] =
        std::numeric_limits<long double>::digits;
    long_double_capabilities["digits10"] =
        std::numeric_limits<long double>::digits10;
    long_double_capabilities["max_digits10"] =
        std::numeric_limits<long double>::max_digits10;
    long_double_capabilities["epsilon"] =
        std::numeric_limits<long double>::epsilon();
    nb::dict double_capabilities;
    double_capabilities["sizeof_bytes"] = sizeof(double);
    double_capabilities["digits"] = std::numeric_limits<double>::digits;
    double_capabilities["digits10"] = std::numeric_limits<double>::digits10;
    double_capabilities["max_digits10"] =
        std::numeric_limits<double>::max_digits10;
    double_capabilities["epsilon"] =
        std::numeric_limits<double>::epsilon();
    floating_point_capabilities["long_double"] =
        std::move(long_double_capabilities);
    floating_point_capabilities["double"] =
        std::move(double_capabilities);
    result["floating_point_capabilities"] =
        std::move(floating_point_capabilities);
    result["classification_changes_production_decision"] = false;
    result["detailed_column_limit"] = kNnIntegrityDiagnosticColumnLimit;
    result["flagged_column_count"] = diagnostic.flagged_columns.size();
    result["captured_flagged_column_count"] =
        diagnostic.captured_flagged_columns.size();
    result["dropped_flagged_column_count"] =
        diagnostic.flagged_columns.size()
        - diagnostic.captured_flagged_columns.size();
    result["flagged_check_count"] = diagnostic.flagged_check_ids.size();
    result["captured_flagged_check_count"] =
        diagnostic.flagged_checks.size();
    result["dropped_flagged_check_count"] =
        diagnostic.flagged_check_ids.size()
        - diagnostic.flagged_checks.size();
    if (diagnostic.flagged_check_ids.empty()) {
        result["first_flagged_column"] = nb::none();
        result["first_flagged_check"] = nb::none();
    } else {
        result["first_flagged_column"] =
            diagnostic.flagged_check_ids.front().first;
        result["first_flagged_check"] =
            diagnostic.flagged_check_ids.front().second;
    }
    result["flagged_columns"] = diagnostic.flagged_columns;
    result["captured_flagged_columns"] =
        diagnostic.captured_flagged_columns;

    nb::list flagged_check_ids;
    for (const auto& identifier : diagnostic.flagged_check_ids) {
        nb::dict item;
        item["column"] = identifier.first;
        item["check"] = identifier.second;
        flagged_check_ids.append(std::move(item));
    }
    result["flagged_check_ids"] = std::move(flagged_check_ids);

    nb::list flagged_checks;
    for (const auto& record : diagnostic.flagged_checks) {
        nb::dict item;
        item["column"] = record.column;
        item["check"] = record.check;
        item["expected"] = record.expected;
        item["observed"] = record.observed;
        item["difference"] = record.difference;
        item["absolute_difference"] = record.absolute_difference;
        item["current_relative_bound"] = record.current_relative_bound;
        item["current_tolerance"] = record.current_tolerance;
        item["direct_absolute_product_sum"] =
            record.direct_absolute_product_sum;
        item["factored_expected_absolute_sum"] =
            record.factored_expected_absolute_sum;
        item["observed_checksum_absolute_sum"] =
            record.observed_checksum_absolute_sum;
        item["projection_roundoff_component"] =
            record.projection_roundoff_component;
        item["expected_reduction_roundoff_component"] =
            record.expected_reduction_roundoff_component;
        item["vendor_product_roundoff_component"] =
            record.vendor_product_roundoff_component;
        item["observed_reduction_roundoff_component"] =
            record.observed_reduction_roundoff_component;
        item["cancellation_aware_bound"] =
            record.cancellation_aware_bound;
        item["cancellation_aware_disagrees"] =
            record.cancellation_aware_disagrees;
        flagged_checks.append(std::move(item));
    }
    result["flagged_checks"] = std::move(flagged_checks);

    nb::list column_comparisons;
    for (const auto& comparison : diagnostic.column_comparisons) {
        nb::dict item;
        item["column"] = comparison.column;
        item["classification"] = comparison.classification;
        item["raw_vendor_nonfinite_count"] =
            comparison.raw_vendor_nonfinite_count;
        item["deterministic_tiled_nonfinite_count"] =
            comparison.deterministic_tiled_nonfinite_count;
        item["long_double_reference_nonfinite_count"] =
            comparison.long_double_reference_nonfinite_count;
        item["vendor_tiled_unequal_count"] =
            comparison.vendor_tiled_unequal_count;
        item["vendor_reference_unequal_count"] =
            comparison.vendor_reference_unequal_count;
        item["tiled_reference_unequal_count"] =
            comparison.tiled_reference_unequal_count;
        item["vendor_rows_outside_forward_error_bound"] =
            comparison.vendor_rows_outside_forward_error_bound;
        item["tiled_rows_outside_forward_error_bound"] =
            comparison.tiled_rows_outside_forward_error_bound;
        item["max_abs_vendor_minus_tiled"] =
            comparison.max_abs_vendor_minus_tiled;
        item["max_abs_vendor_minus_long_double"] =
            comparison.max_abs_vendor_minus_long_double;
        item["max_abs_tiled_minus_long_double"] =
            comparison.max_abs_tiled_minus_long_double;
        item["max_vendor_forward_error_ratio"] =
            comparison.max_vendor_forward_error_ratio;
        item["max_tiled_forward_error_ratio"] =
            comparison.max_tiled_forward_error_ratio;
        column_comparisons.append(std::move(item));
    }
    result["column_comparisons"] = std::move(column_comparisons);

    const size_t flagged_count = diagnostic.captured_flagged_columns.size();
    double* raw_vendor = nullptr;
    auto raw_vendor_columns = make_owned_diagnostic_mat2f(
        static_cast<size_t>(diagnostic.m), flagged_count, &raw_vendor
    );
    double* deterministic_tiled = nullptr;
    auto deterministic_tiled_columns = make_owned_diagnostic_mat2f(
        static_cast<size_t>(diagnostic.m), flagged_count,
        &deterministic_tiled
    );
    double* long_double_reference = nullptr;
    auto long_double_reference_columns = make_owned_diagnostic_mat2f(
        static_cast<size_t>(diagnostic.m), flagged_count,
        &long_double_reference
    );
    if (!diagnostic.raw_vendor_columns.empty()) {
        std::copy(
            diagnostic.raw_vendor_columns.begin(),
            diagnostic.raw_vendor_columns.end(), raw_vendor
        );
        std::copy(
            diagnostic.deterministic_tiled_columns.begin(),
            diagnostic.deterministic_tiled_columns.end(), deterministic_tiled
        );
        std::copy(
            diagnostic.long_double_reference_columns.begin(),
            diagnostic.long_double_reference_columns.end(),
            long_double_reference
        );
    }
    result["raw_vendor_columns"] = std::move(raw_vendor_columns);
    result["deterministic_tiled_columns"] =
        std::move(deterministic_tiled_columns);
    result["long_double_reference_columns"] =
        std::move(long_double_reference_columns);

    nb::list decimal_columns;
    for (size_t selected = 0; selected < flagged_count; ++selected) {
        nb::list decimal_column;
        for (int row = 0; row < diagnostic.m; ++row) {
            decimal_column.append(
                diagnostic.long_double_reference_decimal_columns[
                    selected * static_cast<size_t>(diagnostic.m)
                    + static_cast<size_t>(row)
                ]
            );
        }
        decimal_columns.append(std::move(decimal_column));
    }
    result["long_double_reference_decimal_columns"] =
        std::move(decimal_columns);

    nb::dict fault_injection;
    fault_injection["enabled"] = diagnostic.fault_injection_enabled;
    if (diagnostic.fault_injection_enabled) {
        fault_injection["row"] = diagnostic.fault_injection_row;
        fault_injection["column"] = diagnostic.fault_injection_column;
        fault_injection["delta"] = diagnostic.fault_injection_delta;
        fault_injection["timing"] =
            "after_vendor_call_before_observed_checksum";
    } else {
        fault_injection["row"] = nb::none();
        fault_injection["column"] = nb::none();
        fault_injection["delta"] = nb::none();
        fault_injection["timing"] = nb::none();
    }
    result["fault_injection"] = std::move(fault_injection);

    nb::dict fingerprints;
    nb::dict original_b;
    nb::dict protected_b;
    nb::dict equalities;
    if (diagnostic.diagnostic_executed) {
        original_b["initial"] = matrix_fingerprint_to_diagnostic_dict(
            diagnostic.original_b_initial
        );
        original_b["after_expected"] = matrix_fingerprint_to_diagnostic_dict(
            diagnostic.original_b_after_expected
        );
        original_b["after_copy"] = matrix_fingerprint_to_diagnostic_dict(
            diagnostic.original_b_after_copy
        );
        original_b["after_vendor"] = matrix_fingerprint_to_diagnostic_dict(
            diagnostic.original_b_after_vendor
        );
        original_b["after_deterministic"] =
            matrix_fingerprint_to_diagnostic_dict(
                diagnostic.original_b_after_deterministic
            );
        original_b["after_reference"] =
            matrix_fingerprint_to_diagnostic_dict(
                diagnostic.original_b_after_reference
            );
        protected_b["before_vendor"] =
            matrix_fingerprint_to_diagnostic_dict(
                diagnostic.protected_b_before_vendor
            );
        protected_b["after_vendor"] =
            matrix_fingerprint_to_diagnostic_dict(
                diagnostic.protected_b_after_vendor
            );
        equalities["original_unchanged_after_expected"] =
            diagnostic.original_b_initial ==
                diagnostic.original_b_after_expected;
        equalities["original_unchanged_after_copy"] =
            diagnostic.original_b_initial == diagnostic.original_b_after_copy;
        equalities["original_unchanged_after_vendor"] =
            diagnostic.original_b_initial ==
                diagnostic.original_b_after_vendor;
        equalities["original_unchanged_after_deterministic"] =
            diagnostic.original_b_initial ==
                diagnostic.original_b_after_deterministic;
        equalities["original_unchanged_after_reference"] =
            diagnostic.original_b_initial ==
                diagnostic.original_b_after_reference;
        equalities["protected_unchanged_after_vendor"] =
            diagnostic.protected_b_before_vendor ==
                diagnostic.protected_b_after_vendor;
        equalities["original_equals_protected_before_vendor"] =
            diagnostic.original_b_after_copy ==
                diagnostic.protected_b_before_vendor;
        equalities["original_equals_protected_after_vendor"] =
            diagnostic.original_b_after_vendor ==
                diagnostic.protected_b_after_vendor;
    }
    fingerprints["original_b"] = std::move(original_b);
    fingerprints["protected_b_snapshot"] = std::move(protected_b);
    result["fingerprints"] = std::move(fingerprints);
    result["fingerprint_equalities"] = std::move(equalities);
    return result;
}

nb::tuple test_protected_matmul_nn_integrity_diagnostic(
    nb_mat2f_ro<double> left,
    nb_mat2f_ro<double> right,
    int requested_threads,
    int fault_injection_row,
    int fault_injection_column,
    double fault_injection_delta
) {
    validate_protected_gemm_threads(requested_threads);
    if (left.shape(1) != right.shape(0)) {
        throw std::runtime_error(
            "Diagnostic protected GxE NN GEMM operands have incompatible dimensions"
        );
    }
    const int m = checked_blas_dim(
        left.shape(0), "diagnostic protected NN rows"
    );
    const int k = checked_blas_dim(
        left.shape(1), "diagnostic protected NN reduction"
    );
    const int n = checked_blas_dim(
        right.shape(1), "diagnostic protected NN columns"
    );
    if (m == 0 || n == 0 || k == 0) {
        throw std::runtime_error(
            "Diagnostic protected GxE NN GEMM operands must be non-empty"
        );
    }
    const bool fault_injection_requested =
        fault_injection_row != -1
        || fault_injection_column != -1
        || fault_injection_delta != 0.0;
    if (fault_injection_requested
        && (fault_injection_row < 0 || fault_injection_row >= m
            || fault_injection_column < 0 || fault_injection_column >= n
            || !std::isfinite(fault_injection_delta)
            || fault_injection_delta == 0.0)) {
        throw std::runtime_error(
            "Diagnostic NN fault injection requires an in-range row and column "
            "and a finite nonzero delta"
        );
    }
    if (!fault_injection_requested) {
        fault_injection_row = -1;
        fault_injection_column = -1;
        fault_injection_delta = 0.0;
    }

    NnIntegrityDiagnostic diagnostic;
    diagnostic.m = m;
    diagnostic.n = n;
    diagnostic.k = k;
    diagnostic.integrity_check_eligible =
        gemm_requires_integrity_checks(m, n, k);
    if (fault_injection_requested && !diagnostic.integrity_check_eligible) {
        throw std::runtime_error(
            "Diagnostic NN fault injection requires a GEMM at or above the "
            "integrity-check threshold"
        );
    }
    NativeGemmOutputTelemetryScope output_scope;
    NativeGemmOutputAllocation* output_allocation = nullptr;
    double* output = nullptr;
    auto result = make_native_gemm_output_mat2f(
        static_cast<size_t>(m), static_cast<size_t>(n), output_scope,
        &output_allocation, &output
    );
    int64_t repaired = 0;
    {
        nb::gil_scoped_release release;
        if (diagnostic.integrity_check_eligible) {
            diagnostic = dgemm_nn_checked_diagnostic(
                m, n, k,
                left.data(), m,
                right.data(), k,
                output, m,
                requested_threads,
                fault_injection_row,
                fault_injection_column,
                fault_injection_delta
            );
            repaired = static_cast<int64_t>(
                diagnostic.flagged_columns.size()
            );
        } else {
            repaired = dgemm_nn_partitioned_rows(
                m, n, k,
                left.data(), m,
                right.data(), k,
                output, m,
                requested_threads
            );
        }
        output_allocation->verify_after_repair();
    }
    output_scope.complete();
    return nb::make_tuple(
        result, repaired, nn_integrity_diagnostic_to_dict(diagnostic)
    );
}
#endif

nb::tuple protected_matmul_tn(
    nb_mat2f_ro<double> left,
    nb_mat2f_ro<double> right,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    if (left.shape(0) != right.shape(0)) {
        throw std::runtime_error(
            "Protected GxE TN GEMM operands have incompatible dimensions"
        );
    }
    const int k = checked_blas_dim(left.shape(0), "protected TN reduction");
    const int m = checked_blas_dim(left.shape(1), "protected TN rows");
    const int n = checked_blas_dim(right.shape(1), "protected TN columns");
    if (m == 0 || n == 0 || k == 0) {
        throw std::runtime_error("Protected GxE TN GEMM operands must be non-empty");
    }
    NativeGemmOutputTelemetryScope output_scope;
    NativeGemmOutputAllocation* output_allocation = nullptr;
    double* output = nullptr;
    auto result = make_native_gemm_output_mat2f(
        static_cast<size_t>(m), static_cast<size_t>(n), output_scope,
        &output_allocation, &output
    );
    int64_t repaired = 0;
    {
        nb::gil_scoped_release release;
        repaired = dgemm_tn_partitioned_rows(
            m, n, k,
            left.data(), k,
            right.data(), k,
            output, m,
            requested_threads
        );
        output_allocation->verify_after_repair();
    }
    output_scope.complete();
    return nb::make_tuple(result, repaired);
}

void protected_rank_update_nn(
    nb_mat2f_ro<double> basis,
    nb_mat2f_ro<double> coefficients,
    nb_mat2f_rw<double> target,
    int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const int m = checked_blas_dim(
        target.shape(0), "protected rank-update rows"
    );
    const int n = checked_blas_dim(
        target.shape(1), "protected rank-update columns"
    );
    const int k = checked_blas_dim(
        basis.shape(1), "protected rank-update rank"
    );
    if (m <= 0 || n <= 0 || k <= 0 ||
        checked_blas_dim(basis.shape(0), "protected rank-update basis rows") != m ||
        checked_blas_dim(coefficients.shape(0),
                         "protected rank-update coefficient rows") != k ||
        checked_blas_dim(coefficients.shape(1),
                         "protected rank-update coefficient columns") != n) {
        throw std::runtime_error(
            "Protected GxE rank-update operands have incompatible dimensions"
        );
    }
    {
        nb::gil_scoped_release release;
#if defined(GWLDCORE_GEMM_INTEGRITY)
        dgemm_nn_tiled(
            m, n, k,
            basis.data(), m,
            coefficients.data(), k,
            target.data(), m,
            requested_threads,
            -1.0, 1.0
        );
#else
        dgemm_nn_tiled(
            m, n, k,
            basis.data(), m,
            coefficients.data(), k,
            target.data(), m,
            requested_threads, -1.0, 1.0
        );
#endif
    }
}

#include "generalized_gxe_variant.inc"

}  // namespace

#include "contextual_dense_v1.inc"
#include "contextual_streamed_reference_v1.inc"

NB_MODULE(gxeldcore, module) {
    module.doc() = "Bounded double-precision native context with guarded, observable GxE GEMMs";
    module.attr("__version__") = "1.7";
    summit::context_v1::bind_contextual_dense_v1(module);
    summit::context_v1::bind_contextual_reference_executor_v1(module);
    nb::class_<GeneralizedGxELDScoreDirectContext>(
        module, "GeneralizedGxELDScoreDirectContext"
    )
        .def(
            nb::init<
                int, int, int, nb::object, int,
                nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                nb_mat2c_ro<double>, nb_vec1_ro<double>,
                nb_mat2c_ro<int64_t>, nb_mat2c_ro<int64_t>,
                nb_vec1_ro<int64_t>,
                nb_mat2c_ro<int64_t>, uint64_t,
                uint64_t, int64_t, int, int, int, int, int, uint64_t,
                int, int, bool, bool, const std::string&, int, int, double
            >(),
            nb::arg("bed_descriptor"), nb::arg("bim_descriptor"),
            nb::arg("fam_descriptor"), nb::arg("row_sel"),
            nb::arg("ddof"), nb::arg("basis"),
            nb::arg("fixed_effect_basis"), nb::arg("annotations"),
            nb::arg("annotation_masses"), nb::arg("pair_table"),
            nb::arg("component_table"),
            nb::arg("product_offsets"), nb::arg("product_terms"),
            nb::arg("root_seed"),
            nb::arg("namespace_key"), nb::arg("probe_offset"),
            nb::arg("probe_count"), nb::arg("variant_block_width"),
            nb::arg("source_probe_tile_width"),
            nb::arg("probe_tile_width"),
            nb::arg("same_person_sample_tile_width"),
            nb::arg("max_workspace_bytes"), nb::arg("decode_threads"),
            nb::arg("threads"), nb::arg("retain_base_sources") = false,
            nb::arg("dense_blas_hybrid") = true,
            nb::arg("qualification_fault_phase") = "none",
            nb::arg("qualification_fault_row") = -1,
            nb::arg("qualification_fault_column") = -1,
            nb::arg("qualification_fault_delta") = 0.0
        )
        .def("info", &GeneralizedGxELDScoreDirectContext::info)
        .def("run", &GeneralizedGxELDScoreDirectContext::run);
    module.def(
        "configure_openmp_placement", &configure_openmp_placement,
        nb::arg("expected_cpu_ids"), nb::arg("threads"),
        "Freeze and verify the exact singleton-place OpenMP worker placement."
    );
    module.def("configure_blas_threads", [](int requested_threads) {
        if (requested_threads <= 0) {
            throw std::runtime_error(
                "The GxE BLAS thread count must be positive"
            );
        }
#ifdef GWLDCORE_USE_FIXED_VENDOR_BLAS
        return configure_fixed_vendor_threads(requested_threads);
#else
        return requested_threads;
#endif
    });
    module.def("configured_blas_threads", []() {
#ifdef GWLDCORE_USE_OPENBLAS
        return fixed_openblas_runtime().threads.load(
            std::memory_order_acquire
        );
#else
        return 0;
#endif
    });
    module.def("build_info", []() {
        nb::dict result;
        result["backend_name"] = "gxeldcore_direct";
        result["backend_version"] = "1.9";
        result["api_version"] = 9;
        result["source_commit"] = GWLDCORE_SOURCE_COMMIT;
        result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
        result["compiler_id"] = GWLDCORE_COMPILER_ID;
        result["compiler_version"] = GWLDCORE_COMPILER_VERSION;
        result["build_type"] = GWLDCORE_BUILD_TYPE;
        result["blas_vendor"] = GWLDCORE_BLAS_VENDOR;
        result["cxx_standard"] = 17;
        result["sanitizer_mode"] = GWLDCORE_SANITIZER_MODE;
        result["address_sanitizer_enabled"] = bool(GWLDCORE_ASAN_ENABLED);
        result["undefined_behavior_sanitizer_enabled"] =
            bool(GWLDCORE_UBSAN_ENABLED);
        result["optimization"] = GWLDCORE_EFFECTIVE_OPTIMIZATION;
        result["architecture_tuning"] = GWLDCORE_ARCHITECTURE_TUNING;
        result["compiler_flags"] = GWLDCORE_CONFIGURED_COMPILER_FLAGS;
        result["compiler_flags_scope"] =
            "cmake_global_configuration_plus_summit_target_options_v1";
        result["openmp_enabled"] = bool(GWLDCORE_OPENMP_ENABLED);
        result["native_optimization_enabled"] = bool(GWLDCORE_NATIVE_OPT);
        result["platform"] = "linux";
        result["protected_pair_input_mode"] = "mprotect_read_only";
        result["gemm_integrity_enabled"] = bool(GWLDCORE_GEMM_INTEGRITY_ENABLED);
        result["gemm_checksum_enabled"] = bool(GWLDCORE_GEMM_CHECKSUM_ENABLED);
#if defined(GWLDCORE_GEMM_INTEGRITY)
        result["gemm_integrity_minimum_vendor_flops"] =
            kCheckedGemmMinimumFlops;
#else
        result["gemm_integrity_minimum_vendor_flops"] = 0;
#endif
        result["blas_runtime_isolation"] = (
            bool(GWLDCORE_PRIVATE_BLAS_ENABLED)
                ? "private_static" : "process_shared"
        );
        result["private_openblas_archive_sha256"] =
            GWLDCORE_PRIVATE_OPENBLAS_SHA256;
        result["private_blas_backend"] = GWLDCORE_PRIVATE_BLAS_BACKEND;
        result["private_blas_archive_sha256"] =
            GWLDCORE_PRIVATE_BLAS_SHA256;
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
        result["gemm_telemetry_schema_version"] = 1;
        result["gemm_telemetry_capacity"] = kGemmTelemetryCapacity;
        result["gemm_vendor_entry_outer_openmp_guard"] = true;
        append_openmp_placement_build_info(result);
        result["gemm_operand_numa_sampling_method"] =
            "move_pages_query_no_migration";
        result["gemm_operand_numa_sample_limit_per_operand"] =
            kNumaPageSamplesPerOperand;
        result["gemm_operand_numa_address_selection_schema_version"] =
            kNumaAddressSelectionSchemaVersion;
        result["gemm_operand_numa_address_selection_policy"] =
            kNumaAddressSelectionPolicy;
        result["gemm_operand_numa_partial_boundary_pages_included"] = false;
        result["native_integrity_snapshot_numa_contract_supported"] = true;
        result["native_integrity_snapshot_numa_contract_schema"] =
            kNativeIntegritySnapshotNumaSchema;
        result["native_integrity_snapshot_numa_query_chunk_page_limit"] =
            kNativeIntegritySnapshotQueryChunkPages;
        result["native_gemm_output_numa_contract_supported"] = true;
        result["native_gemm_output_numa_contract_schema"] =
            kNativeGemmOutputNumaSchema;
        result["native_gemm_output_numa_query_chunk_page_limit"] =
            kNativeGemmOutputQueryChunkPages;
        result["native_gemm_output_numa_evidence_capacity"] =
            kNativeGemmOutputEvidenceCapacity;
        result["multi_environment_native_kernel_supported"] = true;
        result["multi_environment_native_kernel_schema"] =
            "summit.multi_environment_native_kernel.v1";
        result["multi_environment_native_kernel_execution"] =
            "feature_source_projection_target_reduction_normalization";
        result["multi_environment_direct_context_supported"] = true;
        result["multi_environment_direct_context_schema"] =
            "summit.multi_environment_direct_context.v3";
        result["multi_environment_direct_context_execution"] =
            "descriptor_adaptive_dense_blas_or_packed_mailman_feature_source_target_projection_reduction_normalization";
        result["multi_environment_direct_context_max_vendor_probe_chunk"] = 0;
        result["multi_environment_direct_context_mailman_maximum_probes"] =
            kMultiEnvironmentMailmanMaximumProbes;
        result["global_variant_probe_supported"] = true;
        result["global_variant_probe_algorithm"] =
            "counter_global_variant_global_probe_v1";
        result["global_variant_probe_output_dtype"] = "float64";
        result["optimized_fp64_layout"] =
            "source_column_major_tt_target_row_major_tn_v1";
        result["optimized_fp64_row_pair_input_mode"] =
            "mprotect_read_only";
#ifdef GWLDCORE_USE_OPENBLAS
        const char* runtime_config = openblas_get_config();
        const char* runtime_corename = openblas_get_corename();
        result["blas_runtime_config"] = std::string(
            runtime_config == nullptr ? "" : runtime_config
        );
        result["blas_runtime_corename"] = std::string(
            runtime_corename == nullptr ? "" : runtime_corename
        );
        result["blas_runtime_threads"] = openblas_get_num_threads();
        const int parallel_model = openblas_get_parallel();
        result["blas_runtime_threading_layer"] = (
            parallel_model == 2 ? "openmp"
                : parallel_model == 1 ? "pthreads" : "serial"
        );
        result["gemm_execution_mode"] = (
            bool(GWLDCORE_PRIVATE_OPENBLAS_ENABLED)
                ? "serialized_fixed_private_openblas"
                : "serialized_fixed_shared_openblas"
        );
#elif defined(GWLDCORE_USE_BLIS)
        const BlisRequestedContract contract = blis_contract_for_build_info();
        result["blas_runtime_config"] = blis_runtime_config_string();
        result["blas_runtime_corename"] = blis_runtime_corename_string();
        result["blas_runtime_threads"] = contract.threads;
        result["blas_runtime_threading_layer"] = "pthreads";
        result["blas_runtime_thread_strategy"] =
            blis_thread_strategy_name(contract.strategy);
        nb::dict ways;
        ways["jc"] = contract.ways.jc;
        ways["pc"] = contract.ways.pc;
        ways["ic"] = contract.ways.ic;
        ways["jr"] = contract.ways.jr;
        ways["ir"] = contract.ways.ir;
        result["blas_runtime_thread_ways"] = std::move(ways);
        result["blas_runtime_owner_thread_enforced"] = true;
        result["blas_runtime_owner_thread_configured"] =
            blis_owner_thread_configured();
        result["blas_runtime_environment_immutable"] = true;
        result["blas_runtime_environment_contract"] =
            "blis_process_start_v1";
        result["blas_runtime_tls_enabled"] = true;
        result["blas_runtime_worker_affinity_policy"] =
            "inherit_authenticated_selected_cpu_set_per_call";
        result["gemm_execution_mode"] =
            "serialized_fixed_private_blis";
#else
        result["blas_runtime_config"] = nb::none();
        result["blas_runtime_corename"] = nb::none();
        result["gemm_execution_mode"] = "deterministic_tiled";
#endif
        return result;
    });
    module.def(
        "reset_gemm_telemetry", &reset_gemm_telemetry,
        "Clear GxE vendor telemetry and protected-output NUMA evidence counters."
    );
    module.def(
        "consume_gemm_telemetry", &consume_gemm_telemetry,
        "Atomically return and clear buffered GxE vendor-GEMM call records."
    );
    module.def(
        "get_gemm_telemetry", &get_gemm_telemetry,
        "Return a snapshot of buffered GxE vendor-GEMM call records."
    );
    module.def(
        "gemm_telemetry_status", &gemm_telemetry_status,
        "Return bounded-buffer and overflow status for GxE GEMM telemetry."
    );
    module.def(
        "reset_native_gemm_output_numa_evidence",
        &reset_native_gemm_output_numa_evidence,
        "Clear protected-GEMM output NUMA evidence and counters."
    );
    module.def(
        "consume_native_gemm_output_numa_evidence",
        &consume_native_gemm_output_numa_evidence,
        "Atomically return and clear completed protected-output NUMA evidence."
    );
    module.def(
        "get_native_gemm_output_numa_evidence",
        &get_native_gemm_output_numa_evidence,
        "Return a snapshot of completed protected-output NUMA evidence."
    );
    module.def(
        "native_gemm_output_numa_evidence_status",
        &native_gemm_output_numa_evidence_status,
        "Return protected-output NUMA evidence counters and buffer status."
    );
    module.def(
        "_test_vendor_entry_guard", &test_vendor_entry_guard,
        "Probe the direct predicate and production boundary without entering BLAS."
    );
    module.def(
        "_test_operand_numa_page_selection", &test_operand_numa_page_selection,
        nb::arg("operand_values"),
        "Expose boundary-safe NUMA page selection without querying or migrating pages."
    );
    module.def(
        "_test_native_integrity_snapshot_numa",
        &test_native_integrity_snapshot_numa,
        nb::arg("logical_byte_count"),
        "Exercise the native integrity snapshot allocation and verification contract."
    );
    module.def(
        "_test_native_integrity_snapshot_request_match",
        &test_native_integrity_snapshot_request_match,
        nb::arg("evidence_node"),
        "Verify that re-read policy nodes exactly match captured snapshot evidence."
    );
    module.def(
        "protected_matmul_nn", &protected_matmul_nn,
        nb::arg("left"), nb::arg("right"), nb::arg("threads"),
        "Compute left @ right with the fixed-runtime GxE GEMM executor."
    );
#if defined(GWLDCORE_GEMM_INTEGRITY)
    module.def(
        "_test_protected_matmul_nn_integrity_diagnostic",
        &test_protected_matmul_nn_integrity_diagnostic,
        nb::arg("left"), nb::arg("right"), nb::arg("threads"),
        nb::arg("fault_injection_row") = -1,
        nb::arg("fault_injection_column") = -1,
        nb::arg("fault_injection_delta") = 0.0,
        "Retain raw NN integrity-failure evidence without changing production routing."
    );
#endif
    module.def(
        "protected_matmul_tn", &protected_matmul_tn,
        nb::arg("left"), nb::arg("right"), nb::arg("threads"),
        "Compute left.T @ right with the fixed-runtime GxE GEMM executor."
    );
    module.def(
        "protected_matmul_tt_row_major_output",
        &protected_matmul_tt_row_major_output,
        nb::arg("weights"), nb::arg("genotype"), nb::arg("threads"),
        "Compute weights.T @ genotype.T and expose its row-major transpose view."
    );
    module.def(
        "protected_rank_update_nn", &protected_rank_update_nn,
        nb::arg("basis"), nb::arg("coefficients"), nb::arg("target"),
        nb::arg("threads"),
        "Apply target -= basis @ coefficients without a projection allocation."
    );
    module.def(
        "standardize_genotype_block", &standardize_genotype_block,
        nb::arg("genotype"), nb::arg("missingness_targets"),
        nb::arg("ddof"), nb::arg("hwe_scale"), nb::arg("eps"),
        nb::arg("threads"),
        "Mean-impute and standardize genotype columns in place with bounded diagnostics."
    );
    module.def(
        "standardize_genotype_block_row_major",
        &standardize_genotype_block_row_major,
        nb::arg("genotype"), nb::arg("missingness_targets"),
        nb::arg("ddof"), nb::arg("hwe_scale"), nb::arg("eps"),
        nb::arg("threads"),
        "Mean-impute and standardize a directly decoded row-major genotype block."
    );
    module.def(
        "fused_feature_scalar_moments", &fused_feature_scalar_moments,
        nb::arg("genotype"), nb::arg("environments"), nb::arg("threads"),
        "Compute shared G2, E*G2, E2*G2, and E4*G2 feature moments."
    );
    module.def(
        "numpy_philox_rademacher_block",
        &numpy_philox_rademacher_block,
        nb::arg("keys"), nb::arg("rows"), nb::arg("threads"),
        "Generate NumPy-compatible seeded Rademacher probes in native code."
    );
    module.def(
        "global_variant_rademacher",
        &global_variant_rademacher,
        nb::arg("variant_indices"), nb::arg("probe_indices"),
        nb::arg("root_seed"), nb::arg("namespace_key"),
        nb::arg("threads"),
        "Generate globally addressed generalized variant-axis Rademacher probes."
    );
    nb::class_<ProtectedRightPair>(module, "ProtectedRightPair")
        .def_prop_ro("rows", &ProtectedRightPair::rows)
        .def_prop_ro("columns", &ProtectedRightPair::columns);
    module.def(
        "prepare_protected_row_weighted_pair",
        &prepare_protected_row_weighted_pair,
        nb::arg("right"), nb::arg("row_weights"), nb::arg("threads"),
        "Seal an immutable [right, row_weight*right] target operand pair."
    );
    module.def(
        "protected_matmul_tn_pair", &protected_matmul_tn_pair,
        nb::arg("left"), nb::arg("right_pair"), nb::arg("threads"),
        "Compute left.T against both halves of a sealed protected operand pair."
    );
    nb::class_<MultiEnvironmentKernel>(module, "MultiEnvironmentKernel")
        .def(
            nb::init<
                nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                nb_vec1_ro<int64_t>, nb_vec1_ro<int64_t>,
                nb_vec1_ro<int32_t>, nb_vec1_ro<double>,
                nb_vec1_ro<int64_t>, nb_vec1_ro<double>,
                nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                int, int, int
            >(),
            nb::arg("environments"), nb::arg("feature_basis"),
            nb::arg("feature_power_indices"),
            nb::arg("feature_power_offsets"),
            nb::arg("feature_ranks"),
            nb::arg("feature_gram_values"),
            nb::arg("feature_gram_offsets"),
            nb::arg("feature_e2_gram_values"),
            nb::arg("common_basis"), nb::arg("directions"),
            nb::arg("ddof"), nb::arg("annotation_bins"),
            nb::arg("threads"),
            "Construct the persistent multi-environment native numerical kernel."
        )
        .def("info", &MultiEnvironmentKernel::info)
        .def(
            "feature_block", &MultiEnvironmentKernel::feature_block,
            nb::arg("genotype"), nb::arg("eps_var"),
            nb::arg("standardized")
        )
        .def(
            "source_block", &MultiEnvironmentKernel::source_block,
            nb::arg("target"), nb::arg("genotype"), nb::arg("probes"),
            nb::arg("annotation"), nb::arg("scale_x"),
            nb::arg("scale_w"), nb::arg("environment_start"),
            nb::arg("annotation_is_sqrt") = false
        )
        .def(
            "project_sources", &MultiEnvironmentKernel::project_sources,
            nb::arg("panel"), nb::arg("columns_per_environment"),
            nb::arg("environment_start")
        )
        .def(
            "target_score_block", &MultiEnvironmentKernel::target_score_block,
            nb::arg("genotype"), nb::arg("right_pair"),
            nb::arg("scale_x"), nb::arg("scale_w"),
            nb::arg("accum_xx"), nb::arg("accum_xw"),
            nb::arg("accum_wx"), nb::arg("accum_ww"),
            nb::arg("block_start"), nb::arg("total_variants"),
            nb::arg("probe_count"), nb::arg("environment_start")
        )
        .def(
            "normalize_scores", &MultiEnvironmentKernel::normalize_scores,
            nb::arg("accum_xx"), nb::arg("accum_xw"),
            nb::arg("accum_wx"), nb::arg("accum_ww"),
            nb::arg("probe_count")
        );
    nb::class_<MultiEnvironmentDirectContext>(
        module, "MultiEnvironmentDirectContext"
    )
        .def(
            nb::init<
                int, int, int, nb::object, nb_vec1_ro<double>,
                nb_mat2f_ro<double>, int, uint64_t,
                nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                nb_vec1_ro<int64_t>, nb_vec1_ro<int64_t>,
                nb_vec1_ro<int32_t>, nb_vec1_ro<double>,
                nb_vec1_ro<int64_t>, nb_vec1_ro<double>,
                nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                nb_mat2c_ro<double>, nb_vec1_ro<double>,
                nb_mat2c_ro<int64_t>, nb_mat2c_ro<int64_t>,
                nb_mat2c_ro<int64_t>, nb_mat2c_ro<uint64_t>,
                int, int, double, bool, bool, int
            >(),
            nb::arg("bed_descriptor"), nb::arg("bim_descriptor"),
            nb::arg("fam_descriptor"), nb::arg("row_sel"),
            nb::arg("reader_environment"), nb::arg("reader_q_basis"),
            nb::arg("decode_threads"), nb::arg("max_workspace_bytes"),
            nb::arg("environments"), nb::arg("feature_basis"),
            nb::arg("feature_power_indices"),
            nb::arg("feature_power_offsets"), nb::arg("feature_ranks"),
            nb::arg("feature_gram_values"),
            nb::arg("feature_gram_offsets"),
            nb::arg("feature_e2_gram_values"), nb::arg("common_basis"),
            nb::arg("directions"), nb::arg("annotations"),
            nb::arg("annotation_masses"), nb::arg("blocks"),
            nb::arg("environment_tiles"), nb::arg("probe_tiles"),
            nb::arg("philox_keys"), nb::arg("ddof"),
            nb::arg("total_probes"), nb::arg("eps_var"),
            nb::arg("standardized"), nb::arg("dense_blas_hybrid"),
            nb::arg("threads"),
            "Construct a single-use descriptor-owned multi-environment executor."
        )
        .def("info", &MultiEnvironmentDirectContext::info)
        .def(
            "release_execution_scratch",
            &MultiEnvironmentDirectContext::release_execution_scratch,
            "Release execution-only native scratch after the completed run; "
            "idempotent and required before publication."
        )
        .def("run", &MultiEnvironmentDirectContext::run);
    module.def(
        "project_protected_row_major_sources",
        &project_protected_row_major_sources,
        nb::arg("panel"), nb::arg("common_basis"), nb::arg("directions"),
        nb::arg("columns_per_environment"), nb::arg("threads"),
        "Project a row-major packed source panel in place without shared BLAS."
    );
    nb::class_<ProtectedRowMajorPair>(module, "ProtectedRowMajorPair")
        .def_prop_ro("rows", &ProtectedRowMajorPair::rows)
        .def_prop_ro("columns", &ProtectedRowMajorPair::columns);
    module.def(
        "prepare_protected_row_major_weighted_pair",
        &prepare_protected_row_major_weighted_pair,
        nb::arg("right"), nb::arg("row_weights"), nb::arg("threads"),
        "Seal an immutable row-major [right, row_weight*right] target pair."
    );
    module.def(
        "protected_matmul_row_major_tn_pair",
        &protected_matmul_row_major_tn_pair,
        nb::arg("genotype"), nb::arg("right_pair"), nb::arg("threads"),
        "Compute row-major genotype.T against a sealed row-major target pair."
    );
    nb::class_<ProjectedPanel>(module, "ProjectedPanel")
        .def_prop_ro("columns", &ProjectedPanel::columns)
        .def_prop_ro("leakage", &ProjectedPanel::leakage);
    nb::class_<DirectContext>(module, "DirectContext")
        .def(
            nb::init<int, int, int, nb::object, int, nb_vec1_ro<double>, nb_mat2f_ro<double>, int, uint64_t, int, bool, int>(),
            nb::arg("bed_descriptor"), nb::arg("bim_descriptor"), nb::arg("fam_descriptor"),
            nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1,
            nb::arg("env"), nb::arg("q_basis"), nb::arg("decode_threads"),
            nb::arg("max_workspace_bytes"), nb::arg("target_panel_columns") = 64,
            nb::arg("strict_feature_moment_verification") = true,
            nb::arg("blas_threads") = 0
        )
        .def("close", &DirectContext::close)
        .def("info", &DirectContext::info)
        .def(
            "feature_block", &DirectContext::feature_block,
            nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("eps_var") = 1.0e-10,
            nb::arg("require_missing_free") = true
        )
        .def(
            "source_block", &DirectContext::source_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"), nb::arg("sqrt_annotation"),
            nb::arg("probes"),
            nb::arg("require_missing_free") = true
        )
        .def(
            "prepare_projected_sources", &DirectContext::prepare_projected_sources,
            nb::arg("sources"), nb::arg("tolerance") = 1.0e-10
        )
        .def(
            "validate_projected_sources", &DirectContext::validate_projected_sources,
            nb::arg("sources"), nb::arg("tolerance") = 1.0e-10
        )
        .def(
            "target_block", &DirectContext::target_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"), nb::arg("sources"),
            nb::arg("require_missing_free") = true
        )
        .def(
            "target_projected_block", &DirectContext::target_projected_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"), nb::arg("sources"),
            nb::arg("require_missing_free") = true
        )
        .def(
            "phenotype_score_block", &DirectContext::phenotype_score_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("phenotype"), nb::arg("eps_var") = 1.0e-10,
            nb::arg("require_missing_free") = false
        );
}
