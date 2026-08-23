#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/types.h>
#include <sched.h>
#include <time.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

// This executable is intentionally a standalone process.  The launcher links
// exactly one static OpenBLAS archive into it and verifies that the resulting
// executable has no dynamic BLAS dependency.  Keeping the benchmark out of the
// Python process prevents NumPy's BLAS state from participating in a run.
extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
int openblas_get_num_procs(void);
int openblas_get_parallel(void);
char* openblas_get_config(void);
char* openblas_get_corename(void);
}

// The accepted FP64 production archive was intentionally built without the
// public single-precision CBLAS interface.  The launcher defines
// SUMMIT_OPENBLAS_HAS_SGEMM only after finding cblas_sgemm in a candidate
// archive.  Conditional compilation is important here: leaving an unresolved
// weak cblas_sgemm would permit accidental interposition from a process-shared
// BLAS and would defeat the private-runtime boundary.

#ifndef SUMMIT_OPENBLAS_ARCHIVE
#define SUMMIT_OPENBLAS_ARCHIVE "unknown"
#endif
#ifndef SUMMIT_OPENBLAS_ARCHIVE_SHA256
#define SUMMIT_OPENBLAS_ARCHIVE_SHA256 "unknown"
#endif
#ifndef SUMMIT_BENCHMARK_SOURCE_SHA256
#define SUMMIT_BENCHMARK_SOURCE_SHA256 "unknown"
#endif

namespace {

using steady_clock = std::chrono::steady_clock;

struct options {
    std::string mode = "square";
    std::string dtype = "f64";
    std::string layout = "col";
    std::string orientation = "current";
    int size = 512;
    int n_samples = 2048;
    int block_width = 256;
    int probe_tile = 4;
    int environment_tile = 1;
    std::size_t stream_elements = 4U * 1024U * 1024U;
    int threads = 1;
    int warmups = 3;
    int repeats = 5;
    int correctness_samples = 12;
    std::uint64_t seed = 20260816ULL;
};

struct entry_snapshot {
    int omp_in_parallel = 0;
    int omp_level = 0;
    int omp_active_level = 0;
    int omp_max_active_levels = 0;
    int omp_max_threads = 0;
    int openblas_threads = 0;
    int cpu_before = -1;
    int cpu_after = -1;
};

struct measurement {
    double wall_seconds = 0.0;
    double process_cpu_seconds = 0.0;
    double active_core_equivalents = 0.0;
    double gflops_per_second = 0.0;
    double bandwidth_gbytes_per_second = 0.0;
    double process_cpu_core_minutes = 0.0;
    std::string sampled_fingerprint;
    entry_snapshot entry;
    int observed_omp_team_size = 1;
};

struct correctness_result {
    int sample_count = 0;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double max_error_to_roundoff_bound = 0.0;
    bool finite = true;
    bool within_roundoff_bound = true;
    bool repeated_outputs_bitwise_equal_at_samples = true;
};

struct process_counters {
    long major_faults = 0;
    long minor_faults = 0;
    long peak_rss_kib = 0;
};

std::mutex vendor_entry_mutex;

std::string json_escape(const std::string& value) {
    std::ostringstream stream;
    for (const unsigned char character : value) {
        switch (character) {
            case '\"': stream << "\\\""; break;
            case '\\': stream << "\\\\"; break;
            case '\b': stream << "\\b"; break;
            case '\f': stream << "\\f"; break;
            case '\n': stream << "\\n"; break;
            case '\r': stream << "\\r"; break;
            case '\t': stream << "\\t"; break;
            default:
                if (character < 0x20U) {
                    stream << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    stream << character;
                }
        }
    }
    return stream.str();
}

std::string quote(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) return "";
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string status_field(const std::string& label) {
    std::istringstream status(read_text_file("/proc/self/status"));
    std::string line;
    const std::string prefix = label + ":";
    while (std::getline(status, line)) {
        if (line.rfind(prefix, 0) != 0) continue;
        std::string value = line.substr(prefix.size());
        const auto first = value.find_first_not_of(" \t");
        return first == std::string::npos ? "" : value.substr(first);
    }
    return "";
}

std::vector<int> affinity_cpus() {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        throw std::runtime_error(
            "sched_getaffinity failed: " + std::string(std::strerror(errno))
        );
    }
    std::vector<int> result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &cpu_set)) result.push_back(cpu);
    }
    return result;
}

int numa_node_for_cpu(int cpu) {
    if (cpu < 0) return -1;
    const std::filesystem::path base(
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
    );
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(base, error)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("node", 0) != 0 || name.size() <= 4) continue;
        try {
            return std::stoi(name.substr(4));
        } catch (const std::exception&) {
            continue;
        }
    }
    return -1;
}

std::map<int, std::uint64_t> resident_numa_pages() {
    std::istringstream maps(read_text_file("/proc/self/numa_maps"));
    std::map<int, std::uint64_t> totals;
    std::string line;
    while (std::getline(maps, line)) {
        std::istringstream tokens(line);
        std::string token;
        while (tokens >> token) {
            if (token.size() < 4 || token[0] != 'N') continue;
            const std::size_t equals = token.find('=');
            if (equals == std::string::npos || equals <= 1) continue;
            bool node_is_numeric = true;
            for (std::size_t index = 1; index < equals; ++index) {
                if (token[index] < '0' || token[index] > '9') {
                    node_is_numeric = false;
                    break;
                }
            }
            if (!node_is_numeric) continue;
            try {
                const int node = std::stoi(token.substr(1, equals - 1));
                const std::uint64_t pages = std::stoull(token.substr(equals + 1));
                totals[node] += pages;
            } catch (const std::exception&) {
                continue;
            }
        }
    }
    return totals;
}

process_counters read_process_counters() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return {usage.ru_majflt, usage.ru_minflt, usage.ru_maxrss};
}

double process_cpu_seconds() {
    struct timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
        throw std::runtime_error("CLOCK_PROCESS_CPUTIME_ID failed");
    }
    return static_cast<double>(value.tv_sec)
        + static_cast<double>(value.tv_nsec) * 1.0e-9;
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    struct tm utc {};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

template <typename T>
T frozen_value(
    std::size_t row, std::size_t column, std::uint64_t stream,
    std::uint64_t seed
) {
    const std::uint64_t row_bits = splitmix64(row + 0x632be59bd9b4e019ULL);
    const std::uint64_t column_bits = splitmix64(column + 0x8cb92ba72f3d8dd7ULL);
    const std::uint64_t bits = splitmix64(seed ^ stream ^ row_bits ^ (column_bits << 1U));
    const double unit = static_cast<double>(bits >> 11U) / 9007199254740992.0;
    return static_cast<T>(unit - 0.5);
}

std::size_t checked_elements(std::size_t rows, std::size_t columns) {
    if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::overflow_error("matrix element count overflows size_t");
    }
    return rows * columns;
}

template <typename T>
class matrix {
public:
    matrix(int rows, int columns, CBLAS_LAYOUT layout)
        : rows_(rows), columns_(columns), layout_(layout),
          values_(checked_elements(
              static_cast<std::size_t>(rows), static_cast<std::size_t>(columns)
          )) {
        if (rows <= 0 || columns <= 0) {
            throw std::invalid_argument("matrix dimensions must be positive");
        }
    }

    int rows() const { return rows_; }
    int columns() const { return columns_; }
    int leading_dimension() const {
        return layout_ == CblasColMajor ? rows_ : columns_;
    }
    CBLAS_LAYOUT layout() const { return layout_; }
    std::size_t size() const { return values_.size(); }
    T* data() { return values_.data(); }
    const T* data() const { return values_.data(); }

    std::size_t offset(int row, int column) const {
        return layout_ == CblasColMajor
            ? static_cast<std::size_t>(column) * rows_ + row
            : static_cast<std::size_t>(row) * columns_ + column;
    }
    T& at(int row, int column) { return values_[offset(row, column)]; }
    const T& at(int row, int column) const { return values_[offset(row, column)]; }

    template <typename Function>
    void fill_parallel(Function&& function) {
        const std::int64_t count = static_cast<std::int64_t>(values_.size());
        #pragma omp parallel for schedule(static)
        for (std::int64_t linear = 0; linear < count; ++linear) {
            int row = 0;
            int column = 0;
            if (layout_ == CblasColMajor) {
                row = static_cast<int>(linear % rows_);
                column = static_cast<int>(linear / rows_);
            } else {
                row = static_cast<int>(linear / columns_);
                column = static_cast<int>(linear % columns_);
            }
            values_[static_cast<std::size_t>(linear)] = function(row, column);
        }
    }

private:
    int rows_;
    int columns_;
    CBLAS_LAYOUT layout_;
    std::vector<T> values_;
};

struct gemm_descriptor {
    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_TRANSPOSE transpose_a = CblasNoTrans;
    CBLAS_TRANSPOSE transpose_b = CblasNoTrans;
    int m = 0;
    int n = 0;
    int k = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
};

std::string layout_name(CBLAS_LAYOUT layout) {
    return layout == CblasColMajor ? "CblasColMajor" : "CblasRowMajor";
}

std::string transpose_name(CBLAS_TRANSPOSE transpose) {
    if (transpose == CblasNoTrans) return "CblasNoTrans";
    if (transpose == CblasTrans) return "CblasTrans";
    return "CblasConjTrans";
}

entry_snapshot capture_vendor_entry() {
    entry_snapshot snapshot;
    snapshot.omp_in_parallel = omp_in_parallel();
    snapshot.omp_level = omp_get_level();
    snapshot.omp_active_level = omp_get_active_level();
    snapshot.omp_max_active_levels = omp_get_max_active_levels();
    snapshot.omp_max_threads = omp_get_max_threads();
    snapshot.openblas_threads = openblas_get_num_threads();
    snapshot.cpu_before = sched_getcpu();
    if (snapshot.omp_in_parallel != 0 || snapshot.omp_level != 0
        || snapshot.omp_active_level != 0) {
        throw std::runtime_error(
            "refusing vendor GEMM entry from an active OpenMP region"
        );
    }
    return snapshot;
}

template <typename T>
void gemm_call(
    const gemm_descriptor& descriptor, const T* a, const T* b, T* c
) {
    if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(
            descriptor.layout, descriptor.transpose_a, descriptor.transpose_b,
            descriptor.m, descriptor.n, descriptor.k,
            1.0, a, descriptor.lda, b, descriptor.ldb,
            0.0, c, descriptor.ldc
        );
    } else {
#ifdef SUMMIT_OPENBLAS_HAS_SGEMM
        cblas_sgemm(
            descriptor.layout, descriptor.transpose_a, descriptor.transpose_b,
            descriptor.m, descriptor.n, descriptor.k,
            1.0F, a, descriptor.lda, b, descriptor.ldb,
            0.0F, c, descriptor.ldc
        );
#else
        (void)descriptor;
        (void)a;
        (void)b;
        (void)c;
        throw std::runtime_error(
            "this private OpenBLAS archive does not export cblas_sgemm"
        );
#endif
    }
}

template <typename T>
measurement timed_gemm(
    const gemm_descriptor& descriptor, const T* a, const T* b, T* c
) {
    if (!vendor_entry_mutex.try_lock()) {
        throw std::runtime_error("concurrent vendor entry was attempted");
    }
    std::unique_lock<std::mutex> lock(vendor_entry_mutex, std::adopt_lock);
    measurement result;
    result.entry = capture_vendor_entry();
    const double cpu_start = process_cpu_seconds();
    const auto wall_start = steady_clock::now();
    gemm_call(descriptor, a, b, c);
    const auto wall_stop = steady_clock::now();
    const double cpu_stop = process_cpu_seconds();
    result.entry.cpu_after = sched_getcpu();
    result.wall_seconds = std::chrono::duration<double>(wall_stop - wall_start).count();
    result.process_cpu_seconds = cpu_stop - cpu_start;
    result.active_core_equivalents = result.wall_seconds > 0.0
        ? result.process_cpu_seconds / result.wall_seconds : 0.0;
    const long double operations = 2.0L * descriptor.m * descriptor.n * descriptor.k;
    result.gflops_per_second = result.wall_seconds > 0.0
        ? static_cast<double>(operations / result.wall_seconds / 1.0e9L) : 0.0;
    result.process_cpu_core_minutes = result.process_cpu_seconds / 60.0;
    return result;
}

std::vector<std::size_t> sample_indices(std::size_t size, int requested) {
    if (size == 0 || requested <= 0) return {};
    const std::size_t count = std::min<std::size_t>(
        size, static_cast<std::size_t>(requested)
    );
    std::set<std::size_t> unique;
    unique.insert(0);
    unique.insert(size - 1);
    for (std::size_t index = 0; unique.size() < count; ++index) {
        unique.insert(static_cast<std::size_t>(
            splitmix64(0x6a09e667f3bcc909ULL + index) % size
        ));
    }
    return {unique.begin(), unique.end()};
}

template <typename T>
std::string sampled_fingerprint(const matrix<T>& values, int samples = 4096) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::size_t index : sample_indices(values.size(), samples)) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, values.data() + index, sizeof(T));
        hash ^= splitmix64(bits ^ index);
        hash *= 1099511628211ULL;
    }
    std::ostringstream rendered;
    rendered << std::hex << std::setfill('0') << std::setw(16) << hash;
    return rendered.str();
}

template <typename T, typename Oracle>
correctness_result check_output(
    const matrix<T>& output, int requested_samples, int reduction_length,
    Oracle&& oracle, const std::vector<std::string>& fingerprints
) {
    correctness_result result;
    const std::vector<std::size_t> indices = sample_indices(
        output.size(), requested_samples
    );
    result.sample_count = static_cast<int>(indices.size());
    const long double epsilon = std::numeric_limits<T>::epsilon();
    const long double k_epsilon = reduction_length * epsilon;
    const long double gamma = k_epsilon < 1.0L
        ? k_epsilon / (1.0L - k_epsilon)
        : std::numeric_limits<long double>::infinity();
    for (const std::size_t linear : indices) {
        int logical_row = 0;
        int logical_column = 0;
        if (output.layout() == CblasColMajor) {
            logical_row = static_cast<int>(linear % output.rows());
            logical_column = static_cast<int>(linear / output.rows());
        } else {
            logical_row = static_cast<int>(linear / output.columns());
            logical_column = static_cast<int>(linear % output.columns());
        }
        const auto expected_and_absolute_sum = oracle(logical_row, logical_column);
        const long double expected = expected_and_absolute_sum.first;
        const long double absolute_sum = expected_and_absolute_sum.second;
        const long double observed = output.at(logical_row, logical_column);
        const long double absolute_error = std::abs(observed - expected);
        const long double relative_error = absolute_error
            / std::max(std::abs(expected), std::numeric_limits<long double>::min());
        const long double bound = gamma * absolute_sum
            + 4.0L * epsilon * std::abs(expected)
            + 4.0L * std::numeric_limits<T>::min();
        result.max_absolute_error = std::max(
            result.max_absolute_error, static_cast<double>(absolute_error)
        );
        result.max_relative_error = std::max(
            result.max_relative_error, static_cast<double>(relative_error)
        );
        if (bound > 0.0L && std::isfinite(static_cast<double>(bound))) {
            result.max_error_to_roundoff_bound = std::max(
                result.max_error_to_roundoff_bound,
                static_cast<double>(absolute_error / bound)
            );
        }
        if (!std::isfinite(static_cast<double>(observed))) result.finite = false;
        if (!(absolute_error <= bound)) result.within_roundoff_bound = false;
    }
    if (!fingerprints.empty()) {
        result.repeated_outputs_bitwise_equal_at_samples = std::all_of(
            fingerprints.begin() + 1, fingerprints.end(),
            [&](const std::string& value) { return value == fingerprints.front(); }
        );
    }
    return result;
}

template <typename T>
struct gemm_case {
    gemm_descriptor descriptor;
    matrix<T> a;
    matrix<T> b;
    matrix<T> output;
    std::string conceptual_product;

    gemm_case(
        gemm_descriptor descriptor_value, matrix<T>&& a_value,
        matrix<T>&& b_value, matrix<T>&& output_value,
        std::string conceptual_product_value
    ) : descriptor(descriptor_value), a(std::move(a_value)), b(std::move(b_value)),
        output(std::move(output_value)),
        conceptual_product(std::move(conceptual_product_value)) {}
};

template <typename T>
gemm_case<T> make_square_case(const options& settings, CBLAS_LAYOUT layout) {
    matrix<T> a(settings.size, settings.size, layout);
    matrix<T> b(settings.size, settings.size, layout);
    matrix<T> output(settings.size, settings.size, layout);
    a.fill_parallel([&](int row, int column) {
        return frozen_value<T>(row, column, 0x243f6a8885a308d3ULL, settings.seed);
    });
    b.fill_parallel([&](int row, int column) {
        return frozen_value<T>(row, column, 0x13198a2e03707344ULL, settings.seed);
    });
    output.fill_parallel([](int, int) { return T(0); });
    gemm_descriptor descriptor {
        layout, CblasNoTrans, CblasNoTrans,
        settings.size, settings.size, settings.size,
        a.leading_dimension(), b.leading_dimension(), output.leading_dimension()
    };
    return gemm_case<T>(
        descriptor, std::move(a), std::move(b), std::move(output), "A @ B"
    );
}

template <typename T>
correctness_result square_correctness(
    const gemm_case<T>& work, int samples,
    const std::vector<std::string>& fingerprints
) {
    return check_output(
        work.output, samples, work.descriptor.k,
        [&](int row, int column) {
            long double sum = 0.0L;
            long double absolute_sum = 0.0L;
            for (int inner = 0; inner < work.descriptor.k; ++inner) {
                const long double product = static_cast<long double>(work.a.at(row, inner))
                    * static_cast<long double>(work.b.at(inner, column));
                sum += product;
                absolute_sum += std::abs(product);
            }
            return std::make_pair(sum, absolute_sum);
        },
        fingerprints
    );
}

template <typename T>
struct exact_case {
    gemm_descriptor descriptor;
    matrix<T> left;
    matrix<T> right;
    matrix<T> output;
    std::string conceptual_product;
    int logical_panel_width = 0;

    exact_case(
        gemm_descriptor descriptor_value, matrix<T>&& left_value,
        matrix<T>&& right_value, matrix<T>&& output_value,
        std::string conceptual_product_value, int panel_width
    ) : descriptor(descriptor_value), left(std::move(left_value)),
        right(std::move(right_value)), output(std::move(output_value)),
        conceptual_product(std::move(conceptual_product_value)),
        logical_panel_width(panel_width) {}
};

template <typename T>
exact_case<T> make_source_case(const options& settings, CBLAS_LAYOUT layout) {
    const int panel = 2 * settings.probe_tile * settings.environment_tile;
    matrix<T> genotype(settings.n_samples, settings.block_width, layout);
    matrix<T> weights(settings.block_width, panel, layout);
    genotype.fill_parallel([&](int row, int column) {
        return frozen_value<T>(row, column, 0xa4093822299f31d0ULL, settings.seed);
    });
    weights.fill_parallel([&](int row, int column) {
        return frozen_value<T>(row, column, 0x082efa98ec4e6c89ULL, settings.seed);
    });
    if (settings.orientation == "current") {
        matrix<T> output(settings.n_samples, panel, layout);
        output.fill_parallel([](int, int) { return T(0); });
        const gemm_descriptor descriptor {
            layout, CblasNoTrans, CblasNoTrans,
            settings.n_samples, panel, settings.block_width,
            genotype.leading_dimension(), weights.leading_dimension(),
            output.leading_dimension()
        };
        return exact_case<T>(
            descriptor, std::move(genotype), std::move(weights),
            std::move(output), "C = G @ W", panel
        );
    }
    matrix<T> output(panel, settings.n_samples, layout);
    output.fill_parallel([](int, int) { return T(0); });
    const gemm_descriptor descriptor {
        layout, CblasTrans, CblasTrans,
        panel, settings.n_samples, settings.block_width,
        weights.leading_dimension(), genotype.leading_dimension(),
        output.leading_dimension()
    };
    // Move W into A and G into B.  Both were populated directly in the chosen
    // storage layout; no O(NM) transpose or layout-conversion buffer exists.
    return exact_case<T>(
        descriptor, std::move(weights), std::move(genotype),
        std::move(output), "C' = W' @ G'", panel
    );
}

template <typename T>
correctness_result source_correctness(
    const exact_case<T>& work, const options& settings,
    const std::vector<std::string>& fingerprints
) {
    return check_output(
        work.output, settings.correctness_samples, settings.block_width,
        [&](int output_row, int output_column) {
            const int sample = settings.orientation == "current"
                ? output_row : output_column;
            const int panel_column = settings.orientation == "current"
                ? output_column : output_row;
            long double sum = 0.0L;
            long double absolute_sum = 0.0L;
            for (int variant = 0; variant < settings.block_width; ++variant) {
                const T genotype = settings.orientation == "current"
                    ? work.left.at(sample, variant)
                    : work.right.at(sample, variant);
                const T weight = settings.orientation == "current"
                    ? work.right.at(variant, panel_column)
                    : work.left.at(variant, panel_column);
                const long double product = static_cast<long double>(genotype)
                    * static_cast<long double>(weight);
                sum += product;
                absolute_sum += std::abs(product);
            }
            return std::make_pair(sum, absolute_sum);
        },
        fingerprints
    );
}

template <typename T>
exact_case<T> make_target_case(const options& settings, CBLAS_LAYOUT layout) {
    const int panel = 4 * settings.probe_tile * settings.environment_tile;
    matrix<T> genotype(settings.n_samples, settings.block_width, layout);
    matrix<T> sources(settings.n_samples, panel, layout);
    genotype.fill_parallel([&](int row, int column) {
        return frozen_value<T>(row, column, 0xa4093822299f31d0ULL, settings.seed);
    });
    sources.fill_parallel([&](int row, int column) {
        return frozen_value<T>(row, column, 0x452821e638d01377ULL, settings.seed);
    });
    if (settings.orientation == "current") {
        matrix<T> output(settings.block_width, panel, layout);
        output.fill_parallel([](int, int) { return T(0); });
        const gemm_descriptor descriptor {
            layout, CblasTrans, CblasNoTrans,
            settings.block_width, panel, settings.n_samples,
            genotype.leading_dimension(), sources.leading_dimension(),
            output.leading_dimension()
        };
        return exact_case<T>(
            descriptor, std::move(genotype), std::move(sources),
            std::move(output), "C = G' @ S", panel
        );
    }
    matrix<T> output(panel, settings.block_width, layout);
    output.fill_parallel([](int, int) { return T(0); });
    const gemm_descriptor descriptor {
        layout, CblasTrans, CblasNoTrans,
        panel, settings.block_width, settings.n_samples,
        sources.leading_dimension(), genotype.leading_dimension(),
        output.leading_dimension()
    };
    return exact_case<T>(
        descriptor, std::move(sources), std::move(genotype),
        std::move(output), "C' = S' @ G", panel
    );
}

template <typename T>
correctness_result target_correctness(
    const exact_case<T>& work, const options& settings,
    const std::vector<std::string>& fingerprints
) {
    return check_output(
        work.output, settings.correctness_samples, settings.n_samples,
        [&](int output_row, int output_column) {
            const int variant = settings.orientation == "current"
                ? output_row : output_column;
            const int panel_column = settings.orientation == "current"
                ? output_column : output_row;
            long double sum = 0.0L;
            long double absolute_sum = 0.0L;
            for (int sample = 0; sample < settings.n_samples; ++sample) {
                const T genotype = settings.orientation == "current"
                    ? work.left.at(sample, variant)
                    : work.right.at(sample, variant);
                const T source = settings.orientation == "current"
                    ? work.right.at(sample, panel_column)
                    : work.left.at(sample, panel_column);
                const long double product = static_cast<long double>(genotype)
                    * static_cast<long double>(source);
                sum += product;
                absolute_sum += std::abs(product);
            }
            return std::make_pair(sum, absolute_sum);
        },
        fingerprints
    );
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2U != 0U) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

template <typename T, typename Case, typename Correctness>
std::pair<std::vector<measurement>, correctness_result> run_gemm_case(
    Case& work, const options& settings, Correctness&& correctness
) {
    std::vector<std::string> fingerprints;
    fingerprints.reserve(static_cast<std::size_t>(settings.warmups + settings.repeats));
    for (int repeat = 0; repeat < settings.warmups; ++repeat) {
        (void)timed_gemm(
            work.descriptor, work.left.data(), work.right.data(), work.output.data()
        );
        fingerprints.push_back(sampled_fingerprint(work.output));
    }
    std::vector<measurement> measurements;
    measurements.reserve(static_cast<std::size_t>(settings.repeats));
    for (int repeat = 0; repeat < settings.repeats; ++repeat) {
        measurement value = timed_gemm(
            work.descriptor, work.left.data(), work.right.data(), work.output.data()
        );
        value.sampled_fingerprint = sampled_fingerprint(work.output);
        fingerprints.push_back(value.sampled_fingerprint);
        measurements.push_back(std::move(value));
    }
    return {measurements, correctness(work, settings, fingerprints)};
}

template <typename T>
std::pair<std::vector<measurement>, correctness_result> run_square(
    gemm_case<T>& work, const options& settings
) {
    std::vector<std::string> fingerprints;
    fingerprints.reserve(static_cast<std::size_t>(settings.warmups + settings.repeats));
    for (int repeat = 0; repeat < settings.warmups; ++repeat) {
        (void)timed_gemm(
            work.descriptor, work.a.data(), work.b.data(), work.output.data()
        );
        fingerprints.push_back(sampled_fingerprint(work.output));
    }
    std::vector<measurement> measurements;
    for (int repeat = 0; repeat < settings.repeats; ++repeat) {
        measurement value = timed_gemm(
            work.descriptor, work.a.data(), work.b.data(), work.output.data()
        );
        value.sampled_fingerprint = sampled_fingerprint(work.output);
        fingerprints.push_back(value.sampled_fingerprint);
        measurements.push_back(std::move(value));
    }
    return {
        measurements,
        square_correctness(work, settings.correctness_samples, fingerprints)
    };
}

template <typename T>
struct stream_result {
    std::vector<measurement> measurements;
    correctness_result correctness;
    std::size_t elements = 0;
};

template <typename T>
stream_result<T> run_stream(const options& settings) {
    const std::size_t count = settings.stream_elements;
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("STREAM element count is too large");
    }
    std::vector<T> a(count), b(count), c(count);
    const std::int64_t signed_count = static_cast<std::int64_t>(count);
    #pragma omp parallel for schedule(static)
    for (std::int64_t index = 0; index < signed_count; ++index) {
        b[static_cast<std::size_t>(index)] = static_cast<T>(1.0)
            + static_cast<T>(index % 97) * static_cast<T>(0.0001);
        c[static_cast<std::size_t>(index)] = static_cast<T>(0.5)
            - static_cast<T>(index % 89) * static_cast<T>(0.00005);
        a[static_cast<std::size_t>(index)] = T(0);
    }
    const T scalar = static_cast<T>(1.6180339887498948482);
    std::vector<std::string> fingerprints;
    auto one_triad = [&]() {
        measurement value;
        value.entry.omp_in_parallel = omp_in_parallel();
        value.entry.omp_level = omp_get_level();
        value.entry.omp_active_level = omp_get_active_level();
        value.entry.omp_max_active_levels = omp_get_max_active_levels();
        value.entry.omp_max_threads = omp_get_max_threads();
        value.entry.openblas_threads = openblas_get_num_threads();
        value.entry.cpu_before = sched_getcpu();
        if (value.entry.omp_in_parallel != 0 || value.entry.omp_level != 0
            || value.entry.omp_active_level != 0) {
            throw std::runtime_error("STREAM outer region was already active");
        }
        int team_size = 0;
        const double cpu_start = process_cpu_seconds();
        const auto wall_start = steady_clock::now();
        #pragma omp parallel
        {
            #pragma omp single
            team_size = omp_get_num_threads();
            #pragma omp for schedule(static)
            for (std::int64_t index = 0; index < signed_count; ++index) {
                a[static_cast<std::size_t>(index)] = b[static_cast<std::size_t>(index)]
                    + scalar * c[static_cast<std::size_t>(index)];
            }
        }
        const auto wall_stop = steady_clock::now();
        const double cpu_stop = process_cpu_seconds();
        value.entry.cpu_after = sched_getcpu();
        value.observed_omp_team_size = team_size;
        value.wall_seconds = std::chrono::duration<double>(wall_stop - wall_start).count();
        value.process_cpu_seconds = cpu_stop - cpu_start;
        value.active_core_equivalents = value.wall_seconds > 0.0
            ? value.process_cpu_seconds / value.wall_seconds : 0.0;
        const long double bytes = 3.0L * count * sizeof(T);
        value.bandwidth_gbytes_per_second = value.wall_seconds > 0.0
            ? static_cast<double>(bytes / value.wall_seconds / 1.0e9L) : 0.0;
        value.gflops_per_second = value.wall_seconds > 0.0
            ? static_cast<double>(2.0L * count / value.wall_seconds / 1.0e9L) : 0.0;
        value.process_cpu_core_minutes = value.process_cpu_seconds / 60.0;
        std::uint64_t hash = 1469598103934665603ULL;
        for (const std::size_t index : sample_indices(count, 4096)) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, a.data() + index, sizeof(T));
            hash ^= splitmix64(bits ^ index);
            hash *= 1099511628211ULL;
        }
        std::ostringstream rendered;
        rendered << std::hex << std::setfill('0') << std::setw(16) << hash;
        value.sampled_fingerprint = rendered.str();
        fingerprints.push_back(value.sampled_fingerprint);
        return value;
    };
    for (int warmup = 0; warmup < settings.warmups; ++warmup) (void)one_triad();
    stream_result<T> result;
    result.elements = count;
    for (int repeat = 0; repeat < settings.repeats; ++repeat) {
        result.measurements.push_back(one_triad());
    }
    result.correctness.sample_count = static_cast<int>(
        sample_indices(count, settings.correctness_samples).size()
    );
    for (const std::size_t index : sample_indices(count, settings.correctness_samples)) {
        const long double expected = static_cast<long double>(b[index])
            + static_cast<long double>(scalar) * static_cast<long double>(c[index]);
        const long double error = std::abs(static_cast<long double>(a[index]) - expected);
        result.correctness.max_absolute_error = std::max(
            result.correctness.max_absolute_error, static_cast<double>(error)
        );
        const long double bound = 2.0L * std::numeric_limits<T>::epsilon()
            * (std::abs(static_cast<long double>(b[index]))
               + std::abs(static_cast<long double>(scalar) * c[index]));
        if (!(error <= bound)) result.correctness.within_roundoff_bound = false;
    }
    result.correctness.repeated_outputs_bitwise_equal_at_samples = std::all_of(
        fingerprints.begin() + 1, fingerprints.end(),
        [&](const std::string& value) { return value == fingerprints.front(); }
    );
    return result;
}

std::vector<int> observe_omp_thread_cpus(int threads, int& observed_team_size) {
    std::vector<int> cpus(static_cast<std::size_t>(threads), -1);
    observed_team_size = 0;
    #pragma omp parallel num_threads(threads)
    {
        const int thread = omp_get_thread_num();
        if (thread < threads) cpus[static_cast<std::size_t>(thread)] = sched_getcpu();
        #pragma omp single
        observed_team_size = omp_get_num_threads();
    }
    cpus.resize(static_cast<std::size_t>(std::min(threads, observed_team_size)));
    return cpus;
}

void print_int_vector(std::ostream& output, const std::vector<int>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
}

void print_numa_pages(
    std::ostream& output, const std::map<int, std::uint64_t>& values
) {
    output << '{';
    bool first = true;
    for (const auto& [node, pages] : values) {
        if (!first) output << ',';
        first = false;
        output << quote(std::to_string(node)) << ':' << pages;
    }
    output << '}';
}

void print_measurements(
    std::ostream& output, const std::vector<measurement>& values
) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        const measurement& value = values[index];
        output << '{'
               << "\"repeat\":" << index << ','
               << "\"wall_seconds\":" << value.wall_seconds << ','
               << "\"process_cpu_seconds\":" << value.process_cpu_seconds << ','
               << "\"active_core_equivalents\":" << value.active_core_equivalents << ','
               << "\"gflops_per_second\":" << value.gflops_per_second << ','
               << "\"bandwidth_gbytes_per_second\":"
               << value.bandwidth_gbytes_per_second << ','
               << "\"process_cpu_core_minutes\":"
               << value.process_cpu_core_minutes << ','
               << "\"sampled_fingerprint\":" << quote(value.sampled_fingerprint) << ','
               << "\"observed_omp_team_size\":" << value.observed_omp_team_size << ','
               << "\"vendor_entry\":{"
               << "\"omp_in_parallel\":" << value.entry.omp_in_parallel << ','
               << "\"omp_get_level\":" << value.entry.omp_level << ','
               << "\"omp_get_active_level\":" << value.entry.omp_active_level << ','
               << "\"omp_get_max_active_levels\":"
               << value.entry.omp_max_active_levels << ','
               << "\"omp_get_max_threads\":" << value.entry.omp_max_threads << ','
               << "\"openblas_get_num_threads\":"
               << value.entry.openblas_threads << ','
               << "\"cpu_before\":" << value.entry.cpu_before << ','
               << "\"cpu_after\":" << value.entry.cpu_after
               << "}}";
    }
    output << ']';
}

void print_result(
    const options& settings, const gemm_descriptor& descriptor,
    const std::string& conceptual_product,
    const std::vector<measurement>& measurements,
    const correctness_result& correctness,
    const std::vector<int>& process_affinity,
    const std::vector<int>& omp_thread_cpus,
    int observed_omp_team_size,
    const std::map<int, std::uint64_t>& numa_pages,
    const process_counters& counters_before,
    const process_counters& counters_after,
    std::size_t allocated_bytes,
    std::size_t stream_elements
) {
    std::vector<double> walls, cpus, cores, gflops, bandwidths;
    double total_cpu = 0.0;
    for (const measurement& value : measurements) {
        walls.push_back(value.wall_seconds);
        cpus.push_back(value.process_cpu_seconds);
        cores.push_back(value.active_core_equivalents);
        gflops.push_back(value.gflops_per_second);
        bandwidths.push_back(value.bandwidth_gbytes_per_second);
        total_cpu += value.process_cpu_seconds;
    }
    std::cout << std::setprecision(17);
    std::cout << '{'
              << "\"schema\":\"summit.gxe.local_ceiling_orientation\","
              << "\"schema_version\":1,"
              << "\"timestamp_utc\":" << quote(utc_timestamp()) << ','
              << "\"mode\":" << quote(settings.mode) << ','
              << "\"dtype\":" << quote(settings.dtype) << ','
              << "\"arithmetic_dtype\":"
              << quote(settings.dtype == "f64" ? "float64" : "float32") << ','
              << "\"storage_dtype\":"
              << quote(settings.dtype == "f64" ? "float64" : "float32") << ','
              << "\"orientation\":" << quote(settings.orientation) << ','
              << "\"conceptual_product\":" << quote(conceptual_product) << ','
              << "\"cblas\":{"
              << "\"layout\":" << quote(layout_name(descriptor.layout)) << ','
              << "\"transpose_a\":" << quote(transpose_name(descriptor.transpose_a)) << ','
              << "\"transpose_b\":" << quote(transpose_name(descriptor.transpose_b)) << ','
              << "\"m\":" << descriptor.m << ','
              << "\"n\":" << descriptor.n << ','
              << "\"k\":" << descriptor.k << ','
              << "\"lda\":" << descriptor.lda << ','
              << "\"ldb\":" << descriptor.ldb << ','
              << "\"ldc\":" << descriptor.ldc << "},"
              << "\"shape_context\":{"
              << "\"n_samples\":" << settings.n_samples << ','
              << "\"genotype_block_width\":" << settings.block_width << ','
              << "\"probe_tile\":" << settings.probe_tile << ','
              << "\"environment_tile\":" << settings.environment_tile << ','
              << "\"stream_elements\":" << stream_elements << "},"
              << "\"protocol\":{"
              << "\"warmups\":" << settings.warmups << ','
              << "\"timed_repeats\":" << settings.repeats << ','
              << "\"one_serial_vendor_entry_per_process\":true,"
              << "\"full_matrix_transpose_performed\":false},"
              << "\"backend\":{"
              << "\"name\":\"private_static_openblas\","
              << "\"archive_path\":" << quote(SUMMIT_OPENBLAS_ARCHIVE) << ','
              << "\"archive_sha256\":" << quote(SUMMIT_OPENBLAS_ARCHIVE_SHA256) << ','
              << "\"benchmark_source_sha256\":"
              << quote(SUMMIT_BENCHMARK_SOURCE_SHA256) << ','
              << "\"openblas_config\":"
              << quote(openblas_get_config() ? openblas_get_config() : "") << ','
              << "\"openblas_corename\":"
              << quote(openblas_get_corename() ? openblas_get_corename() : "") << ','
              << "\"openblas_parallel\":" << openblas_get_parallel() << ','
              << "\"openblas_threads\":" << openblas_get_num_threads() << ','
              << "\"openblas_num_procs\":" << openblas_get_num_procs() << "},"
              << "\"threading\":{"
              << "\"requested_threads\":" << settings.threads << ','
              << "\"omp_dynamic\":" << omp_get_dynamic() << ','
              << "\"omp_max_threads\":" << omp_get_max_threads() << ','
              << "\"observed_setup_team_size\":" << observed_omp_team_size << ','
              << "\"observed_setup_thread_cpus\":";
    print_int_vector(std::cout, omp_thread_cpus);
    std::cout << "},\"placement\":{"
              << "\"sched_affinity_cpus\":";
    print_int_vector(std::cout, process_affinity);
    const int current_cpu = sched_getcpu();
    std::cout << ",\"current_cpu\":" << current_cpu
              << ",\"current_cpu_numa_node\":" << numa_node_for_cpu(current_cpu)
              << ",\"cpus_allowed_list\":" << quote(status_field("Cpus_allowed_list"))
              << ",\"mems_allowed_list\":" << quote(status_field("Mems_allowed_list"))
              << ",\"resident_pages_by_numa_node\":";
    print_numa_pages(std::cout, numa_pages);
    std::cout << "},\"memory\":{"
              << "\"allocated_operand_bytes\":" << allocated_bytes << ','
              << "\"peak_rss_kib\":" << counters_after.peak_rss_kib << ','
              << "\"case_major_faults\":"
              << counters_after.major_faults - counters_before.major_faults << ','
              << "\"case_minor_faults\":"
              << counters_after.minor_faults - counters_before.minor_faults << "},"
              << "\"measurements\":";
    print_measurements(std::cout, measurements);
    std::cout << ",\"summary\":{"
              << "\"median_wall_seconds\":" << median(walls) << ','
              << "\"median_process_cpu_seconds\":" << median(cpus) << ','
              << "\"median_active_core_equivalents\":" << median(cores) << ','
              << "\"median_gflops_per_second\":" << median(gflops) << ','
              << "\"median_bandwidth_gbytes_per_second\":" << median(bandwidths) << ','
              << "\"total_timed_process_cpu_core_minutes\":" << total_cpu / 60.0
              << ",\"matrix_minutes_definition\":"
              << quote("sum(timed process CPU seconds) / 60") << "},"
              << "\"correctness\":{"
              << "\"sample_count\":" << correctness.sample_count << ','
              << "\"maximum_absolute_error\":" << correctness.max_absolute_error << ','
              << "\"maximum_relative_error\":" << correctness.max_relative_error << ','
              << "\"maximum_error_to_roundoff_bound\":"
              << correctness.max_error_to_roundoff_bound << ','
              << "\"finite\":" << (correctness.finite ? "true" : "false") << ','
              << "\"within_roundoff_bound\":"
              << (correctness.within_roundoff_bound ? "true" : "false") << ','
              << "\"repeated_outputs_bitwise_equal_at_samples\":"
              << (correctness.repeated_outputs_bitwise_equal_at_samples
                  ? "true" : "false")
              << "}}\n";
}

int parse_positive_int(const std::string& value, const std::string& label) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() || parsed <= 0
        || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(label + " must be a positive 32-bit integer");
    }
    return static_cast<int>(parsed);
}

std::size_t parse_positive_size(const std::string& value, const std::string& label) {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0
        || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(label + " must be a positive size_t");
    }
    return static_cast<std::size_t>(parsed);
}

void print_help() {
    std::cout
        << "Usage: gxe_ceiling_orientation [options]\n"
        << "  --mode square|stream|source|target\n"
        << "  --dtype f64|f32 --layout col|row\n"
        << "  --orientation current|transposed\n"
        << "  --size INT --n INT --k INT\n"
        << "  --probe-tile INT --environment-tile INT\n"
        << "  --stream-elements INT --threads INT\n"
        << "  --warmups INT --repeats INT --correctness-samples INT --seed INT\n";
}

options parse_options(int argc, char** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--mode") result.mode = value;
        else if (argument == "--dtype") result.dtype = value;
        else if (argument == "--layout") result.layout = value;
        else if (argument == "--orientation") result.orientation = value;
        else if (argument == "--size") result.size = parse_positive_int(value, "size");
        else if (argument == "--n") result.n_samples = parse_positive_int(value, "n");
        else if (argument == "--k") result.block_width = parse_positive_int(value, "k");
        else if (argument == "--probe-tile") {
            result.probe_tile = parse_positive_int(value, "probe-tile");
        } else if (argument == "--environment-tile") {
            result.environment_tile = parse_positive_int(value, "environment-tile");
        } else if (argument == "--stream-elements") {
            result.stream_elements = parse_positive_size(value, "stream-elements");
        } else if (argument == "--threads") {
            result.threads = parse_positive_int(value, "threads");
        } else if (argument == "--warmups") {
            result.warmups = parse_positive_int(value, "warmups");
        } else if (argument == "--repeats") {
            result.repeats = parse_positive_int(value, "repeats");
        } else if (argument == "--correctness-samples") {
            result.correctness_samples = parse_positive_int(value, "correctness-samples");
        } else if (argument == "--seed") {
            result.seed = std::stoull(value);
        } else {
            throw std::invalid_argument("unknown option " + argument);
        }
    }
    if (result.mode != "square" && result.mode != "stream"
        && result.mode != "source" && result.mode != "target") {
        throw std::invalid_argument("mode must be square, stream, source, or target");
    }
    if (result.dtype != "f64" && result.dtype != "f32") {
        throw std::invalid_argument("dtype must be f64 or f32");
    }
    if (result.layout != "col" && result.layout != "row") {
        throw std::invalid_argument("layout must be col or row");
    }
    if (result.orientation != "current" && result.orientation != "transposed") {
        throw std::invalid_argument("orientation must be current or transposed");
    }
    if (result.warmups < 3 || result.repeats < 5) {
        throw std::invalid_argument("protocol requires at least 3 warmups and 5 repeats");
    }
    const long long source_panel = 2LL * result.probe_tile * result.environment_tile;
    const long long target_panel = 4LL * result.probe_tile * result.environment_tile;
    if (source_panel > std::numeric_limits<int>::max()
        || target_panel > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("panel width exceeds the CBLAS integer range");
    }
    return result;
}

template <typename T>
int run_typed(const options& settings) {
    const CBLAS_LAYOUT layout = settings.layout == "col"
        ? CblasColMajor : CblasRowMajor;
    int observed_team_size = 0;
    const std::vector<int> thread_cpus = observe_omp_thread_cpus(
        settings.threads, observed_team_size
    );
    if (observed_team_size != settings.threads) {
        throw std::runtime_error("OpenMP did not create the requested setup team");
    }
    const std::set<int> distinct_thread_cpus(
        thread_cpus.begin(), thread_cpus.end()
    );
    if (distinct_thread_cpus.size() != static_cast<std::size_t>(settings.threads)
        || distinct_thread_cpus.count(-1) != 0U) {
        throw std::runtime_error(
            "OpenMP setup workers were not pinned to distinct CPU places"
        );
    }
    const std::vector<int> affinity = affinity_cpus();
    const process_counters before = read_process_counters();

    if (settings.mode == "stream") {
        const stream_result<T> result = run_stream<T>(settings);
        const process_counters after = read_process_counters();
        gemm_descriptor descriptor {
            layout, CblasNoTrans, CblasNoTrans, 0, 0, 0, 0, 0, 0
        };
        print_result(
            settings, descriptor, "a = b + scalar * c", result.measurements,
            result.correctness, affinity, thread_cpus, observed_team_size,
            resident_numa_pages(), before, after,
            3U * settings.stream_elements * sizeof(T), settings.stream_elements
        );
        return result.correctness.finite
            && result.correctness.within_roundoff_bound
            && result.correctness.repeated_outputs_bitwise_equal_at_samples ? 0 : 2;
    }

    if (settings.mode == "square") {
        gemm_case<T> work = make_square_case<T>(settings, layout);
        const auto [measurements, correctness] = run_square(work, settings);
        const process_counters after = read_process_counters();
        print_result(
            settings, work.descriptor, work.conceptual_product, measurements,
            correctness, affinity, thread_cpus, observed_team_size,
            resident_numa_pages(), before, after,
            (work.a.size() + work.b.size() + work.output.size()) * sizeof(T), 0
        );
        return correctness.finite && correctness.within_roundoff_bound
            && correctness.repeated_outputs_bitwise_equal_at_samples ? 0 : 2;
    }

    exact_case<T> work = settings.mode == "source"
        ? make_source_case<T>(settings, layout)
        : make_target_case<T>(settings, layout);
    std::pair<std::vector<measurement>, correctness_result> result;
    if (settings.mode == "source") {
        result = run_gemm_case<T>(work, settings, source_correctness<T>);
    } else {
        result = run_gemm_case<T>(work, settings, target_correctness<T>);
    }
    const process_counters after = read_process_counters();
    print_result(
        settings, work.descriptor, work.conceptual_product,
        result.first, result.second, affinity, thread_cpus, observed_team_size,
        resident_numa_pages(), before, after,
        (work.left.size() + work.right.size() + work.output.size()) * sizeof(T), 0
    );
    return result.second.finite && result.second.within_roundoff_bound
        && result.second.repeated_outputs_bitwise_equal_at_samples ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) try {
    const options settings = parse_options(argc, argv);
    omp_set_dynamic(0);
    omp_set_num_threads(settings.threads);
    // This is the only OpenBLAS thread mutation in the process.  No benchmark
    // mode changes it after the first OpenBLAS query or vendor call.
    openblas_set_num_threads(settings.threads);
    if (openblas_get_num_threads() != settings.threads) {
        throw std::runtime_error("OpenBLAS did not retain the immutable thread count");
    }
    return settings.dtype == "f64"
        ? run_typed<double>(settings)
        : run_typed<float>(settings);
} catch (const std::exception& error) {
    std::cerr << "gxe_ceiling_orientation: " << error.what() << '\n';
    return 1;
}
