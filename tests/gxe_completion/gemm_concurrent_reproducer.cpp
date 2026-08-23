#include <cblas.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" void openblas_set_num_threads(int);
extern "C" char* openblas_get_config(void);
extern "C" char* openblas_get_corename(void);

namespace {

constexpr std::size_t guard_size = 32;
constexpr std::uint64_t guard_bits = 0x7ff4d00dcafebeefULL;

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double frozen_value(std::size_t index, std::uint64_t stream) {
    const std::uint64_t bits = splitmix64(index ^ stream);
    return (static_cast<double>(bits >> 11U) / 9007199254740992.0) - 0.5;
}

class guarded_matrix {
public:
    explicit guarded_matrix(std::size_t elements)
        : elements_(elements), storage_(elements + 2 * guard_size) {
        double guard;
        std::memcpy(&guard, &guard_bits, sizeof(guard));
        std::fill(storage_.begin(), storage_.begin() + guard_size, guard);
        std::fill(storage_.end() - guard_size, storage_.end(), guard);
    }
    double* data() { return storage_.data() + guard_size; }
    const double* data() const { return storage_.data() + guard_size; }
    std::size_t size() const { return elements_; }
    bool guards_intact() const {
        for (std::size_t index = 0; index < guard_size; ++index) {
            std::uint64_t left = 0;
            std::uint64_t right = 0;
            std::memcpy(&left, storage_.data() + index, sizeof(left));
            std::memcpy(
                &right,
                storage_.data() + guard_size + elements_ + index,
                sizeof(right)
            );
            if (left != guard_bits || right != guard_bits) return false;
        }
        return true;
    }
private:
    std::size_t elements_;
    std::vector<double> storage_;
};

std::uint64_t fingerprint(const double* values, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < size; ++index) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        hash ^= bits ^ splitmix64(index);
        hash *= 1099511628211ULL;
    }
    return hash;
}

int integer(const char* value, const char* label) {
    const long long parsed = std::stoll(value);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(label) + " must be a positive BLAS int");
    }
    return static_cast<int>(parsed);
}

void gemm(const std::string& operation, int m, int n, int k,
          const double* a, const double* b, double* c) {
    if (operation == "nn") {
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, 1.0, a, m, b, k, 0.0, c, m
        );
    } else {
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans,
            m, n, k, 1.0, a, k, b, k, 0.0, c, m
        );
    }
}

void externally_partitioned(
    const std::string& operation, int m, int n, int k,
    const double* a, const double* b, double* c, int threads,
    bool mutexed, bool partition_columns, bool private_rows
) {
    std::mutex vendor_mutex;
    openblas_set_num_threads(1);
    std::vector<std::unique_ptr<guarded_matrix>> private_output;
    std::vector<int> private_first;
    std::vector<int> private_rows_count;
    if (private_rows) {
        private_output.reserve(static_cast<std::size_t>(threads));
        private_first.reserve(static_cast<std::size_t>(threads));
        private_rows_count.reserve(static_cast<std::size_t>(threads));
        for (int worker = 0; worker < threads; ++worker) {
            const int first = static_cast<int>(
                static_cast<std::int64_t>(m) * worker / threads
            );
            const int stop = static_cast<int>(
                static_cast<std::int64_t>(m) * (worker + 1) / threads
            );
            const int rows = stop - first;
            private_first.push_back(first);
            private_rows_count.push_back(rows);
            private_output.emplace_back(std::make_unique<guarded_matrix>(
                static_cast<std::size_t>(rows) * n
            ));
        }
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&, worker]() {
            if (partition_columns) {
                const int first = static_cast<int>(
                    static_cast<std::int64_t>(n) * worker / threads
                );
                const int stop = static_cast<int>(
                    static_cast<std::int64_t>(n) * (worker + 1) / threads
                );
                gemm(
                    operation, m, stop - first, k, a,
                    b + static_cast<std::size_t>(first) * k,
                    c + static_cast<std::size_t>(first) * m
                );
                return;
            }
            const int first = static_cast<int>(
                static_cast<std::int64_t>(m) * worker / threads
            );
            const int stop = static_cast<int>(
                static_cast<std::int64_t>(m) * (worker + 1) / threads
            );
            const int rows = stop - first;
            auto call = [&]() {
            double* worker_c = private_rows
                ? private_output[static_cast<std::size_t>(worker)]->data()
                : c + first;
            const int worker_ldc = private_rows ? rows : m;
            if (operation == "nn") {
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                    rows, n, k, 1.0, a + first, m, b, k,
                    0.0, worker_c, worker_ldc
                );
            } else {
                cblas_dgemm(
                    CblasColMajor, CblasTrans, CblasNoTrans,
                    rows, n, k, 1.0,
                    a + static_cast<std::size_t>(first) * k, k,
                    b, k, 0.0, worker_c, worker_ldc
                );
            }
            };
            if (mutexed) {
                std::lock_guard<std::mutex> lock(vendor_mutex);
                call();
            } else {
                call();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    if (private_rows) {
        for (int worker = 0; worker < threads; ++worker) {
            const int first = private_first[static_cast<std::size_t>(worker)];
            const int rows = private_rows_count[static_cast<std::size_t>(worker)];
            const guarded_matrix& local = *private_output[
                static_cast<std::size_t>(worker)
            ];
            if (!local.guards_intact()) {
                throw std::runtime_error("private output guard was overwritten");
            }
            for (int column = 0; column < n; ++column) {
                std::memcpy(
                    c + static_cast<std::size_t>(column) * m + first,
                    local.data() + static_cast<std::size_t>(column) * rows,
                    static_cast<std::size_t>(rows) * sizeof(double)
                );
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) try {
    std::string operation = "nn";
    std::string mode = "sequential";
    int m = 256;
    int n = 32;
    int k = 512;
    int threads = 4;
    int repeats = 2;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("option lacks a value");
        const std::string option = argv[index];
        const char* value = argv[index + 1];
        if (option == "--op") operation = value;
        else if (option == "--mode") mode = value;
        else if (option == "--m") m = integer(value, "m");
        else if (option == "--n") n = integer(value, "n");
        else if (option == "--k") k = integer(value, "k");
        else if (option == "--threads") threads = integer(value, "threads");
        else if (option == "--repeats") repeats = integer(value, "repeats");
        else throw std::invalid_argument("unknown option: " + option);
    }
    if (operation != "nn" && operation != "tn") {
        throw std::invalid_argument("--op must be nn or tn");
    }
    if (
        mode != "sequential" && mode != "external"
        && mode != "external_mutex" && mode != "external_columns"
        && mode != "external_private" && mode != "internal"
        && mode != "internal_restore"
    ) {
        throw std::invalid_argument("unsupported --mode");
    }
    const std::size_t a_size = operation == "nn"
        ? static_cast<std::size_t>(m) * k
        : static_cast<std::size_t>(k) * m;
    const std::size_t b_size = static_cast<std::size_t>(k) * n;
    const std::size_t c_size = static_cast<std::size_t>(m) * n;
    guarded_matrix a(a_size), b(b_size), reference(c_size), output(c_size);
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(a_size); ++index) {
        a.data()[index] = frozen_value(static_cast<std::size_t>(index), 0x12345678ULL);
    }
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(b_size); ++index) {
        b.data()[index] = frozen_value(static_cast<std::size_t>(index), 0x87654321ULL);
    }
    const std::uint64_t a_before = fingerprint(a.data(), a.size());
    const std::uint64_t b_before = fingerprint(b.data(), b.size());
    openblas_set_num_threads(1);
    gemm(operation, m, n, k, a.data(), b.data(), reference.data());

    double maximum_error = 0.0;
    double maximum_relative_error = 0.0;
    int corrupt_repeats = 0;
    int first_corrupt_repeat = -1;
    std::size_t first_corrupt_index = 0;
    std::size_t first_corrupt_count = 0;
    int first_corrupt_min_row = m;
    int first_corrupt_max_row = -1;
    int first_corrupt_min_column = n;
    int first_corrupt_max_column = -1;
    const auto started = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (mode == "external" || mode == "external_mutex"
            || mode == "external_columns" || mode == "external_private") {
            externally_partitioned(
                operation, m, n, k, a.data(), b.data(), output.data(), threads,
                mode == "external_mutex", mode == "external_columns",
                mode == "external_private"
            );
        } else {
            openblas_set_num_threads(
                mode == "internal" || mode == "internal_restore" ? threads : 1
            );
            gemm(operation, m, n, k, a.data(), b.data(), output.data());
            if (mode == "internal_restore") openblas_set_num_threads(1);
        }
        double repeat_maximum_error = 0.0;
        std::size_t repeat_corrupt_count = 0;
        int repeat_min_row = m;
        int repeat_max_row = -1;
        int repeat_min_column = n;
        int repeat_max_column = -1;
        for (std::size_t index = 0; index < c_size; ++index) {
            const double error = std::abs(output.data()[index] - reference.data()[index]);
            maximum_error = std::max(maximum_error, error);
            repeat_maximum_error = std::max(repeat_maximum_error, error);
            maximum_relative_error = std::max(
                maximum_relative_error,
                error / std::max(1.0, std::abs(reference.data()[index]))
            );
            if (error > 1.0e-10) {
                ++repeat_corrupt_count;
                const int row = static_cast<int>(index % static_cast<std::size_t>(m));
                const int column = static_cast<int>(index / static_cast<std::size_t>(m));
                repeat_min_row = std::min(repeat_min_row, row);
                repeat_max_row = std::max(repeat_max_row, row);
                repeat_min_column = std::min(repeat_min_column, column);
                repeat_max_column = std::max(repeat_max_column, column);
            }
        }
        if (repeat_maximum_error > 1.0e-10) {
            ++corrupt_repeats;
            if (first_corrupt_repeat < 0) {
                first_corrupt_repeat = repeat;
                first_corrupt_count = repeat_corrupt_count;
                first_corrupt_min_row = repeat_min_row;
                first_corrupt_max_row = repeat_max_row;
                first_corrupt_min_column = repeat_min_column;
                first_corrupt_max_column = repeat_max_column;
                for (std::size_t index = 0; index < c_size; ++index) {
                    if (std::abs(output.data()[index] - reference.data()[index]) > 1.0e-10) {
                        first_corrupt_index = index;
                        break;
                    }
                }
            }
        }
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    const bool input_unchanged =
        a_before == fingerprint(a.data(), a.size())
        && b_before == fingerprint(b.data(), b.size());
    const bool guards = a.guards_intact() && b.guards_intact()
        && reference.guards_intact() && output.guards_intact();
    std::cout << std::setprecision(17)
        << "{\"op\":\"" << operation << "\",\"mode\":\"" << mode
        << "\",\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
        << ",\"threads\":" << threads << ",\"repeats\":" << repeats
        << ",\"seconds\":" << seconds
        << ",\"corrupt_repeats\":" << corrupt_repeats
        << ",\"first_corrupt_repeat\":" << first_corrupt_repeat
        << ",\"first_corrupt_index\":" << first_corrupt_index
        << ",\"first_corrupt_count\":" << first_corrupt_count
        << ",\"first_corrupt_min_row\":" << first_corrupt_min_row
        << ",\"first_corrupt_max_row\":" << first_corrupt_max_row
        << ",\"first_corrupt_min_column\":" << first_corrupt_min_column
        << ",\"first_corrupt_max_column\":" << first_corrupt_max_column
        << ",\"max_abs_error\":" << maximum_error
        << ",\"max_relative_error\":" << maximum_relative_error
        << ",\"input_unchanged\":" << (input_unchanged ? "true" : "false")
        << ",\"guards_intact\":" << (guards ? "true" : "false")
        << ",\"openblas_config\":\"" << openblas_get_config()
        << "\",\"openblas_core\":\"" << openblas_get_corename() << "\"}\n";
    return input_unchanged && guards ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
