#include "nb_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sched.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "blas_compat.hpp"

#ifdef GWLDCORE_USE_OPENBLAS
extern "C" char* openblas_get_config(void);
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

constexpr double kOrthonormalTolerance = 1.0e-10;
constexpr double kStandardizedEnvTolerance = 1.0e-8;
#ifdef GWLDCORE_USE_OPENBLAS
constexpr const char* kNativeGemmIntegrityMode =
    "openmp_partitioned_single_thread_openblas";
#else
constexpr const char* kNativeGemmIntegrityMode =
    "deterministic_disjoint_output_tiled_gemm";
#endif

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

void dgemm_nn(int m, int n, int k,
              const double* a, int lda,
              const double* b, int ldb,
              double* c, int ldc,
              double alpha = 1.0, double beta = 0.0) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm_tn(int m, int n, int k,
              const double* a, int lda,
              const double* b, int ldb,
              double* c, int ldc,
              double alpha = 1.0, double beta = 0.0) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// Internally threaded vendor GEMMs have produced rare corrupt output on the
// large products used by the GxE path on the affected production host. These
// cache-tiled kernels are deliberately BLAS-independent. OpenMP assigns
// disjoint output tiles, every output entry is computed exactly once, and the
// reduction order within an entry is deterministic.
#if defined(GWLDCORE_GEMM_INTEGRITY)
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

constexpr int kGemmIntegrityChecks = 8;
constexpr double kGemmIntegrityTolerance = 2.0e-11;
constexpr int64_t kCheckedGemmMinimumFlops = 1000000000LL;

bool gemm_requires_integrity_checks(int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) return false;
    constexpr uint64_t minimum_products =
        (static_cast<uint64_t>(kCheckedGemmMinimumFlops) + 1U) / 2U;
    const uint64_t mn =
        static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    const uint64_t required_mn =
        (minimum_products + static_cast<uint64_t>(k) - 1U) /
        static_cast<uint64_t>(k);
    return mn >= required_mn;
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
    const size_t operand_a = checked_mul(
        static_cast<size_t>(m), static_cast<size_t>(k),
        "protected GEMM operand A"
    );
    const size_t operand_b = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected GEMM operand B"
    );
    return checked_add(
        checks, std::min(operand_a, operand_b),
        "protected GEMM workspace"
    );
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

bool gemm_integrity_disagrees(double expected, double observed) {
    const double tolerance = kGemmIntegrityTolerance * std::max(
        1.0, std::max(std::abs(expected), std::abs(observed))
    );
    return !std::isfinite(expected) || !std::isfinite(observed)
        || std::abs(expected - observed) > tolerance;
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

int64_t dgemm_tn_checked(int m, int n, int k,
                         const double* a, int lda,
                         const double* b, int ldb,
                         double* c, int ldc,
                         int requested_threads) {
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
    dgemm_tn_tiled(
        kGemmIntegrityChecks, n, k,
        projected.data(), k, b, ldb,
        expected.data(), kGemmIntegrityChecks, requested_threads
    );
    const size_t operand_a = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(m),
        "protected TN operand A"
    );
    const size_t operand_b = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected TN operand B"
    );
    std::unique_ptr<double[]> protected_operand(
        new double[std::min(operand_a, operand_b)]
    );
    const double* vendor_a = a;
    const double* vendor_b = b;
    int vendor_lda = lda;
    int vendor_ldb = ldb;
    if (operand_a <= operand_b) {
        copy_col_major_matrix(
            k, m, a, lda, protected_operand.get(), k, requested_threads
        );
        vendor_a = protected_operand.get();
        vendor_lda = k;
    } else {
        copy_col_major_matrix(
            k, n, b, ldb, protected_operand.get(), k, requested_threads
        );
        vendor_b = protected_operand.get();
        vendor_ldb = k;
    }
    // Expected fingerprints are complete before vendor BLAS starts, and only
    // the smaller input is snapshotted.  Multi-GiB decoded genotype blocks are
    // therefore never duplicated.
    dgemm_tn(
        m, n, k, vendor_a, vendor_lda, vendor_b, vendor_ldb, c, ldc
    );
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
            if (gemm_integrity_disagrees(expected[index], observed[index])) {
                return true;
            }
        }
        return false;
    };
    bool any_disagreement = false;
    for (int column = 0; column < n; ++column) {
        any_disagreement = any_disagreement || column_disagrees(column);
    }
    if (any_disagreement) {
        // A repair is valid only if both inputs still reproduce their exact
        // pre-call fingerprints.  These deterministic recalculations run only
        // on the rare fault path; a mismatch fails closed instead of repairing
        // from a potentially altered input.
        std::vector<double> projected_after(projected.size());
        dgemm_nn_tiled(
            k, kGemmIntegrityChecks, m,
            a, lda, coefficients.data(), m,
            projected_after.data(), k, requested_threads
        );
        dgemm_tn_tiled(
            kGemmIntegrityChecks, n, k,
            projected_after.data(), k, b, ldb,
            observed.data(), kGemmIntegrityChecks, requested_threads
        );
        if (std::memcmp(projected.data(), projected_after.data(),
                        projected.size() * sizeof(double)) != 0 ||
            std::memcmp(expected.data(), observed.data(),
                        expected.size() * sizeof(double)) != 0) {
            throw std::runtime_error(
                "Vendor BLAS altered a protected TN GEMM input; refusing repair"
            );
        }
    }
    // Threaded BLAS partition failures usually damage a contiguous output range.
    // Recompute each such range as one cache-tiled product; launching a full
    // reduction independently for every flagged column rereads A needlessly.
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
        const int count = column - first;
        dgemm_tn_tiled(
            m, count, k, a, lda,
            b + static_cast<size_t>(first) * static_cast<size_t>(ldb),
            ldb,
            c + static_cast<size_t>(first) * static_cast<size_t>(ldc),
            ldc, requested_threads
        );
    }
    if (repaired > 0) {
        dgemm_tn_tiled(
            kGemmIntegrityChecks, n, m,
            coefficients.data(), m, c, ldc,
            observed.data(), kGemmIntegrityChecks, requested_threads
        );
        for (int column = 0; column < n; ++column) {
            if (column_disagrees(column)) {
                throw std::runtime_error(
                    "Independent TN GEMM repair failed its integrity check"
                );
            }
        }
    }
    return repaired;
}

int64_t dgemm_nn_checked(int m, int n, int k,
                         const double* a, int lda,
                         const double* b, int ldb,
                         double* c, int ldc,
                         int requested_threads) {
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
    const size_t operand_a = checked_mul(
        static_cast<size_t>(m), static_cast<size_t>(k),
        "protected NN operand A"
    );
    const size_t operand_b = checked_mul(
        static_cast<size_t>(k), static_cast<size_t>(n),
        "protected NN operand B"
    );
    std::unique_ptr<double[]> protected_operand(
        new double[std::min(operand_a, operand_b)]
    );
    const double* vendor_a = a;
    const double* vendor_b = b;
    int vendor_lda = lda;
    int vendor_ldb = ldb;
    if (operand_a <= operand_b) {
        copy_col_major_matrix(
            m, k, a, lda, protected_operand.get(), m, requested_threads
        );
        vendor_a = protected_operand.get();
        vendor_lda = m;
    } else {
        copy_col_major_matrix(
            k, n, b, ldb, protected_operand.get(), k, requested_threads
        );
        vendor_b = protected_operand.get();
        vendor_ldb = k;
    }
    dgemm_nn(
        m, n, k, vendor_a, vendor_lda, vendor_b, vendor_ldb, c, ldc
    );
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
            if (gemm_integrity_disagrees(expected[index], observed[index])) {
                return true;
            }
        }
        return false;
    };
    bool any_disagreement = false;
    for (int column = 0; column < n; ++column) {
        any_disagreement = any_disagreement || column_disagrees(column);
    }
    if (any_disagreement) {
        std::vector<double> projected_after(projected.size());
        dgemm_tn_tiled(
            k, kGemmIntegrityChecks, m,
            a, lda, coefficients.data(), m,
            projected_after.data(), k, requested_threads
        );
        dgemm_tn_tiled(
            kGemmIntegrityChecks, n, k,
            projected_after.data(), k, b, ldb,
            observed.data(), kGemmIntegrityChecks, requested_threads
        );
        if (std::memcmp(projected.data(), projected_after.data(),
                        projected.size() * sizeof(double)) != 0 ||
            std::memcmp(expected.data(), observed.data(),
                        expected.size() * sizeof(double)) != 0) {
            throw std::runtime_error(
                "Vendor BLAS altered a protected NN GEMM input; refusing repair"
            );
        }
    }
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
        const int count = column - first;
        dgemm_nn_tiled(
            m, count, k, a, lda,
            b + static_cast<size_t>(first) * static_cast<size_t>(ldb),
            ldb,
            c + static_cast<size_t>(first) * static_cast<size_t>(ldc),
            ldc, requested_threads
        );
    }
    if (repaired > 0) {
        dgemm_tn_tiled(
            kGemmIntegrityChecks, n, m,
            coefficients.data(), m, c, ldc,
            observed.data(), kGemmIntegrityChecks, requested_threads
        );
        for (int column = 0; column < n; ++column) {
            if (column_disagrees(column)) {
                throw std::runtime_error(
                    "Independent NN GEMM repair failed its integrity check"
                );
            }
        }
    }
    return repaired;
}
#endif

size_t partitioned_gemm_integrity_workspace_elements(int m, int n, int k) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
    // Large GxE products use the BLAS-independent tiled kernels below. They
    // write disjoint output tiles and need no checksum or operand-copy scratch.
    (void)m; (void)n; (void)k;
    return 0;
#else
    (void)m; (void)n; (void)k;
    return 0;
#endif
}

int64_t dgemm_tn_partitioned_columns(int m, int n, int k,
                                     const double* a, int lda,
                                     const double* b, int ldb,
                                     double* c, int ldc,
                                     int requested_threads) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_OPENBLAS
    // OpenBLAS 0.3.30's internal threaded GEMM has corrupted large products on
    // the production host. Keep each vendor call single-threaded and expose
    // parallelism only across SUMMIT-owned, disjoint output-column ranges.
    openblas_set_num_threads(1);
    const int threads = std::max(1, std::min(requested_threads, n));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int worker = 0; worker < threads; ++worker) {
        const int first = static_cast<int>(
            static_cast<int64_t>(n) * worker / threads
        );
        const int stop = static_cast<int>(
            static_cast<int64_t>(n) * (worker + 1) / threads
        );
        dgemm_tn(
            m, stop - first, k, a, lda,
            b + static_cast<size_t>(first) * static_cast<size_t>(ldb), ldb,
            c + static_cast<size_t>(first) * static_cast<size_t>(ldc), ldc
        );
    }
#else
    dgemm_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc, requested_threads
    );
#endif
    return 0;
#else
    (void) requested_threads;
    dgemm_tn(m, n, k, a, lda, b, ldb, c, ldc);
    return 0;
#endif
}

int64_t dgemm_nn_partitioned_rows(int m, int n, int k,
                                  const double* a, int lda,
                                  const double* b, int ldb,
                                  double* c, int ldc,
                                  int requested_threads,
                                  double alpha = 1.0, double beta = 0.0) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_OPENBLAS
    openblas_set_num_threads(1);
    const int threads = std::max(1, std::min(requested_threads, m));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int worker = 0; worker < threads; ++worker) {
        const int first = static_cast<int>(
            static_cast<int64_t>(m) * worker / threads
        );
        const int stop = static_cast<int>(
            static_cast<int64_t>(m) * (worker + 1) / threads
        );
        dgemm_nn(
            stop - first, n, k, a + first, lda, b, ldb,
            c + first, ldc, alpha, beta
        );
    }
#else
    dgemm_nn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    return 0;
#else
    (void) requested_threads;
    dgemm_nn(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
    return 0;
#endif
}

int64_t dgemm_tn_partitioned_rows(int m, int n, int k,
                                  const double* a, int lda,
                                  const double* b, int ldb,
                                  double* c, int ldc,
                                  int requested_threads,
                                  double alpha = 1.0, double beta = 0.0) {
#if defined(GWLDCORE_GEMM_INTEGRITY)
#ifdef GWLDCORE_USE_OPENBLAS
    openblas_set_num_threads(1);
    const int threads = std::max(1, std::min(requested_threads, m));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int worker = 0; worker < threads; ++worker) {
        const int first = static_cast<int>(
            static_cast<int64_t>(m) * worker / threads
        );
        const int stop = static_cast<int>(
            static_cast<int64_t>(m) * (worker + 1) / threads
        );
        dgemm_tn(
            stop - first, n, k,
            a + static_cast<size_t>(first) * static_cast<size_t>(lda), lda,
            b, ldb, c + first, ldc, alpha, beta
        );
    }
#else
    dgemm_tn_tiled(
        m, n, k, a, lda, b, ldb, c, ldc,
        requested_threads, alpha, beta
    );
#endif
    return 0;
#else
    (void) requested_threads;
    dgemm_tn(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
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
                   std::unique_ptr<double[]>&& data)
        : context_id_(context_id), rows_(rows), columns_(columns),
          leakage_(leakage), elements_(elements), data_(std::move(data)) {}

    uint64_t context_id_ = 0;
    int rows_ = 0;
    int columns_ = 0;
    double leakage_ = 0.0;
    size_t elements_ = 0;
    // One column-major [S, E*S] allocation lets target work consume both
    // left operators in one wide GEMM without copying the persistent panel.
    std::unique_ptr<double[]> data_;
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
                  bool strict_feature_moment_verification)
        : context_id_(next_context_id()),
          ddof_(ddof),
          decode_threads_(decode_threads),
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
#ifdef _OPENMP
        int thread_limit = omp_get_thread_limit();
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        if (::sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
            const int affinity_count = CPU_COUNT(&affinity);
            if (affinity_count > 0) thread_limit = std::min(thread_limit, affinity_count);
        }
        if (decode_threads_ > thread_limit) {
            throw std::runtime_error(
                "GxE native decode_threads exceeds the OpenMP/CPU-affinity limit"
            );
        }
#else
        if (decode_threads_ != 1) {
            throw std::runtime_error(
                "GxE native decode_threads must be one when OpenMP is unavailable"
            );
        }
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
        result["max_workspace_bytes"] = max_workspace_bytes_;
        result["target_panel_columns"] = target_panel_columns_;
        result["projected_target_full_width"] = true;
        result["strict_feature_moment_verification"] = strict_feature_moment_verification_;
        result["feature_moment_integrity_mode"] = strict_feature_moment_verification_
            ? "strict_duplicate"
            : kNativeGemmIntegrityMode;
        result["repaired_gemm_output_columns"] =
            repaired_gemm_output_columns_.load(std::memory_order_relaxed);
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
        const int rank = n_ - q_;
        const int moment_rows = 4 * q_;
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "feature genotype");
        const size_t moment_copies = strict_feature_moment_verification_ ? 8U : 4U;
        elements = checked_add(elements, checked_mul(moment_copies, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(l), "feature moments"), "feature moments"), "feature workspace");
        elements = checked_add(elements, checked_mul(11U, static_cast<size_t>(l), "feature vectors"), "feature workspace");
        elements = checked_add(
            elements,
            partitioned_gemm_integrity_workspace_elements(moment_rows, l, n_),
            "feature integrity workspace"
        );
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
            std::vector<double> moments(
                checked_mul(4U, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(l), "feature moments"), "feature moments"),
                0.0
            );
            std::vector<double> moment_verification;
            if (strict_feature_moment_verification_) {
                moment_verification.assign(
                    checked_mul(
                        4U,
                        checked_mul(static_cast<size_t>(q_), static_cast<size_t>(l), "feature moment verification"),
                        "feature moment verification"
                    ),
                    0.0
                );
            }
            std::vector<double> scalar(4U * static_cast<size_t>(l), 0.0);
            double* s0 = scalar.data();
            double* s1 = s0 + l;
            double* s2 = s1 + l;
            double* s4 = s2 + l;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
            for (int j = 0; j < l; ++j) {
                const double* column = geno.get() + static_cast<size_t>(j) * static_cast<size_t>(n_);
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a4 = 0.0;
                for (int i = 0; i < n_; ++i) {
                    const double e = env_[static_cast<size_t>(i)];
                    const double g2 = column[i] * column[i];
                    const double e2 = e * e;
                    a0 += g2;
                    a1 += e * g2;
                    a2 += e2 * g2;
                    a4 += e2 * e2 * g2;
                }
                s0[j] = a0; s1[j] = a1; s2[j] = a2; s4[j] = a4;
            }
            // [Q, E Q, E^2 Q, E^3 Q]^T G yields every required projected
            // moment in one cache-efficient GEMM instead of four skinny
            // products and three full genotype rewrites.
            const int64_t abft_repairs = dgemm_tn_partitioned_columns(
                moment_rows, l, n_, feature_moment_basis_.data(), n_,
                geno.get(), n_, moments.data(), moment_rows, decode_threads_
            );
            repaired_feature_moment_columns += abft_repairs;
            record_gemm_repairs(abft_repairs);
            if (strict_feature_moment_verification_) {
                // Strict mode repeats the full product. It is retained for
                // stress testing and unusually conservative deployments.
                dgemm_tn_tiled(
                    moment_rows, l, n_, feature_moment_basis_.data(), n_,
                    geno.get(), n_, moment_verification.data(), moment_rows,
                    decode_threads_
                );
            }
            for (int j = 0; j < l; ++j) {
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
                    const double* genotype_column = geno.get() +
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
            for (int j = 0; j < l; ++j) {
                const double* u0 = moments.data() +
                    static_cast<size_t>(j) * 4U * static_cast<size_t>(q_);
                const double* u1 = u0 + q_;
                const double* u2 = u1 + q_;
                const double* u3 = u2 + q_;
                double u00 = 0.0, u11 = 0.0, u01 = 0.0;
                double u0u2 = 0.0, u1u3 = 0.0;
                double u0e2u0 = 0.0, u1e2u1 = 0.0;
                double leak_x_sq = 0.0, leak_w_sq = 0.0;
                for (int a = 0; a < q_; ++a) {
                    u00 += u0[a] * u0[a];
                    u11 += u1[a] * u1[a];
                    u01 += u0[a] * u1[a];
                    u0u2 += u0[a] * u2[a];
                    u1u3 += u1[a] * u3[a];
                    double e2u0 = 0.0, e2u1 = 0.0;
                    double gram_u0 = 0.0, gram_u1 = 0.0;
                    for (int b = 0; b < q_; ++b) {
                        const size_t index = static_cast<size_t>(b) * static_cast<size_t>(q_) + static_cast<size_t>(a);
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
                norm_x[j] = scale_x[j] * scale_x[j] * ssx / static_cast<double>(rank);
                norm_w[j] = scale_w[j] * scale_w[j] * ssw / static_cast<double>(rank);
                diag_x[j] = scale_x[j] * scale_x[j] *
                    (s2[j] - 2.0 * u0u2 + u0e2u0) / static_cast<double>(rank);
                diag_w[j] = scale_w[j] * scale_w[j] *
                    (s4[j] - 2.0 * u1u3 + u1e2u1) / static_cast<double>(rank);
                corr_xw[j] = scale_x[j] * scale_w[j] *
                    (s1[j] - u01) / static_cast<double>(rank);
                max_leak_x = std::max(max_leak_x, std::sqrt(std::max(0.0, leak_x_sq) / std::max(ssx, std::numeric_limits<double>::min())));
                max_leak_w = std::max(max_leak_w, std::sqrt(std::max(0.0, leak_w_sq) / std::max(ssw, std::numeric_limits<double>::min())));
                if (!std::isfinite(scale_x[j]) || !std::isfinite(scale_w[j]) ||
                    !std::isfinite(norm_x[j]) || !std::isfinite(norm_w[j]) ||
                    !std::isfinite(diag_x[j]) || !std::isfinite(diag_w[j]) ||
                    !std::isfinite(corr_xw[j])) {
                    throw std::runtime_error("GxE native feature output contains a non-finite value");
                }
            }
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
            : kNativeGemmIntegrityMode;
        result["missing_genotype_calls"] = missing;
        return result;
    }

    nb::tuple source_block(int blk_start,
                           int blk_end,
                           nb_vec1_ro<double> scale_x,
                           nb_vec1_ro<double> scale_w,
                           nb_vec1_ro<double> sqrt_annotation,
                           nb_mat2f_ro<double> probes,
                           nb_vec1_ro<int32_t> group_ids,
                           int num_groups,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int v = checked_blas_dim(probes.shape(1), "source probes");
        if (v <= 0 || checked_blas_dim(probes.shape(0), "source variants") != l ||
            checked_blas_dim(scale_x.shape(0), "source scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "source scale_w") != l ||
            checked_blas_dim(sqrt_annotation.shape(0), "source annotation") != l ||
            checked_blas_dim(group_ids.shape(0), "source groups") != l) {
            throw std::runtime_error("GxE native source input shape mismatch");
        }
        if (num_groups < 1 || num_groups > 4) {
            throw std::runtime_error("GxE native source supports one to four local groups");
        }
        const size_t columns_size = checked_mul(static_cast<size_t>(num_groups), static_cast<size_t>(v), "source columns");
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
        elements = checked_add(elements, checked_mul(4U, static_cast<size_t>(l), "source vector snapshots"), "source input snapshots");
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
        const std::vector<int32_t> group_snapshot(
            group_ids.data(), group_ids.data() + static_cast<size_t>(l)
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
            const int32_t* gp = group_snapshot.data();
            for (int j = 0; j < l; ++j) {
                const int group = static_cast<int>(gp[j]);
                if (group < 0 || group >= num_groups) {
                    throw std::runtime_error("GxE native source group ID is out of range");
                }
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
                    const size_t index = static_cast<size_t>(group * v + c) * static_cast<size_t>(l) + static_cast<size_t>(j);
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
            record_gemm_repairs(dgemm_nn_partitioned_rows(
                n_, fused_columns, l, geno.get(), n_, weighted.data(), l,
                fused_source, n_, decode_threads_
            ));
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
        return ProjectedPanel(
            context_id_, n_, columns, leakage, snapshot_elements,
            std::move(snapshot)
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
                    qts.data(), q_, decode_threads_
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
                    projected.data(), n_, decode_threads_, -1.0, 1.0
                ));
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
                for (int c = 0; c < count; ++c) {
                    const double* source = projected.data() + static_cast<size_t>(c) * static_cast<size_t>(n_);
                    double* weighted = env_projected.data() + static_cast<size_t>(c) * static_cast<size_t>(n_);
                    for (int i = 0; i < n_; ++i) weighted[i] = env_[static_cast<size_t>(i)] * source[i];
                }
                record_gemm_repairs(dgemm_tn_partitioned_rows(
                    l, count, n_, geno.get(), n_, projected.data(), n_,
                    work_x + static_cast<size_t>(c0) * static_cast<size_t>(l),
                    l, decode_threads_
                ));
                record_gemm_repairs(dgemm_tn_partitioned_rows(
                    l, count, n_, geno.get(), n_, env_projected.data(), n_,
                    work_w + static_cast<size_t>(c0) * static_cast<size_t>(l),
                    l, decode_threads_
                ));
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
        return target_projected_panels(
            blk_start, blk_end, scale_x, scale_w, sources, nullptr,
            require_missing_free
        );
    }

    nb::tuple target_projected_pair_block(int blk_start,
                                          int blk_end,
                                          nb_vec1_ro<double> scale_x,
                                          nb_vec1_ro<double> scale_w,
                                          const ProjectedPanel& first,
                                          const ProjectedPanel& second,
                                          bool require_missing_free) const {
        return target_projected_panels(
            blk_start, blk_end, scale_x, scale_w, first, &second,
            require_missing_free
        );
    }

private:
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
            coefficients.data(), q_, decode_threads_
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

    nb::tuple target_projected_panels(
        int blk_start,
        int blk_end,
        nb_vec1_ro<double> scale_x,
        nb_vec1_ro<double> scale_w,
        const ProjectedPanel& first,
        const ProjectedPanel* second,
        bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        validate_projected_panel_handle(first);
        if (second != nullptr) validate_projected_panel_handle(*second);
        const int l = blk_end - blk_start;
        const size_t columns_size = checked_add(
            static_cast<size_t>(first.columns_),
            second == nullptr ? 0U : static_cast<size_t>(second->columns_),
            "projected target columns"
        );
        const int columns = checked_blas_dim(columns_size, "projected target columns");
        if (checked_blas_dim(scale_x.shape(0), "target scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "target scale_w") != l) {
            throw std::runtime_error("GxE native target input shape mismatch");
        }
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "target genotype");
        elements = checked_add(elements, checked_mul(2U, static_cast<size_t>(l), "target scale snapshots"), "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(l), columns_size, "target outputs"), "target outputs"), "target workspace");
        const size_t widest_panel = std::max(
            static_cast<size_t>(first.columns_),
            second == nullptr ? 0U : static_cast<size_t>(second->columns_)
        );
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
                record_gemm_repairs(dgemm_tn_partitioned_rows(
                    l, fused_columns, n_, geno.get(), n_,
                    panel.data_.get(), n_,
                    fused_work.get(), l, decode_threads_
                ));
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
            consume(first);
            if (second != nullptr) consume(*second);
            scale_target_outputs(
                work_x, work_w, l, columns, scale_x_snapshot, scale_w_snapshot
            );
            check_files_unchanged();
        }
        const double leakage = std::max(
            first.leakage_, second == nullptr ? 0.0 : second->leakage_
        );
        return nb::make_tuple(work_x_out, work_w_out, missing, leakage);
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
        observed.assign(static_cast<size_t>(l), 0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int j = 0; j < l; ++j) {
            const unsigned char* bytes = bed_base_ + 3 +
                static_cast<size_t>(blk_start + j) * bytes_per_snp_;
            double* column = geno.get() +
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
            coefficients.data(), q_, decode_threads_
        ));
        record_gemm_repairs(dgemm_nn_partitioned_rows(
            n_, columns, q_, q_basis_.data(), n_, coefficients.data(), q_,
            panel, n_, decode_threads_, -1.0, 1.0
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

}  // namespace

NB_MODULE(gxeldcore, module) {
    module.doc() = "Bounded double-precision native context for standardized one-environment GxE sketches";
    module.attr("__version__") = "1.0";
    module.def("build_info", []() {
        nb::dict result;
        result["backend_name"] = "gxeldcore_direct";
        result["backend_version"] = "1.0";
        result["api_version"] = 2;
        result["source_commit"] = GWLDCORE_SOURCE_COMMIT;
        result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
        result["compiler_id"] = GWLDCORE_COMPILER_ID;
        result["compiler_version"] = GWLDCORE_COMPILER_VERSION;
        result["build_type"] = GWLDCORE_BUILD_TYPE;
        result["blas_vendor"] = GWLDCORE_BLAS_VENDOR;
        result["cxx_standard"] = 17;
        result["optimization"] = "-O3";
        result["architecture_tuning"] = (
            bool(GWLDCORE_NATIVE_OPT) ? "-march=native" : "portable"
        );
        result["openmp_enabled"] = bool(GWLDCORE_OPENMP_ENABLED);
        result["native_optimization_enabled"] = bool(GWLDCORE_NATIVE_OPT);
        result["platform"] = "linux";
#ifdef GWLDCORE_USE_OPENBLAS
        result["blas_runtime_config"] = std::string(openblas_get_config());
#else
        result["blas_runtime_config"] = nb::none();
#endif
        return result;
    });
    nb::class_<ProjectedPanel>(module, "ProjectedPanel")
        .def_prop_ro("columns", &ProjectedPanel::columns)
        .def_prop_ro("leakage", &ProjectedPanel::leakage);
    nb::class_<DirectContext>(module, "DirectContext")
        .def(
            nb::init<int, int, int, nb::object, int, nb_vec1_ro<double>, nb_mat2f_ro<double>, int, uint64_t, int, bool>(),
            nb::arg("bed_descriptor"), nb::arg("bim_descriptor"), nb::arg("fam_descriptor"),
            nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1,
            nb::arg("env"), nb::arg("q_basis"), nb::arg("decode_threads"),
            nb::arg("max_workspace_bytes"), nb::arg("target_panel_columns") = 64,
            nb::arg("strict_feature_moment_verification") = true
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
            nb::arg("probes"), nb::arg("group_ids"), nb::arg("num_groups"),
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
            "target_projected_pair_block", &DirectContext::target_projected_pair_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"),
            nb::arg("first"), nb::arg("second"),
            nb::arg("require_missing_free") = true
        );
}
