#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "genotype.hpp"

namespace summit::mailman {

template <typename CodeT, typename Tacc, typename ValueAt>
inline void pre_multiply_generated(
    const CodeT* packed,
    int segment_size_actual,
    int rows,
    int columns,
    ValueAt&& value_at,
    Tacc* result,
    Tacc* work_table
) {
    const int64_t table_size = compute_mailman_table_size(segment_size_actual);
    for (int row = 0; row < rows; ++row) {
        Tacc* table = work_table
            + static_cast<size_t>(packed[static_cast<size_t>(row)])
                * static_cast<size_t>(columns);
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int column = 0; column < columns; ++column) {
            table[static_cast<size_t>(column)] +=
                static_cast<Tacc>(value_at(row, column));
        }
    }

    int64_t divisor = table_size;
    for (int snp = 0; snp < segment_size_actual; ++snp) {
        divisor /= 3;
        Tacc* output = result
            + static_cast<size_t>(snp) * static_cast<size_t>(columns);
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int column = 0; column < columns; ++column) {
            output[static_cast<size_t>(column)] = Tacc(0);
        }
        for (int64_t index = 0; index < divisor; ++index) {
            Tacc* row0 = work_table
                + static_cast<size_t>(index) * static_cast<size_t>(columns);
            Tacc* row1 = work_table
                + static_cast<size_t>(index + divisor)
                    * static_cast<size_t>(columns);
            Tacc* row2 = work_table
                + static_cast<size_t>(index + 2 * divisor)
                    * static_cast<size_t>(columns);
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int column = 0; column < columns; ++column) {
                const Tacc one = row1[static_cast<size_t>(column)];
                const Tacc two = row2[static_cast<size_t>(column)];
                row1[static_cast<size_t>(column)] = Tacc(0);
                row2[static_cast<size_t>(column)] = Tacc(0);
                row0[static_cast<size_t>(column)] += one + two;
                output[static_cast<size_t>(column)] += one + Tacc(2) * two;
            }
        }
    }
#ifdef _OPENMP
#pragma omp simd
#endif
    for (int column = 0; column < columns; ++column) {
        work_table[static_cast<size_t>(column)] = Tacc(0);
    }
}

// ``segment_buffers`` counts the per-column segment-sized outputs a kernel
// keeps alongside its lookup table (one for the plain pre/post kernels, two
// for the squares-augmented feature kernel), so every Mailman caller shares
// one q-panel budget policy.
template <typename Tacc>
inline int qpanel_width(
    int64_t table_size,
    int q_total,
    int segment_size,
    int segment_buffers = 1
) {
    if (q_total <= 0) return 1;
    if (const char* value = std::getenv("SUMMIT_MAILMAN_QPANEL")) {
        const int requested = std::atoi(value);
        if (requested > 0) return std::max(1, std::min(requested, q_total));
    }

    long long table_megabytes = 8;
    if (const char* value = std::getenv("SUMMIT_MAILMAN_WORK_MB")) {
        const long long requested = std::atoll(value);
        if (requested > 0) table_megabytes = requested;
    }
    const size_t table_budget =
        static_cast<size_t>(table_megabytes) * 1024ULL * 1024ULL;
    const size_t table_bytes_per_column =
        static_cast<size_t>(table_size) * sizeof(Tacc)
        + static_cast<size_t>(std::max(1, segment_size)) * sizeof(Tacc)
            * static_cast<size_t>(std::max(1, segment_buffers))
        + sizeof(double);
    int width = table_bytes_per_column == 0
        ? q_total
        : static_cast<int>(table_budget / table_bytes_per_column);

    width = std::max(1, std::min(width, q_total));
    if (width >= 64) width = (width / 64) * 64;
    return std::max(1, width);
}

template <typename CodeT, typename T, typename Tacc>
inline void pre_multiply_rowmajor(
    const CodeT* packed,
    int segment_size_actual,
    int rows,
    int columns,
    const T* operand,
    int operand_stride,
    Tacc* result,
    Tacc* work_table
) {
    pre_multiply_generated(
        packed, segment_size_actual, rows, columns,
        [&](int row, int column) {
            return operand[
                static_cast<size_t>(row) * static_cast<size_t>(operand_stride)
                + static_cast<size_t>(column)
            ];
        },
        result, work_table
    );
}

// The same lookup-table traversal can expose both sum(g*r) and sum(g^2*r).
// GxE feature moments need the latter, so computing both here avoids decoding
// an N-by-K floating-point genotype matrix solely to square it.
template <typename CodeT, typename T, typename Tacc>
inline void pre_multiply_rowmajor_with_squares(
    const CodeT* packed,
    int segment_size_actual,
    int rows,
    int columns,
    const T* operand,
    int operand_stride,
    Tacc* linear_result,
    Tacc* squared_result,
    Tacc* work_table
) {
    const int64_t table_size = compute_mailman_table_size(segment_size_actual);
    for (int row = 0; row < rows; ++row) {
        Tacc* table = work_table
            + static_cast<size_t>(packed[static_cast<size_t>(row)])
                * static_cast<size_t>(columns);
        const T* values = operand
            + static_cast<size_t>(row) * static_cast<size_t>(operand_stride);
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int column = 0; column < columns; ++column) {
            table[static_cast<size_t>(column)] +=
                static_cast<Tacc>(values[static_cast<size_t>(column)]);
        }
    }

    int64_t divisor = table_size;
    for (int snp = 0; snp < segment_size_actual; ++snp) {
        divisor /= 3;
        Tacc* linear = linear_result
            + static_cast<size_t>(snp) * static_cast<size_t>(columns);
        Tacc* squared = squared_result
            + static_cast<size_t>(snp) * static_cast<size_t>(columns);
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int column = 0; column < columns; ++column) {
            linear[static_cast<size_t>(column)] = Tacc(0);
            squared[static_cast<size_t>(column)] = Tacc(0);
        }
        for (int64_t index = 0; index < divisor; ++index) {
            Tacc* row0 = work_table
                + static_cast<size_t>(index) * static_cast<size_t>(columns);
            Tacc* row1 = work_table
                + static_cast<size_t>(index + divisor)
                    * static_cast<size_t>(columns);
            Tacc* row2 = work_table
                + static_cast<size_t>(index + 2 * divisor)
                    * static_cast<size_t>(columns);
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int column = 0; column < columns; ++column) {
                const Tacc one = row1[static_cast<size_t>(column)];
                const Tacc two = row2[static_cast<size_t>(column)];
                row1[static_cast<size_t>(column)] = Tacc(0);
                row2[static_cast<size_t>(column)] = Tacc(0);
                row0[static_cast<size_t>(column)] += one + two;
                linear[static_cast<size_t>(column)] += one + Tacc(2) * two;
                squared[static_cast<size_t>(column)] += one + Tacc(4) * two;
            }
        }
    }
#ifdef _OPENMP
#pragma omp simd
#endif
    for (int column = 0; column < columns; ++column) {
        work_table[static_cast<size_t>(column)] = Tacc(0);
    }
}

template <typename CodeT, typename T, typename Scale>
inline void post_multiply_colmajor_subset_transform(
    const CodeT* packed,
    int segment_size_actual,
    int row_start,
    int row_count,
    int columns,
    const double* operand,
    int operand_stride,
    T* result,
    int result_stride,
    double* work_table,
    Scale&& scale
) {
    const int64_t table_size = compute_mailman_table_size(segment_size_actual);
    std::memset(
        work_table, 0,
        static_cast<size_t>(table_size) * static_cast<size_t>(columns)
            * sizeof(double)
    );
    int64_t prefix = 1;
    for (int snp = segment_size_actual - 1; snp >= 0; --snp) {
        const double* operand_row = operand
            + static_cast<size_t>(snp) * static_cast<size_t>(operand_stride);
        for (int64_t index = 0; index < prefix; ++index) {
            const int64_t offset0 = index * static_cast<int64_t>(columns);
            const int64_t offset1 = (prefix + index)
                * static_cast<int64_t>(columns);
            const int64_t offset2 = (2 * prefix + index)
                * static_cast<int64_t>(columns);
            for (int column = 0; column < columns; ++column) {
                const double base = work_table[
                    static_cast<size_t>(offset0) + static_cast<size_t>(column)
                ];
                work_table[
                    static_cast<size_t>(offset1) + static_cast<size_t>(column)
                ] = base + operand_row[static_cast<size_t>(column)];
                work_table[
                    static_cast<size_t>(offset2) + static_cast<size_t>(column)
                ] = base + 2.0 * operand_row[static_cast<size_t>(column)];
            }
        }
        prefix *= 3;
    }
    for (int local_row = 0; local_row < row_count; ++local_row) {
        const CodeT code = packed[
            static_cast<size_t>(row_start + local_row)
        ];
        const double* source = work_table
            + static_cast<size_t>(code) * static_cast<size_t>(columns);
        T* destination = result
            + static_cast<size_t>(row_start + local_row);
        for (int column = 0; column < columns; ++column) {
            destination[
                static_cast<size_t>(column)
                    * static_cast<size_t>(result_stride)
            ] += static_cast<T>(
                source[static_cast<size_t>(column)]
                * scale(row_start + local_row, column)
            );
        }
    }
}

template <typename CodeT, typename T>
inline void post_multiply_colmajor_subset(
    const CodeT* packed,
    int segment_size_actual,
    int row_start,
    int row_count,
    int columns,
    const double* operand,
    int operand_stride,
    T* result,
    int result_stride,
    double* work_table
) {
    post_multiply_colmajor_subset_transform(
        packed, segment_size_actual, row_start, row_count, columns,
        operand, operand_stride, result, result_stride, work_table,
        [](int, int) noexcept { return 1.0; }
    );
}

}  // namespace summit::mailman
