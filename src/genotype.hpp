#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <type_traits>

// Exported helper (external linkage)
int64_t count_lines_cached(const std::string& path);

// Exported concrete overloads (external symbols)
void read_block_standardized_float(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    std::vector<float>& Geno, // (N x L), col-major
    int& N, int& L);

void read_block_standardized_double(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    std::vector<double>& Geno, // (N x L), col-major
    int& N, int& L);

// Convenience inline that keeps your existing call sites unchanged
template <typename T>
inline void read_block_standardized(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    std::vector<T>& Geno,
    int& N, int& L)
{
    if constexpr (std::is_same_v<T,float>) {
        read_block_standardized_float(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    } else {
        read_block_standardized_double(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    }
}
