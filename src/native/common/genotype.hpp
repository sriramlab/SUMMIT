#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <type_traits>

void prefetch_bed_block(const std::string& bed_path,
                        const std::string& fam_path,
                        int blk_start, int blk_end,
                        int ahead_blocks = 1);

int64_t count_lines_cached(const std::string& path);

struct BedMappingCacheInfo {
    std::size_t entries = 0;
    std::size_t capacity = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

BedMappingCacheInfo bed_mapping_cache_info();
void clear_bed_mapping_cache();

enum class ImputeMode : int {
    Mean = 0,
    Hwe  = 1,
};

struct MailmanPackedBlock {
    int N = 0;
    int L = 0;
    int segment_size = 1;
    int64_t n_segments = 0;
    int64_t table_size = 1;
    bool use_u16 = true;
    std::vector<uint16_t> packed16;
    std::vector<uint32_t> packed32;
    std::vector<double> mean;
    std::vector<double> inv_std;
};

int compute_mailman_segment_size_optimized(int64_t n_rows);
int64_t compute_mailman_table_size(int segment_size);

void read_block_standardized_float(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    ImputeMode impute_mode,
    uint64_t impute_seed,
    std::vector<float>& Geno,
    int& N, int& L);

void read_block_standardized_double(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    ImputeMode impute_mode,
    uint64_t impute_seed,
    std::vector<double>& Geno,
    int& N, int& L);

void read_block_standardized_float_into(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    ImputeMode impute_mode,
    uint64_t impute_seed,
    float* Geno,
    std::size_t Geno_elems,
    int& N, int& L);

void read_block_standardized_double_into(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    ImputeMode impute_mode,
    uint64_t impute_seed,
    double* Geno,
    std::size_t Geno_elems,
    int& N, int& L);

void read_block_mailman_hwe(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start,
    int blk_end,
    const std::vector<int>& rows,
    int ddof,
    uint64_t impute_seed,
    MailmanPackedBlock& out);

void compute_maf_block(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start,
    int blk_end,
    const std::vector<int>& rows,
    std::vector<double>& maf);

template <typename T>
inline void read_block_standardized(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    ImputeMode impute_mode,
    uint64_t impute_seed,
    std::vector<T>& Geno,
    int& N, int& L)
{
    if constexpr (std::is_same_v<T,float>) {
        read_block_standardized_float(bed_path, fam_path, blk_start, blk_end,
                                      rows, ddof, impute_mode, impute_seed,
                                      Geno, N, L);
    } else {
        read_block_standardized_double(bed_path, fam_path, blk_start, blk_end,
                                       rows, ddof, impute_mode, impute_seed,
                                       Geno, N, L);
    }
}

template <typename T>
inline void read_block_standardized_into(
    const std::string& bed_path,
    const std::string& fam_path,
    int blk_start, int blk_end,
    const std::vector<int>& rows,
    int ddof,
    ImputeMode impute_mode,
    uint64_t impute_seed,
    T* Geno,
    std::size_t Geno_elems,
    int& N, int& L)
{
    if constexpr (std::is_same_v<T,float>) {
        read_block_standardized_float_into(bed_path, fam_path, blk_start, blk_end,
                                           rows, ddof, impute_mode, impute_seed,
                                           Geno, Geno_elems, N, L);
    } else {
        read_block_standardized_double_into(bed_path, fam_path, blk_start, blk_end,
                                            rows, ddof, impute_mode, impute_seed,
                                            Geno, Geno_elems, N, L);
    }
}
