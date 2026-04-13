#include "genotype.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <climits>
  #include <sched.h>   // sched_getaffinity
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

// ---------------- small helpers ----------------
static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

static inline int64_t count_lines_plain(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open file: " + path);
    int64_t n = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '#') continue;
        ++n;
    }
    return n;
}

int64_t count_lines_cached(const std::string &path) {
    static std::mutex m;
    static std::unordered_map<std::string, int64_t> cache;
    {
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(path);
        if (it != cache.end()) return it->second;
    }
    int64_t n = count_lines_plain(path);
    {
        std::lock_guard<std::mutex> lk(m);
        cache[path] = n;
    }
    return n;
}

static inline int env_int(const char* k, int defv) {
    const char* s = std::getenv(k);
    if (!s || !*s) return defv;
    return std::atoi(s);
}

// Prefer affinity-aware thread cap (cpuset / taskset / cgroups pinning).
static inline int affinity_thread_cap() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int cnt = 0;
        for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &set)) ++cnt;
        if (cnt > 0) return cnt;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#else
    return 1;
#endif
}

static inline uint64_t mix64_local(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= (x >> 31);
    return x;
}

static inline double u01_from_u64(uint64_t x) {
    constexpr double den = 1.0 / 9007199254740992.0; // 2^53
    return (double)((x >> 11) & ((1ULL << 53) - 1)) * den;
}

static inline uint8_t bits_to_geno_value(uint8_t bits) {
    if (bits == 0u) return 0u;
    if (bits == 2u) return 1u;
    if (bits == 3u) return 2u;
    return 0u;
}

static inline uint8_t geno_value_to_bits(uint8_t val) {
    return (val == 0u) ? 0u : (val == 1u) ? 2u : 3u;
}

static inline uint64_t hwe_seed_base(uint64_t impute_seed,
                                     int global_snp)
{
    uint64_t h = mix64_local(impute_seed);
    h ^= mix64_local((uint64_t)(uint32_t)global_snp + 0x243f6a8885a308d3ULL);
    return h;
}

static inline void hwe_thresholds_from_obs(long long obs_sum,
                                           long long nobs,
                                           double& q0,
                                           double& q01)
{
    double p = 0.0;
    if (nobs > 0) {
        p = (double)obs_sum / (2.0 * (double)nobs);
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
    }
    q0 = (1.0 - p) * (1.0 - p);
    q01 = q0 + 2.0 * p * (1.0 - p);
}

static inline uint8_t hwe_impute_value_prepared(double q0,
                                                double q01,
                                                uint64_t seed_base,
                                                int global_row)
{
    if (q0 >= 1.0) return 0u;
    if (q01 <= 0.0) return 2u;

    uint64_t h = seed_base;
    h ^= mix64_local((uint64_t)(uint32_t)global_row + 0x9e3779b97f4a7c15ULL);
    const double u = u01_from_u64(mix64_local(h));

    if (u < q0) return 0u;
    if (u < q01) return 1u;
    return 2u;
}

static inline int compute_mailman_segment_size_from_n(int64_t n_rows) {
    if (const char* s = std::getenv("SUMMIT_MAILMAN_SEGMENT_SIZE")) {
        const int v = std::atoi(s);
        if (v > 0) return std::min(v, 19);
    }
    if (n_rows <= 27) return 1;
    const double lg = std::log((double)n_rows) / std::log(3.0);

    // The original floor(log3(N)) - 2 choice is often too conservative for
    // N around 5k-20k on modern CPUs. Use one larger segment by default, but
    // cap at 8 so the per-thread lookup table stays cache-friendly.
    int seg = (int)std::floor(lg) - 1;
    if (seg < 1) seg = 1;
    if (seg > 8) seg = 8;
    return seg;
}

int compute_mailman_segment_size_optimized(int64_t n_rows) {
    return compute_mailman_segment_size_from_n(n_rows);
}

int64_t compute_mailman_table_size(int segment_size) {
    if (segment_size < 1) return 1;
    int64_t v = 1;
    for (int i = 0; i < segment_size; ++i) v *= 3;
    return v;
}

struct RowDecodePlan {
    const int* rows_ptr = nullptr;
    int N = -1;
    int N_total = -1;
    bool full_range = false;
    bool use_sparse = false;
    std::vector<uint32_t> row_byte;
    std::vector<uint8_t> row_shift;
};

static const RowDecodePlan& get_row_decode_plan(const std::vector<int>& rows,
                                                int N_total)
{
    static thread_local RowDecodePlan P;

    const int N = (int)rows.size();
    const int* rows_ptr = rows.empty() ? nullptr : rows.data();
    if (P.rows_ptr == rows_ptr && P.N == N && P.N_total == N_total) {
        return P;
    }

    P.rows_ptr = rows_ptr;
    P.N = N;
    P.N_total = N_total;
    P.row_byte.clear();
    P.row_shift.clear();

    P.full_range = (N == N_total);
    if (P.full_range) {
        for (int i = 0; i < N; ++i) {
            if (rows[(size_t)i] != i) {
                P.full_range = false;
                break;
            }
        }
    }

    double sparse_thresh = 0.60;
    if (const char* s = std::getenv("SUMMIT_DECODE_SPARSE_THRESHOLD")) {
        const double v = std::atof(s);
        if (v > 0.0 && v < 1.0) sparse_thresh = v;
    }

    const double density = (N_total > 0) ? ((double)N / (double)N_total) : 1.0;
    P.use_sparse = (!P.full_range && density < sparse_thresh);
    if (P.use_sparse) {
        P.row_byte.resize((size_t)N);
        P.row_shift.resize((size_t)N);
        for (int i = 0; i < N; ++i) {
            const uint32_t r = (uint32_t)rows[(size_t)i];
            P.row_byte[(size_t)i] = (r >> 2);
            P.row_shift[(size_t)i] = (uint8_t)((r & 3u) << 1);
        }
    }

    return P;
}

#if defined(__linux__)
// ---------------- mmap cache for .bed ----------------
struct BedMapping {
    int fd = -1;
    size_t size = 0;
    unsigned char* base = nullptr;
    ~BedMapping() {
        if (base && base != MAP_FAILED) ::munmap(base, size);
        if (fd >= 0) ::close(fd);
    }
};

// Align range to pages and madvise WILLNEED.
static inline void madvise_willneed_range(unsigned char* base, size_t file_size,
                                         size_t off, size_t len) {
    if (!base || len == 0) return;
    if (off >= file_size) return;
    size_t end = off + len;
    if (end > file_size) end = file_size;
    if (end <= off) return;

    const long ps = ::sysconf(_SC_PAGESIZE);
    const size_t pagesz = (ps > 0) ? (size_t)ps : 4096;

    uintptr_t a0 = (uintptr_t)(base + off);
    uintptr_t a1 = (uintptr_t)(base + end);
    uintptr_t p0 = a0 & ~(uintptr_t)(pagesz - 1);
    uintptr_t p1 = (a1 + pagesz - 1) & ~(uintptr_t)(pagesz - 1);
    size_t plen = (size_t)(p1 - p0);
    if (plen == 0) return;

    (void)::madvise((void*)p0, plen, MADV_WILLNEED);
}

static std::shared_ptr<BedMapping> get_bed_mapping_cached(const std::string& bed_path) {
    static std::mutex m;
    static std::unordered_map<std::string, std::shared_ptr<BedMapping>> cache;

    {
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(bed_path);
        if (it != cache.end() && it->second) return it->second;
    }

    auto mm = std::make_shared<BedMapping>();
    mm->fd = ::open(bed_path.c_str(), O_RDONLY);
    if (mm->fd < 0) throw std::runtime_error("Failed to open bed: " + bed_path);

    struct stat st{};
    if (::fstat(mm->fd, &st) != 0 || st.st_size < 3) {
        throw std::runtime_error("stat failed or BED too small: " + bed_path);
    }
    mm->size = (size_t)st.st_size;

    mm->base = (unsigned char*)::mmap(nullptr, mm->size, PROT_READ, MAP_SHARED, mm->fd, 0);
    if (mm->base == MAP_FAILED) {
        mm->base = nullptr;
        throw std::runtime_error("mmap failed for: " + bed_path);
    }

    const unsigned char b0 = mm->base[0];
    const unsigned char b1 = mm->base[1];
    const unsigned char b2 = mm->base[2];
    if (!(b0 == 0x6C && b1 == 0x1B && b2 == 0x01)) {
        throw std::runtime_error("Invalid BED magic/mode (expected 6C 1B 01) for: " + bed_path);
    }

    (void)::madvise(mm->base, mm->size, MADV_SEQUENTIAL);

    {
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(bed_path);
        if (it != cache.end() && it->second) return it->second;
        cache[bed_path] = mm;
    }
    return mm;
}
#endif // __linux__

void prefetch_bed_block(const std::string& bed_path,
                        const std::string& fam_path,
                        int blk_start, int blk_end,
                        int ahead_blocks)
{
#if defined(__linux__)
    const int64_t N_total = count_lines_cached(fam_path);
    if (N_total <= 0) return;

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;

    const int L = std::max(0, blk_end - blk_start);
    if (L <= 0) return;

    auto mm = get_bed_mapping_cached(bed_path);
    if (!mm || !mm->base) return;

    const size_t off0 = (size_t)3 + (size_t)blk_start * per_snp_bytes;
    const size_t len0 = (size_t)L * per_snp_bytes;
    madvise_willneed_range(mm->base, mm->size, off0, len0);

    const int ahead = std::max(0, ahead_blocks);
    if (ahead > 0) {
        const size_t off1 = (size_t)3 + (size_t)blk_end * per_snp_bytes;
        const size_t len1 = (size_t)ahead * (size_t)L * per_snp_bytes;
        madvise_willneed_range(mm->base, mm->size, off1, len1);
    }
#else
    (void)bed_path; (void)fam_path; (void)blk_start; (void)blk_end; (void)ahead_blocks;
#endif
}

// ---------------- decode: selected rows only ----------------
// rows is ALWAYS sorted ascending.

static inline void decode_rows_codes_dense_sorted_into(const unsigned char* bytes,
                                                       int N_total,
                                                       const std::vector<int>& rows,
                                                       uint8_t* __restrict codeN,
                                                       long long& nobs,
                                                       long long& sum,
                                                       long long& sumsq,
                                                       std::vector<int>* miss_idx = nullptr)
{
    const int nbytes = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    nobs = 0; sum = 0; sumsq = 0;
    if (miss_idx) miss_idx->clear();

    const int N = (int)rows.size();
    if (N == 0) return;

    int gidx = 0;
    int j = 0;
    int next = rows[0];

    for (int b = 0; b < nbytes; ++b) {
        const unsigned char c = bytes[b];
        for (int t = 0; t < 4 && gidx < N_total; ++t, ++gidx) {
            if (gidx != next) continue;

            const uint8_t bits = (uint8_t)((c >> (2 * t)) & 0x3);
            codeN[j] = bits;

            if (bits != 1u) {
                ++nobs;
                if (bits == 2u) { sum += 1; sumsq += 1; }
                else if (bits == 3u) { sum += 2; sumsq += 4; }
            } else if (miss_idx) {
                miss_idx->push_back(j);
            }

            ++j;
            if (j >= N) return;
            next = rows[(size_t)j];
        }
    }
}

static inline void decode_rows_codes_sparse_precomp_into(
    const unsigned char* bytes,
    const uint32_t* __restrict row_byte,
    const uint8_t*  __restrict row_shift,
    int N,
    uint8_t* __restrict codeN,
    long long& nobs,
    long long& sum,
    long long& sumsq,
    std::vector<int>* miss_idx = nullptr)
{
    nobs = 0; sum = 0; sumsq = 0;
    if (miss_idx) miss_idx->clear();

    for (int j = 0; j < N; ++j) {
        const unsigned char c = bytes[row_byte[(size_t)j]];
        const uint8_t bits = (uint8_t)((c >> row_shift[(size_t)j]) & 0x3);
        codeN[(size_t)j] = bits;

        if (bits != 1u) {
            ++nobs;
            if (bits == 2u) { sum += 1; sumsq += 1; }
            else if (bits == 3u) { sum += 2; sumsq += 4; }
        } else if (miss_idx) {
            miss_idx->push_back(j);
        }
    }
}

static inline void decode_all_rows_codes_into(const unsigned char* bytes,
                                              int N_total,
                                              uint8_t* __restrict codeN,
                                              long long& nobs,
                                              long long& sum,
                                              long long& sumsq,
                                              std::vector<int>* miss_idx = nullptr)
{
    const int nbytes = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    nobs = 0; sum = 0; sumsq = 0;
    if (miss_idx) miss_idx->clear();

    int j = 0;
    for (int b = 0; b < nbytes; ++b) {
        const unsigned char c = bytes[b];
        for (int t = 0; t < 4 && j < N_total; ++t, ++j) {
            const uint8_t bits = (uint8_t)((c >> (2 * t)) & 0x3);
            codeN[j] = bits;

            if (bits != 1u) {
                ++nobs;
                if (bits == 2u) { sum += 1; sumsq += 1; }
                else if (bits == 3u) { sum += 2; sumsq += 4; }
            } else if (miss_idx) {
                miss_idx->push_back(j);
            }
        }
    }
}

static inline void impute_hwe_missing_inplace(uint8_t* __restrict codes,
                                             const std::vector<int>& miss_idx,
                                             const std::vector<int>& rows,
                                             double q0,
                                             double q01,
                                             uint64_t seed_base,
                                             long long obs_sum,
                                             long long obs_sumsq,
                                             long long& sum,
                                             long long& sumsq)
{
    sum = obs_sum;
    sumsq = obs_sumsq;
    for (int idx : miss_idx) {
        const uint8_t val = hwe_impute_value_prepared(q0, q01, seed_base, rows[(size_t)idx]);
        codes[(size_t)idx] = geno_value_to_bits(val);
        sum += (long long)val;
        sumsq += (long long)val * (long long)val;
    }
}

template <typename T>
static inline void write_completed_standardized_from_codes(const uint8_t* __restrict codes,
                                                           int N,
                                                           double mean,
                                                           double inv_std,
                                                           T* __restrict dst)
{
    const T g0 = (T)((0.0 - mean) * inv_std);
    const T g1 = (T)((1.0 - mean) * inv_std);
    const T g2 = (T)((2.0 - mean) * inv_std);
#ifdef _OPENMP
    #pragma omp simd
#endif
    for (int i = 0; i < N; ++i) {
        const uint8_t bits = codes[(size_t)i];
        dst[(size_t)i] = (bits == 0u) ? g0 : (bits == 2u) ? g1 : g2;
    }
}

template <typename CodeT>
static inline void pack_completed_codes(const uint8_t* __restrict codes,
                                        int N,
                                        CodeT* __restrict packed_seg)
{
#ifdef _OPENMP
    #pragma omp simd
#endif
    for (int i = 0; i < N; ++i) {
        packed_seg[(size_t)i] = (CodeT)(3 * packed_seg[(size_t)i] + (CodeT)bits_to_geno_value(codes[(size_t)i]));
    }
}

static inline void compute_mean_invstd_from_completed(long long sum,
                                                      long long sumsq,
                                                      int N,
                                                      int ddof,
                                                      double& mean,
                                                      double& inv_std)
{
    mean = (N > 0) ? ((double)sum / (double)N) : 0.0;
    inv_std = 1.0;
    const long long denom_ll = (long long)N - (long long)ddof;
    if (N <= 0 || denom_ll <= 0) return;

    double M2 = (double)sumsq - ((double)sum * (double)sum) / (double)N;
    if (M2 < 0.0 && M2 > -1e-12) M2 = 0.0;
    if (M2 > 0.0) {
        const double var = M2 / (double)denom_ll;
        if (var > 0.0 && std::isfinite(var)) {
            inv_std = 1.0 / std::sqrt(var);
        }
    }
}

template <typename T>
static inline void write_mean_imputed_standardized(const uint8_t* __restrict codes,
                                                   int N,
                                                   long long nobs,
                                                   long long sum,
                                                   long long sumsq,
                                                   int ddof,
                                                   T* __restrict dst)
{
    const double dnobs = (double)nobs;
    const long long denom_ll = nobs - (long long)ddof;

    const double mu = (nobs > 0) ? ((double)sum / dnobs) : 0.0;
    double M2 = 0.0;
    if (nobs > 0) {
        M2 = (double)sumsq - ((double)sum * (double)sum) / dnobs;
        if (M2 < 0.0 && M2 > -1e-12) M2 = 0.0;
    }

    double inv_sd = 1.0;
    if (denom_ll > 0 && M2 > 0.0) {
        const double var = M2 / (double)denom_ll;
        if (var > 0.0) inv_sd = 1.0 / std::sqrt(var);
    }

    const double vstd[4] = {
        (0.0 - mu) * inv_sd,
        0.0,
        (1.0 - mu) * inv_sd,
        (2.0 - mu) * inv_sd
    };

#ifdef _OPENMP
    #pragma omp simd
#endif
    for (int i = 0; i < N; ++i) {
        dst[(size_t)i] = (T)vstd[codes[(size_t)i]];
    }
}

// ---------------- core implementation (templated, TU-local) ----------------
template <typename T>
static void read_block_standardized_impl(const std::string &bed_path,
                                         const std::string &fam_path,
                                         int blk_start, int blk_end,
                                         const std::vector<int> &rows,
                                         int ddof,
                                         ImputeMode impute_mode,
                                         uint64_t impute_seed,
                                         std::vector<T> &Geno,
                                         int &N, int &L)
{
    const int64_t N_total64 = count_lines_cached(fam_path);
    if (N_total64 <= 0) throw std::runtime_error("FAM has zero rows: " + fam_path);
    const int N_total = (int)N_total64;

    if (blk_end <= blk_start) {
        N = (int)rows.size();
        L = 0;
        return;
    }

    L = blk_end - blk_start;
    N = (int)rows.size();
    if (N <= 0 || L <= 0) return;

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;
    const size_t need = (size_t)N * (size_t)L;

    if (Geno.size() < need) Geno.resize(need);

    int decode_threads = 0;
    if (const char* s = std::getenv("SUMMIT_DECODE_THREADS")) decode_threads = std::atoi(s);

    int hw = 1;
#ifdef _OPENMP
    hw = affinity_thread_cap();
    if (hw <= 0) hw = 1;
#endif

    if (decode_threads <= 0) {
        const int cap_env = env_int("SUMMIT_DECODE_THREADS_CAP", 16);
        int cap = std::min(cap_env, hw);
        cap = std::min(cap, std::max(1, L));

        const uint64_t work_elems = (uint64_t)N * (uint64_t)L;
        uint64_t target = std::is_same_v<T,double> ? 4ULL*1000*1000 : 8ULL*1000*1000;
        if (const char* s = std::getenv("SUMMIT_DECODE_TARGET_ELEMS_PER_THR")) {
            long long v = std::atoll(s);
            if (v > 0) target = (uint64_t)v;
        }

        int by_work = 1;
        if (target > 0) {
            by_work = (int)((work_elems + target - 1) / target);
            if (by_work < 1) by_work = 1;
        }

        decode_threads = std::min(cap, by_work);

        if (work_elems < 2ULL*1000*1000) decode_threads = 1;
        else if (work_elems < 16ULL*1000*1000) decode_threads = std::min(decode_threads, 2);
        else if (work_elems < 64ULL*1000*1000) decode_threads = std::min(decode_threads, 4);

        if (decode_threads < 1) decode_threads = 1;
    }

    decode_threads = std::min(decode_threads, std::max(1, L));

    const RowDecodePlan& plan = get_row_decode_plan(rows, N_total);

#if defined(__linux__)
    auto mm = get_bed_mapping_cached(bed_path);

    const size_t need_bytes = (size_t)3 + (size_t)blk_end * per_snp_bytes;
    if (!mm || !mm->base || need_bytes > mm->size) {
        throw std::runtime_error("BED file too small for requested block: " + bed_path);
    }

    if (std::getenv("SUMMIT_INTERNAL_PREFETCH") != nullptr) {
        int ahead = env_int("SUMMIT_PREFETCH_AHEAD_BLKS", 1);
        if (ahead < 0) ahead = 0;

        const size_t off0 = (size_t)3 + (size_t)blk_start * per_snp_bytes;
        const size_t len0 = (size_t)L * per_snp_bytes;
        madvise_willneed_range(mm->base, mm->size, off0, len0);

        if (ahead > 0) {
            const size_t off1 = (size_t)3 + (size_t)blk_end * per_snp_bytes;
            const size_t len1 = (size_t)ahead * (size_t)L * per_snp_bytes;
            madvise_willneed_range(mm->base, mm->size, off1, len1);
        }
    }

    const unsigned char* snp0 = mm->base + 3 + (size_t)blk_start * per_snp_bytes;

    auto worker = [&](int col, uint8_t* __restrict codes, std::vector<int>& miss_idx) {
        const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

        long long nobs = 0, sum = 0, sumsq = 0;
        std::vector<int>* miss_ptr = (impute_mode == ImputeMode::Hwe) ? &miss_idx : nullptr;
        if (plan.full_range) {
            decode_all_rows_codes_into(bytes, N_total, codes, nobs, sum, sumsq, miss_ptr);
        } else if (plan.use_sparse) {
            decode_rows_codes_sparse_precomp_into(
                bytes,
                plan.row_byte.data(),
                plan.row_shift.data(),
                N,
                codes,
                nobs, sum, sumsq,
                miss_ptr
            );
        } else {
            decode_rows_codes_dense_sorted_into(bytes, N_total, rows, codes, nobs, sum, sumsq, miss_ptr);
        }

        T* dst = Geno.data() + (size_t)col * (size_t)N;
        if (impute_mode == ImputeMode::Hwe) {
            if (nobs == (long long)N) {
                double mean = 0.0, inv_std = 1.0;
                compute_mean_invstd_from_completed(sum, sumsq, N, ddof, mean, inv_std);
                const double g0 = (0.0 - mean) * inv_std;
                const double g1 = (1.0 - mean) * inv_std;
                const double g2 = (2.0 - mean) * inv_std;
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int i = 0; i < N; ++i) {
                    const uint8_t bits = codes[(size_t)i];
                    dst[(size_t)i] = (bits == 0u) ? (T)g0 : (bits == 2u) ? (T)g1 : (T)g2;
                }
            } else {
                double q0 = 0.0, q01 = 0.0;
                hwe_thresholds_from_obs(sum, nobs, q0, q01);
                const uint64_t seed_base = hwe_seed_base(impute_seed, blk_start + col);
                long long sum_full = 0, sumsq_full = 0;
                impute_hwe_missing_inplace(codes, miss_idx, rows, q0, q01, seed_base, sum, sumsq, sum_full, sumsq_full);
                double mean = 0.0, inv_std = 1.0;
                compute_mean_invstd_from_completed(sum_full, sumsq_full, N, ddof, mean, inv_std);
                write_completed_standardized_from_codes(codes, N, mean, inv_std, dst);
            }
        } else {
            write_mean_imputed_standardized(codes, N, nobs, sum, sumsq, ddof, dst);
        }
    };

    if (decode_threads <= 1) {
        static thread_local std::vector<uint8_t> codes_local;
        static thread_local std::vector<int> miss_local;
        if ((int)codes_local.size() < N) codes_local.resize((size_t)N);
        for (int col = 0; col < L; ++col) {
            worker(col, codes_local.data(), miss_local);
        }
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(decode_threads)
#endif
    {
        static thread_local std::vector<uint8_t> codes_local;
        static thread_local std::vector<int> miss_local;
        if ((int)codes_local.size() < N) codes_local.resize((size_t)N);

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int col = 0; col < L; ++col) {
            worker(col, codes_local.data(), miss_local);
        }
    }

#else
    std::ifstream bed(bed_path, std::ios::binary);
    if (!bed) throw std::runtime_error("Failed to open bed: " + bed_path);

    unsigned char magic[3];
    bed.read(reinterpret_cast<char*>(magic), 3);
    if (!bed) throw std::runtime_error("BED header read failed: " + bed_path);
    if (magic[0] != 0x6c || magic[1] != 0x1b)
        throw std::runtime_error("BED magic bytes mismatch (expected 0x6c 0x1b): " + bed_path);
    if (magic[2] != 0x01)
        throw std::runtime_error("BED file is not SNP-major (third byte != 0x01): " + bed_path);

    const std::streamoff offset =
        3 + (std::streamoff)blk_start * (std::streamoff)per_snp_bytes;
    bed.seekg(offset, std::ios::beg);
    if (!bed) throw std::runtime_error("BED seekg failed: " + bed_path);

    std::vector<unsigned char> line((size_t)nbytes_per_snp);
    std::vector<uint8_t> codes_local((size_t)N);
    std::vector<int> miss_local;

    for (int col = 0; col < L; ++col) {
        bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
        if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

        long long nobs = 0, sum = 0, sumsq = 0;
        std::vector<int>* miss_ptr = (impute_mode == ImputeMode::Hwe) ? &miss_local : nullptr;
        if (plan.full_range) {
            decode_all_rows_codes_into(line.data(), N_total, codes_local.data(), nobs, sum, sumsq, miss_ptr);
        } else if (plan.use_sparse) {
            decode_rows_codes_sparse_precomp_into(
                line.data(),
                plan.row_byte.data(),
                plan.row_shift.data(),
                N,
                codes_local.data(),
                nobs, sum, sumsq,
                miss_ptr
            );
        } else {
            decode_rows_codes_dense_sorted_into(line.data(), N_total, rows, codes_local.data(), nobs, sum, sumsq, miss_ptr);
        }

        T* dst = Geno.data() + (size_t)col * (size_t)N;
        if (impute_mode == ImputeMode::Hwe) {
            if (nobs == (long long)N) {
                double mean = 0.0, inv_std = 1.0;
                compute_mean_invstd_from_completed(sum, sumsq, N, ddof, mean, inv_std);
                const double g0 = (0.0 - mean) * inv_std;
                const double g1 = (1.0 - mean) * inv_std;
                const double g2 = (2.0 - mean) * inv_std;
                for (int i = 0; i < N; ++i) {
                    const uint8_t bits = codes_local[(size_t)i];
                    dst[(size_t)i] = (bits == 0u) ? (T)g0 : (bits == 2u) ? (T)g1 : (T)g2;
                }
            } else {
                double q0 = 0.0, q01 = 0.0;
                hwe_thresholds_from_obs(sum, nobs, q0, q01);
                const uint64_t seed_base = hwe_seed_base(impute_seed, blk_start + col);
                long long sum_full = 0, sumsq_full = 0;
                impute_hwe_missing_inplace(codes_local.data(), miss_local, rows, q0, q01, seed_base, sum, sumsq, sum_full, sumsq_full);
                double mean = 0.0, inv_std = 1.0;
                compute_mean_invstd_from_completed(sum_full, sumsq_full, N, ddof, mean, inv_std);
                write_completed_standardized_from_codes(codes_local.data(), N, mean, inv_std, dst);
            }
        } else {
            write_mean_imputed_standardized(codes_local.data(), N, nobs, sum, sumsq, ddof, dst);
        }
    }
#endif
}

void read_block_mailman_hwe(const std::string& bed_path,
                            const std::string& fam_path,
                            int blk_start,
                            int blk_end,
                            const std::vector<int>& rows,
                            int ddof,
                            uint64_t impute_seed,
                            MailmanPackedBlock& out)
{
    const int64_t N_total64 = count_lines_cached(fam_path);
    if (N_total64 <= 0) throw std::runtime_error("FAM has zero rows: " + fam_path);
    const int N_total = (int)N_total64;

    const int N = (int)rows.size();
    const int L = std::max(0, blk_end - blk_start);

    out.N = N;
    out.L = L;
    out.segment_size = compute_mailman_segment_size_from_n(N);
    out.n_segments = (L > 0) ? ((L + out.segment_size - 1) / out.segment_size) : 0;
    out.table_size = compute_mailman_table_size(out.segment_size);
    out.use_u16 = (out.table_size <= (int64_t)std::numeric_limits<uint16_t>::max());
    if (out.use_u16) {
        out.packed16.resize((size_t)out.n_segments * (size_t)N);
        out.packed32.clear();
    } else {
        out.packed32.resize((size_t)out.n_segments * (size_t)N);
        out.packed16.clear();
    }
    out.mean.resize((size_t)L);
    out.inv_std.resize((size_t)L);

    if (N <= 0 || L <= 0) return;

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;
    const RowDecodePlan& plan = get_row_decode_plan(rows, N_total);

#if defined(__linux__)
    auto mm = get_bed_mapping_cached(bed_path);
    const size_t need_bytes = (size_t)3 + (size_t)blk_end * per_snp_bytes;
    if (!mm || !mm->base || need_bytes > mm->size) {
        throw std::runtime_error("BED file too small for requested block: " + bed_path);
    }

    const unsigned char* snp0 = mm->base + 3 + (size_t)blk_start * per_snp_bytes;

    int n_threads = 1;
#ifdef _OPENMP
    if (const char* s = std::getenv("SUMMIT_MAILMAN_PACK_THREADS")) {
        const int v = std::atoi(s);
        if (v > 0) n_threads = v;
        else n_threads = omp_get_max_threads();
    } else {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min((int)out.n_segments, std::max(1, n_threads));
#endif

#ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads)
#endif
    {
        static thread_local std::vector<uint8_t> codes_local;
        static thread_local std::vector<int> miss_local;
        if ((int)codes_local.size() < N) codes_local.resize((size_t)N);

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int64_t seg = 0; seg < out.n_segments; ++seg) {
            const int c0 = (int)(seg * (int64_t)out.segment_size);
            const int c1 = std::min(L, c0 + out.segment_size);

            if (out.use_u16) {
                uint16_t* packed_seg = out.packed16.data() + (size_t)seg * (size_t)N;
                std::memset(packed_seg, 0, (size_t)N * sizeof(uint16_t));
                for (int col = c0; col < c1; ++col) {
                    const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

                    long long nobs = 0, sum = 0, sumsq = 0;
                    if (plan.full_range) {
                        decode_all_rows_codes_into(bytes, N_total, codes_local.data(), nobs, sum, sumsq, &miss_local);
                    } else if (plan.use_sparse) {
                        decode_rows_codes_sparse_precomp_into(
                            bytes,
                            plan.row_byte.data(),
                            plan.row_shift.data(),
                            N,
                            codes_local.data(),
                            nobs, sum, sumsq,
                            &miss_local
                        );
                    } else {
                        decode_rows_codes_dense_sorted_into(bytes, N_total, rows, codes_local.data(), nobs, sum, sumsq, &miss_local);
                    }

                    if (nobs == (long long)N) {
                        double mean = 0.0, inv_std = 1.0;
                        compute_mean_invstd_from_completed(sum, sumsq, N, ddof, mean, inv_std);
                        out.mean[(size_t)col] = mean;
                        out.inv_std[(size_t)col] = inv_std;
                        pack_completed_codes(codes_local.data(), N, packed_seg);
                    } else {
                        double q0 = 0.0, q01 = 0.0;
                        hwe_thresholds_from_obs(sum, nobs, q0, q01);
                        const uint64_t seed_base = hwe_seed_base(impute_seed, blk_start + col);
                        long long sum_full = 0, sumsq_full = 0;
                        impute_hwe_missing_inplace(codes_local.data(), miss_local, rows, q0, q01, seed_base, sum, sumsq, sum_full, sumsq_full);

                        double mean = 0.0, inv_std = 1.0;
                        compute_mean_invstd_from_completed(sum_full, sumsq_full, N, ddof, mean, inv_std);
                        out.mean[(size_t)col] = mean;
                        out.inv_std[(size_t)col] = inv_std;
                        pack_completed_codes(codes_local.data(), N, packed_seg);
                    }
                }
            } else {
                uint32_t* packed_seg = out.packed32.data() + (size_t)seg * (size_t)N;
                std::memset(packed_seg, 0, (size_t)N * sizeof(uint32_t));
                for (int col = c0; col < c1; ++col) {
                    const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

                    long long nobs = 0, sum = 0, sumsq = 0;
                    if (plan.full_range) {
                        decode_all_rows_codes_into(bytes, N_total, codes_local.data(), nobs, sum, sumsq, &miss_local);
                    } else if (plan.use_sparse) {
                        decode_rows_codes_sparse_precomp_into(
                            bytes,
                            plan.row_byte.data(),
                            plan.row_shift.data(),
                            N,
                            codes_local.data(),
                            nobs, sum, sumsq,
                            &miss_local
                        );
                    } else {
                        decode_rows_codes_dense_sorted_into(bytes, N_total, rows, codes_local.data(), nobs, sum, sumsq, &miss_local);
                    }

                    if (nobs == (long long)N) {
                        double mean = 0.0, inv_std = 1.0;
                        compute_mean_invstd_from_completed(sum, sumsq, N, ddof, mean, inv_std);
                        out.mean[(size_t)col] = mean;
                        out.inv_std[(size_t)col] = inv_std;
                        pack_completed_codes(codes_local.data(), N, packed_seg);
                    } else {
                        double q0 = 0.0, q01 = 0.0;
                        hwe_thresholds_from_obs(sum, nobs, q0, q01);
                        const uint64_t seed_base = hwe_seed_base(impute_seed, blk_start + col);
                        long long sum_full = 0, sumsq_full = 0;
                        impute_hwe_missing_inplace(codes_local.data(), miss_local, rows, q0, q01, seed_base, sum, sumsq, sum_full, sumsq_full);

                        double mean = 0.0, inv_std = 1.0;
                        compute_mean_invstd_from_completed(sum_full, sumsq_full, N, ddof, mean, inv_std);
                        out.mean[(size_t)col] = mean;
                        out.inv_std[(size_t)col] = inv_std;
                        pack_completed_codes(codes_local.data(), N, packed_seg);
                    }
                }
            }
        }
    }
#else
    std::ifstream bed(bed_path, std::ios::binary);
    if (!bed) throw std::runtime_error("Failed to open bed: " + bed_path);

    unsigned char magic[3];
    bed.read(reinterpret_cast<char*>(magic), 3);
    if (!bed) throw std::runtime_error("BED header read failed: " + bed_path);
    if (magic[0] != 0x6c || magic[1] != 0x1b)
        throw std::runtime_error("BED magic bytes mismatch (expected 0x6c 0x1b): " + bed_path);
    if (magic[2] != 0x01)
        throw std::runtime_error("BED file is not SNP-major (third byte != 0x01): " + bed_path);

    std::vector<unsigned char> line((size_t)nbytes_per_snp);
    std::vector<uint8_t> codes_local((size_t)N);
    std::vector<int> miss_local;

    for (int64_t seg = 0; seg < out.n_segments; ++seg) {
        const int c0 = (int)(seg * (int64_t)out.segment_size);
        const int c1 = std::min(L, c0 + out.segment_size);

        if (out.use_u16) {
            uint16_t* packed_seg = out.packed16.data() + (size_t)seg * (size_t)N;
            std::memset(packed_seg, 0, (size_t)N * sizeof(uint16_t));
            for (int col = c0; col < c1; ++col) {
                const std::streamoff offset = 3 + (std::streamoff)(blk_start + col) * (std::streamoff)per_snp_bytes;
                bed.clear();
                bed.seekg(offset, std::ios::beg);
                if (!bed) throw std::runtime_error("BED seekg failed: " + bed_path);
                bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
                if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

                long long nobs = 0, sum = 0, sumsq = 0;
                if (plan.full_range) {
                    decode_all_rows_codes_into(line.data(), N_total, codes_local.data(), nobs, sum, sumsq, &miss_local);
                } else if (plan.use_sparse) {
                    decode_rows_codes_sparse_precomp_into(
                        line.data(),
                        plan.row_byte.data(),
                        plan.row_shift.data(),
                        N,
                        codes_local.data(),
                        nobs, sum, sumsq,
                        &miss_local
                    );
                } else {
                    decode_rows_codes_dense_sorted_into(line.data(), N_total, rows, codes_local.data(), nobs, sum, sumsq, &miss_local);
                }

                if (nobs == (long long)N) {
                    double mean = 0.0, inv_std = 1.0;
                    compute_mean_invstd_from_completed(sum, sumsq, N, ddof, mean, inv_std);
                    out.mean[(size_t)col] = mean;
                    out.inv_std[(size_t)col] = inv_std;
                    pack_completed_codes(codes_local.data(), N, packed_seg);
                } else {
                    double q0 = 0.0, q01 = 0.0;
                    hwe_thresholds_from_obs(sum, nobs, q0, q01);
                    const uint64_t seed_base = hwe_seed_base(impute_seed, blk_start + col);
                    long long sum_full = 0, sumsq_full = 0;
                    impute_hwe_missing_inplace(codes_local.data(), miss_local, rows, q0, q01, seed_base, sum, sumsq, sum_full, sumsq_full);

                    double mean = 0.0, inv_std = 1.0;
                    compute_mean_invstd_from_completed(sum_full, sumsq_full, N, ddof, mean, inv_std);
                    out.mean[(size_t)col] = mean;
                    out.inv_std[(size_t)col] = inv_std;
                    pack_completed_codes(codes_local.data(), N, packed_seg);
                }
            }
        } else {
            uint32_t* packed_seg = out.packed32.data() + (size_t)seg * (size_t)N;
            std::memset(packed_seg, 0, (size_t)N * sizeof(uint32_t));
            for (int col = c0; col < c1; ++col) {
                const std::streamoff offset = 3 + (std::streamoff)(blk_start + col) * (std::streamoff)per_snp_bytes;
                bed.clear();
                bed.seekg(offset, std::ios::beg);
                if (!bed) throw std::runtime_error("BED seekg failed: " + bed_path);
                bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
                if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

                long long nobs = 0, sum = 0, sumsq = 0;
                if (plan.full_range) {
                    decode_all_rows_codes_into(line.data(), N_total, codes_local.data(), nobs, sum, sumsq, &miss_local);
                } else if (plan.use_sparse) {
                    decode_rows_codes_sparse_precomp_into(
                        line.data(),
                        plan.row_byte.data(),
                        plan.row_shift.data(),
                        N,
                        codes_local.data(),
                        nobs, sum, sumsq,
                        &miss_local
                    );
                } else {
                    decode_rows_codes_dense_sorted_into(line.data(), N_total, rows, codes_local.data(), nobs, sum, sumsq, &miss_local);
                }

                if (nobs == (long long)N) {
                    double mean = 0.0, inv_std = 1.0;
                    compute_mean_invstd_from_completed(sum, sumsq, N, ddof, mean, inv_std);
                    out.mean[(size_t)col] = mean;
                    out.inv_std[(size_t)col] = inv_std;
                    pack_completed_codes(codes_local.data(), N, packed_seg);
                } else {
                    double q0 = 0.0, q01 = 0.0;
                    hwe_thresholds_from_obs(sum, nobs, q0, q01);
                    const uint64_t seed_base = hwe_seed_base(impute_seed, blk_start + col);
                    long long sum_full = 0, sumsq_full = 0;
                    impute_hwe_missing_inplace(codes_local.data(), miss_local, rows, q0, q01, seed_base, sum, sumsq, sum_full, sumsq_full);

                    double mean = 0.0, inv_std = 1.0;
                    compute_mean_invstd_from_completed(sum_full, sumsq_full, N, ddof, mean, inv_std);
                    out.mean[(size_t)col] = mean;
                    out.inv_std[(size_t)col] = inv_std;
                    pack_completed_codes(codes_local.data(), N, packed_seg);
                }
            }
        }
    }
#endif
}

void read_block_standardized_float(const std::string& bed_path,
                                  const std::string& fam_path,
                                  int blk_start, int blk_end,
                                  const std::vector<int>& rows,
                                  int ddof,
                                  ImputeMode impute_mode,
                                  uint64_t impute_seed,
                                  std::vector<float>& Geno,
                                  int& N, int& L)
{
    read_block_standardized_impl<float>(bed_path, fam_path, blk_start, blk_end,
                                        rows, ddof, impute_mode, impute_seed,
                                        Geno, N, L);
}

void read_block_standardized_double(const std::string& bed_path,
                                   const std::string& fam_path,
                                   int blk_start, int blk_end,
                                   const std::vector<int>& rows,
                                   int ddof,
                                   ImputeMode impute_mode,
                                   uint64_t impute_seed,
                                   std::vector<double>& Geno,
                                   int& N, int& L)
{
    read_block_standardized_impl<double>(bed_path, fam_path, blk_start, blk_end,
                                         rows, ddof, impute_mode, impute_seed,
                                         Geno, N, L);
}

void compute_maf_block(const std::string& bed_path,
                       const std::string& fam_path,
                       int blk_start,
                       int blk_end,
                       const std::vector<int>& rows,
                       std::vector<double>& maf)
{
    const int64_t N_total64 = count_lines_cached(fam_path);
    if (N_total64 <= 0) throw std::runtime_error("FAM has zero rows: " + fam_path);
    const int N_total = (int)N_total64;

    const int N = (int)rows.size();
    const int L = std::max(0, blk_end - blk_start);
    maf.assign((size_t)L, 0.0);
    if (N <= 0 || L <= 0) return;

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;
    const RowDecodePlan& plan = get_row_decode_plan(rows, N_total);

    int decode_threads = 0;
    if (const char* s = std::getenv("SUMMIT_DECODE_THREADS")) decode_threads = std::atoi(s);

    int hw = 1;
#ifdef _OPENMP
    hw = affinity_thread_cap();
    if (hw <= 0) hw = 1;
#endif
    if (decode_threads <= 0) {
        const int cap_env = env_int("SUMMIT_DECODE_THREADS_CAP", 16);
        int cap = std::min(cap_env, hw);
        cap = std::min(cap, std::max(1, L));

        const uint64_t work_elems = (uint64_t)N * (uint64_t)L;
        uint64_t target = 8ULL * 1000 * 1000;
        if (const char* s = std::getenv("SUMMIT_DECODE_TARGET_ELEMS_PER_THR")) {
            long long v = std::atoll(s);
            if (v > 0) target = (uint64_t)v;
        }

        int by_work = 1;
        if (target > 0) {
            by_work = (int)((work_elems + target - 1) / target);
            if (by_work < 1) by_work = 1;
        }

        decode_threads = std::min(cap, by_work);
        if (work_elems < 2ULL * 1000 * 1000) decode_threads = 1;
        else if (work_elems < 16ULL * 1000 * 1000) decode_threads = std::min(decode_threads, 2);
        else if (work_elems < 64ULL * 1000 * 1000) decode_threads = std::min(decode_threads, 4);
        if (decode_threads < 1) decode_threads = 1;
    }
    decode_threads = std::min(decode_threads, std::max(1, L));

#if defined(__linux__)
    auto mm = get_bed_mapping_cached(bed_path);
    const size_t need_bytes = (size_t)3 + (size_t)blk_end * per_snp_bytes;
    if (!mm || !mm->base || need_bytes > mm->size) {
        throw std::runtime_error("BED file too small for requested block: " + bed_path);
    }
    const unsigned char* snp0 = mm->base + 3 + (size_t)blk_start * per_snp_bytes;

    auto worker = [&](int col, uint8_t* __restrict codes) {
        const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;
        long long nobs = 0, sum = 0, sumsq = 0;
        if (plan.full_range) {
            decode_all_rows_codes_into(bytes, N_total, codes, nobs, sum, sumsq, nullptr);
        } else if (plan.use_sparse) {
            decode_rows_codes_sparse_precomp_into(bytes,
                                                  plan.row_byte.data(),
                                                  plan.row_shift.data(),
                                                  N,
                                                  codes,
                                                  nobs, sum, sumsq,
                                                  nullptr);
        } else {
            decode_rows_codes_dense_sorted_into(bytes, N_total, rows, codes, nobs, sum, sumsq, nullptr);
        }

        double p = 0.0;
        if (nobs > 0) {
            p = (double)sum / (2.0 * (double)nobs);
            if (p < 0.0) p = 0.0;
            if (p > 1.0) p = 1.0;
        }
        maf[(size_t)col] = std::min(p, 1.0 - p);
    };

    if (decode_threads <= 1) {
        static thread_local std::vector<uint8_t> codes_local;
        if ((int)codes_local.size() < N) codes_local.resize((size_t)N);
        for (int col = 0; col < L; ++col) worker(col, codes_local.data());
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(decode_threads)
#endif
    {
        static thread_local std::vector<uint8_t> codes_local;
        if ((int)codes_local.size() < N) codes_local.resize((size_t)N);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int col = 0; col < L; ++col) worker(col, codes_local.data());
    }
#else
    std::ifstream bed(bed_path, std::ios::binary);
    if (!bed) throw std::runtime_error("Failed to open bed: " + bed_path);

    unsigned char magic[3];
    bed.read(reinterpret_cast<char*>(magic), 3);
    if (!bed) throw std::runtime_error("BED header read failed: " + bed_path);
    if (magic[0] != 0x6c || magic[1] != 0x1b)
        throw std::runtime_error("BED magic bytes mismatch (expected 0x6c 0x1b): " + bed_path);
    if (magic[2] != 0x01)
        throw std::runtime_error("BED file is not SNP-major (third byte != 0x01): " + bed_path);

    std::vector<unsigned char> line((size_t)nbytes_per_snp);
    std::vector<uint8_t> codes_local((size_t)N);
    for (int col = 0; col < L; ++col) {
        const std::streamoff offset = 3 + (std::streamoff)(blk_start + col) * (std::streamoff)per_snp_bytes;
        bed.clear();
        bed.seekg(offset, std::ios::beg);
        if (!bed) throw std::runtime_error("BED seekg failed: " + bed_path);
        bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
        if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

        long long nobs = 0, sum = 0, sumsq = 0;
        if (plan.full_range) {
            decode_all_rows_codes_into(line.data(), N_total, codes_local.data(), nobs, sum, sumsq, nullptr);
        } else if (plan.use_sparse) {
            decode_rows_codes_sparse_precomp_into(line.data(),
                                                  plan.row_byte.data(),
                                                  plan.row_shift.data(),
                                                  N,
                                                  codes_local.data(),
                                                  nobs, sum, sumsq,
                                                  nullptr);
        } else {
            decode_rows_codes_dense_sorted_into(line.data(), N_total, rows, codes_local.data(), nobs, sum, sumsq, nullptr);
        }

        double p = 0.0;
        if (nobs > 0) {
            p = (double)sum / (2.0 * (double)nobs);
            if (p < 0.0) p = 0.0;
            if (p > 1.0) p = 1.0;
        }
        maf[(size_t)col] = std::min(p, 1.0 - p);
    }
#endif
}
