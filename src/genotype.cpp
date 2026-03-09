// src/genotype.cpp
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
    // IMPORTANT: use shared_ptr, NOT weak_ptr; otherwise the mapping dies after each call.
    static std::unordered_map<std::string, std::shared_ptr<BedMapping>> cache;

    // Fast path: existing mapping kept alive by shared_ptr in the map
    {
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(bed_path);
        if (it != cache.end() && it->second) return it->second;
    }

    // Build mapping outside the lock (avoid blocking other threads)
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

    // Magic bytes check (PLINK .bed): 0x6C 0x1B, mode 0x01 = SNP-major
    const unsigned char b0 = mm->base[0];
    const unsigned char b1 = mm->base[1];
    const unsigned char b2 = mm->base[2];
    if (!(b0 == 0x6C && b1 == 0x1B && b2 == 0x01)) {
        throw std::runtime_error("Invalid BED magic/mode (expected 6C 1B 01) for: " + bed_path);
    }

    // Helpful hint for sequential scans (best-effort)
    (void)::madvise(mm->base, mm->size, MADV_SEQUENTIAL);

    // Insert (or reuse if someone beat us)
    {
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(bed_path);
        if (it != cache.end() && it->second) return it->second;
        cache[bed_path] = mm;
    }
    return mm;
}
#endif // __linux__

// ---------------- public prefetch hook (optional) ----------------
// NOTE: Ensure genotype.hpp declares this if you call it elsewhere.
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

    // Current block
    const size_t off0 = (size_t)3 + (size_t)blk_start * per_snp_bytes;
    const size_t len0 = (size_t)L * per_snp_bytes;
    madvise_willneed_range(mm->base, mm->size, off0, len0);

    // Ahead blocks
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

// ---------------- decode: selected rows only + integer-moment stats ----------------
// rows is ALWAYS sorted ascending.

static inline void decode_rows_codes_dense_sorted_into(const unsigned char* bytes,
                                                       int N_total,
                                                       const std::vector<int>& rows,
                                                       uint8_t* __restrict codeN, // size N (preallocated)
                                                       long long& nobs,
                                                       long long& sum,
                                                       long long& sumsq)
{
    const int nbytes = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    nobs = 0; sum = 0; sumsq = 0;

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

            if (bits != 1) { // not missing
                ++nobs;
                if (bits == 2) { sum += 1; sumsq += 1; }
                else if (bits == 3) { sum += 2; sumsq += 4; }
            }

            ++j;
            if (j >= N) return;
            next = rows[(size_t)j];
        }
    }
}

static inline void decode_rows_codes_sparse_sorted_into(const unsigned char* bytes,
                                                        const std::vector<int>& rows,
                                                        uint8_t* __restrict codeN, // size N (preallocated)
                                                        long long& nobs,
                                                        long long& sum,
                                                        long long& sumsq)
{
    nobs = 0; sum = 0; sumsq = 0;
    const int N = (int)rows.size();
    if (N == 0) return;

    for (int j = 0; j < N; ++j) {
        const int r = rows[(size_t)j];
        const unsigned char c = bytes[(size_t)r >> 2];
        const int sh = (r & 3) * 2;
        const uint8_t bits = (uint8_t)((c >> sh) & 0x3);
        codeN[j] = bits;

        if (bits != 1) {
            ++nobs;
            if (bits == 2) { sum += 1; sumsq += 1; }
            else if (bits == 3) { sum += 2; sumsq += 4; }
        }
    }
}


// ---------------- core implementation (templated, TU-local) ----------------
template <typename T>
static void read_block_standardized_impl(const std::string &bed_path,
                                         const std::string &fam_path,
                                         int blk_start, int blk_end,
                                         const std::vector<int> &rows,
                                         int ddof,
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

    // Only GROW, never shrink (lets callers reuse capacity cheaply if Geno is reused).
    if (Geno.size() < need) Geno.resize(need);

    // --- choose decode threads ---
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

        // avoid OMP overhead on small blocks
        if (work_elems < 2ULL*1000*1000) decode_threads = 1;
        else if (work_elems < 16ULL*1000*1000) decode_threads = std::min(decode_threads, 2);
        else if (work_elems < 64ULL*1000*1000) decode_threads = std::min(decode_threads, 4);

        if (decode_threads < 1) decode_threads = 1;
    }
    
    decode_threads = std::min(decode_threads, std::max(1, L));

    // --- sparse vs dense ---
    double sparse_thresh = 0.60;
    if (const char* s = std::getenv("SUMMIT_DECODE_SPARSE_THRESHOLD")) {
        const double v = std::atof(s);
        if (v > 0.0 && v < 1.0) sparse_thresh = v;
    }
    const double density = (N_total > 0) ? ((double)N / (double)N_total) : 1.0;
    const bool use_sparse = (density < sparse_thresh);

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

    auto worker = [&](int col, uint8_t* __restrict codes) {
        const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

        long long nobs = 0, sum = 0, sumsq = 0;
        if (use_sparse) decode_rows_codes_sparse_sorted_into(bytes, rows, codes, nobs, sum, sumsq);
        else            decode_rows_codes_dense_sorted_into (bytes, N_total, rows, codes, nobs, sum, sumsq);

        // Use double: plenty accurate here and faster than long double
        const double dnobs = (double)nobs;
        const long long denom_ll = nobs - (long long)ddof;

        const double mu = (nobs > 0) ? ((double)sum / dnobs) : 0.0;
        double M2 = 0.0;
        if (nobs > 0) {
            M2 = (double)sumsq - ((double)sum * (double)sum) / dnobs;
            if (M2 < 0.0) M2 = 0.0;
        }

        double inv_sd = 1.0;
        if (denom_ll > 0 && M2 > 0.0) {
            const double var = M2 / (double)denom_ll;
            if (var > 0.0) inv_sd = 1.0 / std::sqrt(var);
        }

        // Precompute standardized values for codes 0/2/3; code 1 (missing) => 0
        const double vstd[4] = {
            (0.0 - mu) * inv_sd,
            0.0,
            (1.0 - mu) * inv_sd,
            (2.0 - mu) * inv_sd
        };

        T* dst = Geno.data() + (size_t)col * (size_t)N;

#ifdef _OPENMP
        #pragma omp simd
#endif
        for (int i = 0; i < N; ++i) {
            dst[(size_t)i] = (T)vstd[codes[i]];
        }
    };

    if (decode_threads <= 1) {
        static thread_local std::vector<uint8_t> codes_local;
        if ((int)codes_local.size() < N) codes_local.resize((size_t)N);
        worker(0, codes_local.data()); // warm? not required
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
        for (int col = 0; col < L; ++col) {
            worker(col, codes_local.data());
        }
    }

#else
    // non-linux path unchanged (keep your existing streaming fallback if needed)
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

    for (int col = 0; col < L; ++col) {
        bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
        if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

        long long nobs = 0, sum = 0, sumsq = 0;
        decode_rows_codes_dense_sorted_into(line.data(), N_total, rows, codes_local.data(), nobs, sum, sumsq);

        const double dnobs = (double)nobs;
        const long long denom_ll = nobs - (long long)ddof;
        const double mu = (nobs > 0) ? ((double)sum / dnobs) : 0.0;

        double M2 = 0.0;
        if (nobs > 0) {
            M2 = (double)sumsq - ((double)sum * (double)sum) / dnobs;
            if (M2 < 0.0) M2 = 0.0;
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

        T* dst = Geno.data() + (size_t)col * (size_t)N;
        for (int i = 0; i < N; ++i) dst[(size_t)i] = (T)vstd[codes_local[(size_t)i]];
    }
#endif
}


// ---------------- exported concrete wrappers ----------------
void read_block_standardized_float(const std::string& bed_path,
                                  const std::string& fam_path,
                                  int blk_start, int blk_end,
                                  const std::vector<int>& rows,
                                  int ddof,
                                  std::vector<float>& Geno,
                                  int& N, int& L)
{
    read_block_standardized_impl<float>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
}

void read_block_standardized_double(const std::string& bed_path,
                                   const std::string& fam_path,
                                   int blk_start, int blk_end,
                                   const std::vector<int>& rows,
                                   int ddof,
                                   std::vector<double>& Geno,
                                   int& N, int& L)
{
    read_block_standardized_impl<double>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
}
