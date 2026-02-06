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
    static std::unordered_map<std::string, std::weak_ptr<BedMapping>> cache;

    // Fast path: existing mapping
    {
        std::lock_guard<std::mutex> lk(m);
        if (auto it = cache.find(bed_path); it != cache.end()) {
            if (auto sp = it->second.lock()) return sp;
        }
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
        if (auto it = cache.find(bed_path); it != cache.end()) {
            if (auto sp = it->second.lock()) return sp;
        }
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
//
// Works best when `rows` is sorted ascending (true for your pipelines).
// If rows is not sorted, we fall back to a simple idx_of vector.
//
template <typename T>
static void decode_rows_and_counts_sorted(const unsigned char* bytes,
                                         int N_total,
                                         const std::vector<int>& rows, // selected global row indices
                                         std::vector<T>& bufN,         // size N=rows.size()
                                         long long& nobs,
                                         long long& sum,
                                         long long& sumsq)
{
    static const T lut_val[4] = {
        T(0),
        std::numeric_limits<T>::quiet_NaN(), // missing
        T(1),
        T(2)
    };

    const int nbytes = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    nobs = 0; sum = 0; sumsq = 0;

    const int N = (int)rows.size();
    if (N == 0) return;

    int gidx = 0;
    int j = 0;
    int next = rows[0];

    for (int b = 0; b < nbytes; ++b) {
        unsigned char c = bytes[b];
        for (int t = 0; t < 4 && gidx < N_total; ++t, ++gidx) {
            if (gidx != next) continue;

            const int bits = (c >> (2 * t)) & 0x3;
            const T val = lut_val[bits];
            bufN[(size_t)j] = val;

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

template <typename T>
static void decode_rows_and_counts_idxof(const unsigned char* bytes,
                                        int N_total,
                                        const std::vector<int>& idx_of, // size N_total, -1 if not selected
                                        std::vector<T>& bufN,           // size N selected
                                        long long& nobs,
                                        long long& sum,
                                        long long& sumsq)
{
    static const T lut_val[4] = {
        T(0),
        std::numeric_limits<T>::quiet_NaN(), // missing
        T(1),
        T(2)
    };

    const int nbytes = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    nobs = 0; sum = 0; sumsq = 0;

    int gidx = 0;
    for (int b = 0; b < nbytes; ++b) {
        unsigned char c = bytes[b];
        for (int t = 0; t < 4 && gidx < N_total; ++t, ++gidx) {
            const int pos = idx_of[(size_t)gidx];
            if (pos < 0) continue;

            const int bits = (c >> (2 * t)) & 0x3;
            const T val = lut_val[bits];
            bufN[(size_t)pos] = val;

            if (bits != 1) { // not missing
                ++nobs;
                if (bits == 2) { sum += 1; sumsq += 1; }
                else if (bits == 3) { sum += 2; sumsq += 4; }
            }
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
        Geno.clear();
        return;
    }

    L = blk_end - blk_start;
    N = (int)rows.size();
    if (N <= 0 || L <= 0) {
        Geno.clear();
        return;
    }

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;

    const size_t need = (size_t)N * (size_t)L;
    Geno.assign(need, T(0)); // ensure clean fill (safe if decode leaves some entries unwritten)

    // Determine decode threading (independent from outer OMP/BLAS)
    int decode_threads = 0;
    if (const char* s = std::getenv("SUMMIT_DECODE_THREADS")) decode_threads = std::atoi(s);

    int hw = 1;
#ifdef _OPENMP
    hw = omp_get_num_procs();
    if (hw <= 0) hw = 1;
#endif

    if (decode_threads <= 0) {
        // Conservative default: helps on big boxes without tmpN explosion
        decode_threads = std::min(16, hw);
        if (decode_threads < 1) decode_threads = 1;
    }

    // Cap by tmpN memory (default cap ~512 MiB total across decode threads)
    {
        const size_t cap_bytes = (size_t)512 * 1024 * 1024;
        const size_t per_thr = (size_t)N * sizeof(T);
        if (per_thr > 0) {
            int max_by_mem = (int)std::max<size_t>(1, cap_bytes / per_thr);
            if (decode_threads > max_by_mem) decode_threads = max_by_mem;
        }
        if (decode_threads < 1) decode_threads = 1;
    }

    // Check if rows are sorted (expected true)
    bool rows_sorted = true;
    for (int i = 1; i < N; ++i) {
        if (rows[(size_t)i] < rows[(size_t)(i - 1)]) { rows_sorted = false; break; }
    }

    // If not sorted: build idx_of (size N_total). (Expected rare.)
    std::vector<int> idx_of;
    if (!rows_sorted) {
        idx_of.assign((size_t)N_total, -1);
        for (int i = 0; i < N; ++i) {
            const int g = rows[(size_t)i];
            if (g >= 0 && g < N_total) idx_of[(size_t)g] = i;
        }
    }

#if defined(__linux__)
    auto mm = get_bed_mapping_cached(bed_path);

    // Ensure file is large enough for [0..blk_end)
    const size_t need_bytes = (size_t)3 + (size_t)blk_end * per_snp_bytes;
    if (!mm || !mm->base || need_bytes > mm->size) {
        throw std::runtime_error("BED file too small for requested block: " + bed_path);
    }

    // Optional prefetch (WILLNEED) for current and ahead
    int ahead = 1;
    if (const char* s = std::getenv("SUMMIT_PREFETCH_AHEAD_BLKS")) {
        int v = std::atoi(s);
        if (v >= 0) ahead = v;
    }
    {
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

#ifdef _OPENMP
    #pragma omp parallel num_threads(decode_threads)
#endif
    {
        std::vector<T> tmpN_local((size_t)N);

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int col = 0; col < L; ++col) {
            const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

            long long nobs = 0, sum = 0, sumsq = 0;
            if (rows_sorted) {
                decode_rows_and_counts_sorted<T>(bytes, N_total, rows, tmpN_local, nobs, sum, sumsq);
            } else {
                decode_rows_and_counts_idxof<T>(bytes, N_total, idx_of, tmpN_local, nobs, sum, sumsq);
            }

            // mean / sd using ddof
            const long long denom_ll = nobs - (long long)ddof;
            const long double mu = (nobs > 0) ? ((long double)sum / (long double)nobs) : 0.0L;

            long double M2 = 0.0L;
            if (nobs > 0) {
                // M2 = sumsq - sum^2 / nobs
                M2 = (long double)sumsq - ((long double)sum * (long double)sum) / (long double)nobs;
                if (M2 < 0.0L) M2 = 0.0L; // numerical guard
            }

            T sd = (denom_ll > 0 && M2 > 0.0L) ? (T)std::sqrt(M2 / (long double)denom_ll) : T(1);
            if (sd == T(0)) sd = T(1);
            const T inv_sd = T(1) / sd;
            const T mu_t   = (T)mu;

            // Write standardized column (col-major: Geno[col*N + i])
            T* dst = Geno.data() + (size_t)col * (size_t)N;
#ifdef _OPENMP
            #pragma omp simd
#endif
            for (int i = 0; i < N; ++i) {
                const T x = tmpN_local[(size_t)i];
                dst[(size_t)i] = std::isnan(x) ? T(0) : (x - mu_t) * inv_sd;
            }
        }
    }

#else
    // ---------------- non-Linux fallback: streaming I/O + header check ----------------
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
    std::vector<T> tmpN((size_t)N);

    for (int col = 0; col < L; ++col) {
        bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
        if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

        long long nobs = 0, sum = 0, sumsq = 0;
        if (rows_sorted) {
            decode_rows_and_counts_sorted<T>(line.data(), N_total, rows, tmpN, nobs, sum, sumsq);
        } else {
            decode_rows_and_counts_idxof<T>(line.data(), N_total, idx_of, tmpN, nobs, sum, sumsq);
        }

        const long long denom_ll = nobs - (long long)ddof;
        const long double mu = (nobs > 0) ? ((long double)sum / (long double)nobs) : 0.0L;

        long double M2 = 0.0L;
        if (nobs > 0) {
            M2 = (long double)sumsq - ((long double)sum * (long double)sum) / (long double)nobs;
            if (M2 < 0.0L) M2 = 0.0L;
        }

        T sd = (denom_ll > 0 && M2 > 0.0L) ? (T)std::sqrt(M2 / (long double)denom_ll) : T(1);
        if (sd == T(0)) sd = T(1);
        const T inv_sd = T(1) / sd;
        const T mu_t   = (T)mu;

        T* dst = Geno.data() + (size_t)col * (size_t)N;
        for (int i = 0; i < N; ++i) {
            const T x = tmpN[(size_t)i];
            dst[(size_t)i] = std::isnan(x) ? T(0) : (x - mu_t) * inv_sd;
        }
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

