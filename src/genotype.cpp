#include "genotype.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>
#include <cstring>

#if defined(__linux__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

// Optional: let the compiler vectorize simple loops
#ifdef _OPENMP
  #include <omp.h>
#endif

// ---------------- small helpers (external linkage where needed) ----------------
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
    static std::unordered_map<std::string,int64_t> cache;
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

// ---------------- row mapping & per-SNP decode / stats ----------------
template <typename T>
struct RowMap {
    std::vector<int> idx_of; // size N_total, -1 if not selected, else [0..N)
    RowMap(int64_t N_total, const std::vector<int>& rows)
      : idx_of((size_t)N_total, -1)
    {
        for (int i = 0; i < (int)rows.size(); ++i) idx_of[(size_t)rows[i]] = i;
    }
};

template <typename T>
static inline void welford_update(T x, long double& mean, long double& M2, long long& n) {
    long double d  = (long double)x - mean;
    mean += d / (long double)(++n);
    long double d2 = (long double)x - mean;
    M2 += d * d2;
}

template <typename T>
static void decode_rows_and_stats(const unsigned char* bytes,
                                  int N_total,
                                  const RowMap<T>& rmap,
                                  std::vector<T>& bufN,     // size N
                                  long double& mean, long double& M2, long long& nobs)
{
    static const T lut[4] = { T(0), std::numeric_limits<T>::quiet_NaN(), T(1), T(2) };
    const int nbytes = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    int gidx = 0;
    for (int b = 0; b < nbytes; ++b) {
        unsigned char c = bytes[b];
        for (int t = 0; t < 4 && gidx < N_total; ++t, ++gidx) {
            int bits = (c >> (2*t)) & 0x3;
            int pos = rmap.idx_of[(size_t)gidx];
            if (pos >= 0) {
                T val = lut[bits];
                bufN[(size_t)pos] = val;
                if (!std::isnan(val)) welford_update(val, mean, M2, nobs);
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
    const int64_t N_total = count_lines_cached(fam_path);
    if (N_total <= 0) throw std::runtime_error("FAM has zero rows: " + fam_path);

    if (blk_end <= blk_start) {
        N = (int)rows.size(); L = 0; Geno.clear(); return;
    }

    L = blk_end - blk_start;
    N = (int)rows.size();
    Geno.assign((size_t)N * (size_t)L, T(0));

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;

    RowMap<T> rmap(N_total, rows);
    std::vector<T> tmpN((size_t)N);

#if defined(__linux__)
    // memory-map .bed for fast sequential SNP access
    int fd = ::open(bed_path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Failed to open bed: " + bed_path);

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 3 + (off_t)per_snp_bytes * (off_t)blk_end) {
        ::close(fd);
        throw std::runtime_error("BED file too small or stat() failed: " + bed_path);
    }
    unsigned char* base = (unsigned char*)mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed for: " + bed_path);
    }
    const unsigned char* snp0 = base + 3 + (size_t)blk_start * per_snp_bytes;

    for (int col = 0; col < L; ++col) {
        const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

        long double mean = 0.0L, M2 = 0.0L; long long nobs = 0;
        decode_rows_and_stats<T>(bytes, (int)N_total, rmap, tmpN, mean, M2, nobs);

        long long denom = nobs - ddof;
        T sd = (denom > 0 && M2 > 0.0L) ? (T)std::sqrt(M2 / (long double)denom) : T(1);
        if (sd == T(0)) sd = T(1);
        T mu = (nobs > 0) ? (T)(mean) : T(0);

        T *dst = Geno.data() + (size_t)col * (size_t)N;
        #pragma omp simd
        for (int i = 0; i < N; ++i) {
            T x = tmpN[(size_t)i];
            dst[i] = std::isnan(x) ? T(0) : (x - mu) / sd;
        }
    }

    munmap(base, (size_t)st.st_size);
    ::close(fd);

#else
    // Fallback streaming reader
    std::ifstream bed(bed_path, std::ios::binary);
    if (!bed) throw std::runtime_error("Failed to open bed: " + bed_path);

    unsigned char magic[3];
    bed.read(reinterpret_cast<char*>(magic), 3);

    const std::streamoff offset = 3 + (std::streamoff)blk_start * (std::streamoff)per_snp_bytes;
    bed.seekg(offset, std::ios::beg);

    std::vector<unsigned char> line(nbytes_per_snp);

    for (int col = 0; col < L; ++col) {
        bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
        if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(blk_start + col));

        long double mean = 0.0L, M2 = 0.0L; long long nobs = 0;
        decode_rows_and_stats<T>(line.data(), (int)N_total, rmap, tmpN, mean, M2, nobs);

        long long denom = nobs - ddof;
        T sd = (denom > 0 && M2 > 0.0L) ? (T)std::sqrt(M2 / (long double)denom) : T(1);
        if (sd == T(0)) sd = T(1);
        T mu = (nobs > 0) ? (T)(mean) : T(0);

        T *dst = Geno.data() + (size_t)col * (size_t)N;
        #pragma omp simd
        for (int i = 0; i < N; ++i) {
            T x = tmpN[(size_t)i];
            dst[i] = std::isnan(x) ? T(0) : (x - mu) / sd;
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
