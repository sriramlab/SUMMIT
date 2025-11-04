// src/gwldcore.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <immintrin.h>

#include <unordered_map>
#include <mutex>


#if defined(__linux__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <sched.h>
#endif


namespace py = pybind11;

// --- Interrupt handling (Ctrl-C) --------------------------------------------
static inline void check_for_interrupt() {
    py::gil_scoped_acquire gil;
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
}

// --- aligned new/delete ------------------------------------------------------
template <typename T>
struct AlignedBuffer {
    T* ptr = nullptr;
    size_t n = 0;
    AlignedBuffer() = default;
    AlignedBuffer(size_t count, size_t align = 64) { allocate(count, align); }
    void allocate(size_t count, size_t align = 64) {
        free();
        if (count == 0) return;
        void* p = nullptr;
        if (posix_memalign(&p, align, count * sizeof(T)) != 0) throw std::bad_alloc();
        ptr = reinterpret_cast<T*>(p);
        n = count;
    }
    void free() {
        if (ptr) { std::free(ptr); ptr = nullptr; n = 0; }
    }
    ~AlignedBuffer() { free(); }
};


// Return the list of CPUs this process is allowed to run on.
// Falls back to 1..(omp_get_max_threads) if affinity is unavailable.
static inline std::vector<int> active_cpus() {
    std::vector<int> cpus;
#ifdef __linux__
    long nconf = sysconf(_SC_NPROCESSORS_CONF);
    if (nconf < 1) nconf = 1;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        for (int i = 0; i < nconf; ++i) {
            if (CPU_ISSET(i, &mask)) cpus.push_back(i);
        }
        if (!cpus.empty()) return cpus;
    }
#endif
#ifdef _OPENMP
    int n = std::max(1, omp_get_max_threads());
#else
    int n = 1;
#endif
    for (int i = 0; i < n; ++i) cpus.push_back(i);
    return cpus;
}

// Read L3 size for a given CPU’s index3 (heuristic, may differ per CCD on AMD).
static inline size_t read_l3_bytes_for_cpu(int cpu) {
#ifdef __linux__
    std::ostringstream p;
    p << "/sys/devices/system/cpu/cpu" << cpu << "/cache/index3/size";
    std::ifstream f(p.str());
    if (f) {
        std::string s; f >> s;
        if (!s.empty()) {
            char unit = s.back();
            size_t val = std::stoul(s);
            if (unit == 'K' || unit == 'k') return val * 1024ULL;
            if (unit == 'M' || unit == 'm') return val * 1024ULL * 1024ULL;
            return val;
        }
    }
#endif
    return 0;
}

// Count sockets by unique physical_package_id among active CPUs.
static inline int detect_num_sockets() {
#ifdef __linux__
    std::set<int> sockets;
    for (int cpu : active_cpus()) {
        std::ostringstream path;
        path << "/sys/devices/system/cpu/cpu" << cpu << "/topology/physical_package_id";
        std::ifstream f(path.str());
        if (f) {
            int sid = -1; f >> sid;
            if (sid >= 0) sockets.insert(sid);
        }
    }
    if (!sockets.empty()) return (int)sockets.size();
#endif
    return 0; // unknown
}

struct TilePlan { int VPANEL; int CTILE; };

static inline size_t read_l3_per_socket_bytes() {
    const char* path = "/sys/devices/system/cpu/cpu0/cache/index3/size";
    std::ifstream f(path);
    if (!f) return 0;
    std::string s; f >> s;
    if (s.empty()) return 0;
    char unit = s.back();
    size_t val = std::stoul(s);
    if (unit == 'K' || unit == 'k') return val * 1024ULL;
    if (unit == 'M' || unit == 'm') return val * 1024ULL * 1024ULL;
    return val; // bytes
}

static inline int getenv_int(const char* k, int defv) {
    const char* v = std::getenv(k);
    if (!v) return defv;
    try { return std::max(1, std::stoi(v)); } catch (...) { return defv; }
}
static inline double getenv_double(const char* k, double defv) {
    const char* v = std::getenv(k);
    if (!v) return defv;
    try { return std::max(0.0, std::stod(v)); } catch (...) { return defv; }
}

template <typename T>
static inline TilePlan choose_tiles_auto(int N, int L, int B, int nvecs) {
    // ----------------- Env overrides -----------------
    const int    env_ctile     = getenv_int("SUMMIT_CTILE",    -1);      // columns
    const int    env_ctile_mb  = getenv_int("SUMMIT_CTILE_MB", -1);      // MiB
    const double env_l3_pct    = getenv_double("SUMMIT_CTILE_L3PCT", 0.60);

    const size_t bytes = sizeof(T);
    auto round64 = [](int x){ return ((x + 63) / 64) * 64; };
    auto clamp   = [](int x, int lo, int hi){ return std::max(lo, std::min(hi, x)); };

    auto make_plan = [&](int CTILE) -> TilePlan {
        CTILE = round64(std::max(64, CTILE));
        // VPANEL ≈ CTILE/B, rounded to 64; at least 64, at most nvecs
        int vguess = (B > 0) ? (CTILE / B) : CTILE;
        int VPANEL = round64(std::max(64, std::min(nvecs, vguess)));
        if (VPANEL > nvecs) VPANEL = nvecs;
        if (VPANEL < 64)    VPANEL = std::min(64, std::max(1, nvecs)); // nvecs could be <64
        return {VPANEL, CTILE};
    };

    // 1) Explicit CTILE override
    if (env_ctile > 0) {
        return make_plan(env_ctile);
    }

    // 2) Memory-budget override (MiB)
    if (env_ctile_mb > 0) {
        const size_t target_bytes = (size_t)env_ctile_mb << 20; // MiB→bytes
        const size_t denom = (size_t)(N + L) * bytes;           // (rhs + lhs) footprint per column
        int CTILE = denom ? (int)(target_bytes / denom) : 2048;
        // Conservative clamps by dtype
        if (bytes == 4) CTILE = clamp(round64(CTILE), 4096, 16384);
        else            CTILE = clamp(round64(CTILE), 2048,  8192);
        return make_plan(CTILE);
    }

    // ----------------- Auto (L3-aware) -----------------
    // L3 per active CPU (heuristic: read from the first CPU in our affinity)
    size_t l3_bytes = 0;
    {
        auto cpus = active_cpus();
        int probe = cpus.empty() ? 0 : cpus.front();
        l3_bytes = read_l3_bytes_for_cpu(probe); // may be 0 if not available
    }

    // Sockets: env override → topology detection → default=2
    int sockets = getenv_int("SUMMIT_SOCKETS", 0);
    if (sockets <= 0) {
        sockets = detect_num_sockets();
        if (sockets <= 0) sockets = 2;
    }

    // Parallel width: prefer affinity size over omp_get_max_threads()
    int threads_total = 1;
#ifdef _OPENMP
    threads_total = std::max(1, omp_get_max_threads());
#endif
    {
        int aff = (int)active_cpus().size();
        if (aff > 0) threads_total = aff;
    }
    const double t_per_socket = std::max(1.0, (double)threads_total / (double)sockets);

    // Target L3 per thread share
    size_t target_bytes = 0;
    if (l3_bytes > 0) {
        target_bytes = (size_t)(env_l3_pct * (double)l3_bytes / t_per_socket);
    }

    // Convert target bytes → CTILE columns
    int CTILE;
    if (target_bytes > 0) {
        const size_t denom = (size_t)(N + L) * bytes;
        CTILE = denom ? (int)(target_bytes / denom) : 2048;
    } else {
        // Fallback defaults when topology info is missing
        CTILE = (bytes == 4) ? 8192 : 4096;
    }

    // Clamp & round (dtype-aware)
    if (bytes == 4) CTILE = clamp(round64(CTILE), 4096, 16384);
    else            CTILE = clamp(round64(CTILE), 2048,  8192);

    return make_plan(CTILE);
}

// --- timers ------------------------------------------------------------------
struct P1Timers {
    double t_packZ_ms = 0.0;   // pack Z→Bcol (per-bin)
    double t_packA_ms = 0.0;   // pack Geno→A_tile (per N-tile)
    double t_gemm_ms  = 0.0;   // GEMM time
    double t_scatt_ms = 0.0;   // scatter-add C_tile → Xz

    void dump(int blk_start, int blk_end, int B, int vcount) const {
        std::fprintf(stderr,
          "[phase1] block [%d:%d) B=%d V=%d  packZ=%.2f ms  packA=%.2f ms  gemm=%.2f ms  scatter=%.2f ms\n",
          blk_start, blk_end, B, vcount, t_packZ_ms, t_packA_ms, t_gemm_ms, t_scatt_ms);
    }
};
struct BlockTimers {
    double t_pack_ms = 0.0;
    double t_gemm_ms = 0.0;
    double t_reduce_ms = 0.0;
    void add_pack(double ms){ t_pack_ms += ms; }
    void add_gemm(double ms){ t_gemm_ms += ms; }
    void add_reduce(double ms){ t_reduce_ms += ms; }
    void dump(int blk_start, int blk_end, int B, int nvecs) const {
        std::fprintf(stderr,
            "[phase2] block [%d:%d) B=%d V=%d  pack=%.2f ms  gemm=%.2f ms  reduce=%.2f ms\n",
            blk_start, blk_end, B, nvecs, t_pack_ms, t_gemm_ms, t_reduce_ms);
    }
};

// ------------------------------- Small helpers -------------------------------

static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

static inline int64_t count_lines(const std::string &path) {
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

static int64_t count_lines_cached(const std::string &path) {
    static std::mutex m;
    static std::unordered_map<std::string,int64_t> cache;
    {
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(path);
        if (it != cache.end()) return it->second;
    }
    int64_t n = count_lines(path);
    {
        std::lock_guard<std::mutex> lk(m);
        cache[path] = n;
    }
    return n;
}

static inline uint64_t mix64(uint64_t x) {
    // splitmix64
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static inline uint64_t make_seed(uint64_t root, int block, int v0) {
    uint64_t s = 0x1234abcdULL;
    s ^= mix64(root);
    s ^= mix64(static_cast<uint64_t>(block) + 0x9e37ULL);
    s ^= mix64(static_cast<uint64_t>(v0)    + 0x85ebULL);
    return s;
}

// Decode one SNP’s genotypes (2 bits / individual). PLINK .bed SNP-major.
// Mapping: 00->0, 01->missing (NaN), 10->1, 11->2
template <typename T>
static void decode_bed_snp_to_vector(const unsigned char *bytes,
                                     int N_total,
                                     std::vector<T> &out) {
    out.assign(N_total, std::numeric_limits<T>::quiet_NaN());
    const int nbytes = static_cast<int>(ceil_div((std::size_t)N_total, (std::size_t)4));
    int idx = 0;
    for (int b = 0; b < nbytes; ++b) {
        unsigned char c = bytes[b];
        for (int t = 0; t < 4 && idx < N_total; ++t, ++idx) {
            int bits = (c >> (2 * t)) & 0x3;
            T val;
            if (bits == 0b00) val = T(0);
            else if (bits == 0b10) val = T(1);
            else if (bits == 0b11) val = T(2);
            else { // 0b01 missing
                val = std::numeric_limits<T>::quiet_NaN();
            }
            out[idx] = val;
        }
    }
}

// Compute nanmean & nanstd (ddof) over a subset of rows.
// Returns (mean, std) where std=1 if degenerate; NaNs ignored.
template <typename T>
static std::pair<T,T> nan_mean_std(const std::vector<T> &col_full,
                                   const std::vector<int> &rows,
                                   int ddof) {
    long long ct = 0;
    long double sum = 0.0L;
    for (int r : rows) {
        T x = col_full[r];
        if (!std::isnan(x)) { sum += x; ++ct; }
    }
    T mean = (ct > 0) ? static_cast<T>(sum / (long double)ct) : T(0);

    long double ss = 0.0L;
    for (int r : rows) {
        T x = col_full[r];
        if (!std::isnan(x)) {
            long double d = (long double)x - (long double)mean;
            ss += d * d;
        }
    }
    long long denom = ct - ddof;
    T s = (denom > 0) ? static_cast<T>(std::sqrt(ss / (long double)denom)) : T(1);
    if (s == T(0)) s = T(1);
    return {mean, s};
}

// Read a genotype block [s:e) and standardize per SNP.
// Output: column-major Geno (N x L), element at (i,j) is Geno[i + j*N].
template <typename T>
struct RowMap {
    std::vector<int> idx_of; // size N_total, -1 if not selected, else [0..N)
    RowMap(int64_t N_total, const std::vector<int>& rows) : idx_of((size_t)N_total, -1) {
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

// Decode SNP-major 2-bit genotypes but only materialize selected rows (size N)
// and compute mean/var via Welford (NaN-skipping).
template <typename T>
static void decode_rows_and_stats(const unsigned char* bytes,
                                  int N_total,
                                  const RowMap<T>& rmap,
                                  std::vector<T>& bufN,          // size N (output raw)
                                  long double& mean, long double& M2, long long& nobs)
{
    static const T lut[4] = { T(0), std::numeric_limits<T>::quiet_NaN(), T(1), T(2) };
    const int nbytes = static_cast<int>(ceil_div((std::size_t)N_total, (std::size_t)4));
    int gidx = 0; // global person index

    // We will fill every bufN position exactly once
    // (since each selected row appears exactly once among [0..N_total))
    for (int b = 0; b < nbytes; ++b) {
        unsigned char c = bytes[b];
        // Unroll by 4 genotypes contained in this byte
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

// New: decode only selected rows, compute stats once, standardize, write to Geno (N x L)
template <typename T>
static void read_block_standardized(const std::string &bed_path,
                                    const std::string &fam_path,
                                    int blk_start, int blk_end,
                                    const std::vector<int> &rows, // selected individuals
                                    int ddof,
                                    std::vector<T> &Geno, // (N x L), col-major
                                    int &N, int &L)
{
    const int64_t N_total = count_lines_cached(fam_path);
    if (N_total <= 0) throw std::runtime_error("FAM has zero rows: " + fam_path);
    if (blk_end <= blk_start) { N = (int)rows.size(); L = 0; Geno.clear(); return; }

    L = blk_end - blk_start;
    N = (int)rows.size();
    Geno.assign((size_t)N * (size_t)L, T(0)); // final standardized output

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const size_t per_snp_bytes = (size_t)nbytes_per_snp;

    RowMap<T> rmap(N_total, rows);
    std::vector<T> tmpN((size_t)N); // raw genotypes for selected rows

#if defined(__linux__)
    // mmap the bed for fast sequential SNP access
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
    // SNP-major payload starts after 3-byte header
    const unsigned char* snp0 = base + 3 + (size_t)blk_start * per_snp_bytes;

    for (int col = 0; col < L; ++col) {
        const unsigned char* bytes = snp0 + (size_t)col * per_snp_bytes;

        long double mean = 0.0L, M2 = 0.0L; long long nobs = 0;
        decode_rows_and_stats<T>(bytes, (int)N_total, rmap, tmpN, mean, M2, nobs);

        // finalize stats with ddof
        long long denom = nobs - ddof;
        T sd = (denom > 0 && M2 > 0.0L) ? (T)std::sqrt(M2 / (long double)denom) : T(1);
        if (sd == T(0)) sd = T(1);
        T mu = (nobs > 0) ? (T)(mean) : T(0);

        // standardize selected rows into Geno
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
    // Fallback: stream reads, but still avoid N_total-sized intermediates
    std::ifstream bed(bed_path, std::ios::binary);
    if (!bed) throw std::runtime_error("Failed to open bed: " + bed_path);

    // header
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

// BLAS helpers (column-major)
template <typename T>
inline void gemm_col_major_nn(int m, int n, int k,
                              const T* A, int lda,
                              const T* B, int ldb,
                              T* C, int ldc,
                              T alpha = T(1), T beta = T(1)) {
    if constexpr (std::is_same_v<T,double>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}
template <typename T>
inline void gemm_col_major_tn(int m, int n, int k,
                              const T* A, int lda,  // A used with Transpose
                              const T* B, int ldb,
                              T* C, int ldc,
                              T alpha = T(1), T beta = T(0)) {
    if constexpr (std::is_same_v<T,double>) {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// Parse optional row_sel (indices of individuals to keep)
static std::vector<int> parse_row_sel(py::object row_sel_obj, int64_t N_total) {
    if (row_sel_obj.is_none()) {
        std::vector<int> rows((size_t)N_total);
        for (int64_t i = 0; i < N_total; ++i) rows[(size_t)i] = (int)i;
        return rows;
    }
    py::array idx = row_sel_obj.cast<py::array>();
    py::buffer_info bi = idx.request();
    std::vector<int> rows((size_t)bi.shape[0]);
    // accept int32/int64
    if (bi.format == py::format_descriptor<int32_t>::format()) {
        auto p = static_cast<const int32_t*>(bi.ptr);
        for (ssize_t i = 0; i < bi.shape[0]; ++i) rows[(size_t)i] = (int)p[i];
    } else {
        auto p = static_cast<const int64_t*>(bi.ptr);
        for (ssize_t i = 0; i < bi.shape[0]; ++i) rows[(size_t)i] = (int)p[i];
    }
    return rows;
}

inline void cblas_taxpy(int n, float  a, const float*  x, int incx, float*  y, int incy){ cblas_saxpy(n,a,x,incx,y,incy); }
inline void cblas_taxpy(int n, double a, const double* x, int incx, double* y, int incy){ cblas_daxpy(n,a,x,incx,y,incy); }

// -------------------------- Phase 1: compute_Xz (K-major) --------------------------
template <typename T>
void phase1_compute_Xz_bed_chunk_impl(const std::string &bed_prefix,
                                      const std::string &fam_path,
                                      int blk_start, int blk_end,
                                      py::object row_sel_obj,
                                      int ddof,
                                      py::array_t<T, py::array::c_style | py::array::forcecast> annot_blk, // (L x B)
                                      py::array_t<T, py::array::c_style | py::array::forcecast> inv_right,  // (L,)
                                      int v_start,            // global V offset
                                      int v_count,            // V cols in THIS tile
                                      int /*kmax_hint*/,      // unused with CSR
                                      const std::string &rand_dist,
                                      py::object seed_obj,    // None or int
                                      py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d_chunk, // (N x (B*v_count)), Fortran, **K-major**
                                      bool /*project_right*/ = false,
                                      py::object /*C_opt*/ = py::none(),
                                      py::object /*R_opt*/ = py::none())
{
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    // ---- rows & block genotype ----
    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    std::vector<T> Geno;                       // (N x L), column-major
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // ---- annot & inv ----
    auto Ainfo = annot_blk.request();
    auto Iinfo = inv_right.request();
    if (Ainfo.ndim != 2 || Iinfo.ndim != 1) throw std::runtime_error("annot/inv shapes");
    const int B = (int)Ainfo.shape[1];
    if ((int)Ainfo.shape[0] != L || (int)Iinfo.shape[0] != L) throw std::runtime_error("L mismatch");
    const T* ann = static_cast<const T*>(Ainfo.ptr);
    const T* inv = static_cast<const T*>(Iinfo.ptr);

    // ---- destination: K-major ----
    auto Xinfo = Xz2d_chunk.request();
    if (Xinfo.ndim != 2 || (int)Xinfo.shape[0] != N || (int)Xinfo.shape[1] != B * v_count)
        throw std::runtime_error("Xz2d_chunk shape must be (N, B*v_count)");
    T*  Xptr = static_cast<T*>(Xinfo.ptr);
    const int ldc = N;

    // ---- RNG per block (strict independence) ----
    const bool have_root = !seed_obj.is_none();
    const uint64_t root_seed = have_root ? seed_obj.cast<uint64_t>() : std::random_device{}();
    std::mt19937_64 rng(make_seed(root_seed, /*block=*/blk_start, /*v0=*/v_start));
    std::normal_distribution<T> gN(0, (T)1);
    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_spherical  = (rand_dist == "spherical");

    // ---- Z panel: (L x v_count), column-major ----
    std::vector<T> Z((size_t)L * (size_t)v_count, T(0));
    for (int c = 0; c < v_count; ++c) {
        long double ss = 0.0L;
        for (int r = 0; r < L; ++r) {
            T z = is_rademacher ? ((rng() & 1) ? T(+1) : T(-1)) : gN(rng);
            Z[(size_t)r + (size_t)c * (size_t)L] = z;
            if (is_spherical) ss += (long double)z * (long double)z;
        }
        if (is_spherical) {
            T scale = ss > 0.0L ? (T)std::sqrt((long double)L / ss) : T(1);
            for (int r = 0; r < L; ++r) Z[(size_t)r + (size_t)c * (size_t)L] *= scale;
        }
    }

    // ---- CSR compression of annotation (by columns=bins) ----
    // colptr[B+1], rowind[nnz], scale[nnz] with scale = inv[i] * sqrt(a_ik)
    std::vector<int> colptr(B + 1, 0);
    {
        // Count nnz in each bin
        for (int k = 0; k < B; ++k) {
            int cnt = 0;
            const T* colk = ann + (size_t)k; // column k viewed in row-major indexing i*B + k
            for (int i = 0; i < L; ++i) {
                T a = colk[(size_t)i * (size_t)B];
                if (a != T(0)) ++cnt;
            }
            colptr[(size_t)k + 1] = cnt;
        }
        // prefix sum
        for (int k = 0; k < B; ++k) colptr[(size_t)k + 1] += colptr[(size_t)k];
    }
    const int nnz = colptr[(size_t)B];
    AlignedBuffer<int> rowind_buf((size_t)nnz, 64);
    AlignedBuffer<T>   scale_buf((size_t)nnz, 64);
    {
        std::vector<int> fill(B, 0);
        for (int k = 0; k < B; ++k) {
            int base = colptr[(size_t)k];
            int& f   = fill[(size_t)k];
            const T* colk = ann + (size_t)k;
            for (int i = 0; i < L; ++i) {
                T a = colk[(size_t)i * (size_t)B];
                if (a != T(0)) {
                    const int pos = base + f++;
                    rowind_buf.ptr[(size_t)pos] = i;
                    scale_buf.ptr [(size_t)pos] = inv[i] * std::sqrt(a);
                }
            }
        }
    }

    // ---- First-touch Xz2d (only once at start-of-tile) ----
    if (blk_start == 0) {
        const size_t Q = (size_t)B * (size_t)v_count;
        const size_t elems_per_page = (size_t)(4096 / sizeof(T) ? 4096 / sizeof(T) : 512);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t g = 0; g < (ptrdiff_t)Q; ++g) {
            T* col = Xptr + (size_t)g * (size_t)ldc;
            for (size_t r = 0; r < (size_t)N; r += elems_per_page) {
                col[r] += T(0); // write to establish page ownership
            }
        }
    }

    // ---- constants / timers ----
    const int NTILE = std::is_same_v<T,double> ? 256 : 512;
    P1Timers t1;

    // ---- Outer loop over bins; packZ once per bin, then OMP over N-tiles ----
    for (int k = 0; k < B; ++k) {
        check_for_interrupt();

        const int k0 = colptr[(size_t)k];
        const int k1 = colptr[(size_t)k + 1];
        const int K  = k1 - k0;
        if (K == 0) continue;

        // Pack Bcol (K x v_count) once per bin; gather rows from Z
        AlignedBuffer<T> Bcol((size_t)K * (size_t)v_count, 64);
        auto z0 = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < v_count; ++c) {
            const T* zc = Z.data() + (size_t)c * (size_t)L;
            T*       dst = Bcol.ptr  + (size_t)c * (size_t)K;
            for (int r = 0; r < K; ++r) {
                const int snp = rowind_buf.ptr[(size_t)k0 + (size_t)r];
                dst[(size_t)r] = zc[(size_t)snp];
            }
        }
        auto z1 = std::chrono::high_resolution_clock::now();
        t1.t_packZ_ms += std::chrono::duration<double,std::milli>(z1 - z0).count();

        // OMP over N-tiles; packA → GEMM → K-major scatter via AXPY
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            AlignedBuffer<T> A_tile((size_t)NTILE * (size_t)K, 64);           // (Nt x K)
            AlignedBuffer<T> C_tile((size_t)NTILE * (size_t)v_count, 64);     // (Nt x v_count)

            double packA_ms = 0.0, gemm_ms = 0.0, scatt_ms = 0.0;

            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int n0 = 0; n0 < N; n0 += NTILE) {
                const int Nt = std::min(N - n0, NTILE);

                // packA: A_tile(:, c) = Geno(:, snp_c)[n0:n0+Nt) * scale_c
                auto a0 = std::chrono::high_resolution_clock::now();
                for (int c = 0; c < K; ++c) {
                    const int snp = rowind_buf.ptr[(size_t)k0 + (size_t)c];
                    const T   s   = scale_buf.ptr [(size_t)k0 + (size_t)c];
                    const T* src  = Geno.data() + (size_t)snp * (size_t)N + (size_t)n0;
                    T*       dst  = A_tile.ptr  + (size_t)c   * (size_t)Nt;
                    #pragma omp simd
                    for (int r = 0; r < Nt; ++r) dst[r] = src[r] * s;
                }
                auto a1 = std::chrono::high_resolution_clock::now();
                packA_ms += std::chrono::duration<double,std::milli>(a1 - a0).count();

                // GEMM: C_tile(Nt x v_count) = A_tile(Nt x K) * Bcol(K x v_count)
                auto g0 = std::chrono::high_resolution_clock::now();
                gemm_col_major_nn<T>(/*m=*/Nt, /*n=*/v_count, /*k=*/K,
                                     /*A=*/A_tile.ptr, /*lda=*/Nt,
                                     /*B=*/Bcol.ptr,   /*ldb=*/K,
                                     /*C=*/C_tile.ptr, /*ldc=*/Nt,
                                     /*alpha=*/T(1), /*beta=*/T(0));
                auto g1 = std::chrono::high_resolution_clock::now();
                gemm_ms += std::chrono::duration<double,std::milli>(g1 - g0).count();

                // K-major scatter: columns [k*v_count .. k*v_count+v_count-1]
                auto s0 = std::chrono::high_resolution_clock::now();
                const size_t base_col = (size_t)k * (size_t)v_count;
                for (int c = 0; c < v_count; ++c) {
                    const T* __restrict src_col = C_tile.ptr + (size_t)c * (size_t)Nt;
                    T* __restrict dst_col = Xptr + ((base_col + (size_t)c) * (size_t)ldc) + (size_t)n0;
                    // BLAS AXPY length Nt
                    cblas_taxpy(Nt, T(1), src_col, 1, dst_col, 1);
                }
                auto s1 = std::chrono::high_resolution_clock::now();
                scatt_ms += std::chrono::duration<double,std::milli>(s1 - s0).count();
            } // n0

            #ifdef _OPENMP
            #pragma omp atomic
            t1.t_packA_ms += packA_ms;
            #pragma omp atomic
            t1.t_gemm_ms  += gemm_ms;
            #pragma omp atomic
            t1.t_scatt_ms += scatt_ms;
            #else
            t1.t_packA_ms += packA_ms;
            t1.t_gemm_ms  += gemm_ms;
            t1.t_scatt_ms += scatt_ms;
            #endif
        } // parallel
    } // bins

    t1.dump(blk_start, blk_end, /*B=*/B, /*vcount=*/v_count);
}

// -------------------------- Phase 2: XtXz (K-major) --------------------------
template <typename T>
void phase2_compute_XtXz_bed_impl(const std::string &bed_prefix,
                                  const std::string &fam_path,
                                  int blk_start, int blk_end,
                                  py::object row_sel_obj,
                                  int ddof,
                                  py::array_t<T, py::array::c_style | py::array::forcecast> inv_left, // (L,)
                                  int nvecs,
                                  int /*vchunk*/,  // unused
                                  py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d,      // (N x (B*V)), **K-major**
                                  py::array_t<T, py::array::c_style | py::array::forcecast> meansq,    // (M x B)
                                  py::object C_opt, py::object R_opt,
                                  int N_denom)
{
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);

    // Read & standardize geno block -> Geno (N x L), col-major
    int N = 0, L = 0;
    std::vector<T> Geno;
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // Optional projection: Y = (I - C R) * Geno, done **in-place** in Geno
    if (!C_opt.is_none() && !R_opt.is_none()) {
        py::array_t<T, py::array::f_style | py::array::forcecast> C = C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        py::array_t<T, py::array::f_style | py::array::forcecast> R = R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Ci = C.request(); auto Ri = R.request();
        const int p = (int)Ci.shape[1];
        if ((int)Ci.shape[0] != N || (int)Ri.shape[0] != p || (int)Ri.shape[1] != N)
            throw std::runtime_error("C/R shape mismatch");

        const T* Cptr = static_cast<const T*>(Ci.ptr);
        const T* Rptr = static_cast<const T*>(Ri.ptr);

        // tmpG = R * Geno (p x L)
        AlignedBuffer<T> tmpG((size_t)p * (size_t)L, 64);
        gemm_col_major_nn<T>(/*m=*/p, /*n=*/L, /*k=*/N,
                             /*A=*/Rptr, /*lda=*/p,
                             /*B=*/Geno.data(), /*ldb=*/N,
                             /*C=*/tmpG.ptr, /*ldc=*/p,
                             /*alpha=*/T(1), /*beta=*/T(0));
        // Geno = Geno - C * tmpG  (in-place)
        gemm_col_major_nn<T>(/*m=*/N, /*n=*/L, /*k=*/p,
                             /*A=*/Cptr, /*lda=*/N,
                             /*B=*/tmpG.ptr, /*ldb=*/p,
                             /*C=*/Geno.data(), /*ldc=*/N,
                             /*alpha=*/T(-1), /*beta=*/T(1));
    }

    // inv_left (L,)
    auto Ii = inv_left.request();
    if (Ii.ndim != 1 || (int)Ii.shape[0] != L) throw std::runtime_error("inv_left shape mismatch");
    const T* inv = static_cast<const T*>(Ii.ptr);

    // Xz: (N x (B*V)), **K-major** (column g = k*V + v)
    auto Xi = Xz2d.request();
    if (Xi.ndim != 2 || (int)Xi.shape[0] != N) throw std::runtime_error("Xz2d shape mismatch");
    T* Xptr = static_cast<T*>(Xi.ptr);
    const int BV = (int)Xi.shape[1];
    if (nvecs <= 0 || (BV % nvecs) != 0) throw std::runtime_error("Xz2d col count must be multiple of nvecs");
    const int B = BV / nvecs;

    // meansq: (M x B)
    auto Mi = meansq.request();
    if (Mi.ndim != 2 || (int)Mi.shape[1] != B) throw std::runtime_error("meansq shape mismatch");
    T* Mptr = static_cast<T*>(Mi.ptr);
    const int M = (int)Mi.shape[0];
    if (blk_end > M) throw std::runtime_error("meansq rows smaller than SNP count");

    // Scale Geno columns by inv_left / (N_denom - 1)  (in-place)
    T denom = T(N_denom) - T(1);
    if (denom <= T(0)) denom = T(1);
    for (int i = 0; i < L; ++i) {
        const T s = inv[i] / denom;
        T* col = Geno.data() + (size_t)i * (size_t)N;  // col-major
        #pragma omp simd
        for (int r = 0; r < N; ++r) col[r] *= s;
    }

    // Tile planning over Q = B*V (columns contiguous in K-major)
    const int Q = BV;
    int QPANEL = getenv_int("SUMMIT_P2_QP", 12288);
    if (QPANEL <= 0) QPANEL = 8192;
    if (QPANEL > Q) QPANEL = Q;
    QPANEL = ((QPANEL + 63) / 64) * 64;
    if (QPANEL > Q) QPANEL = Q;
    if (QPANEL < 64) QPANEL = std::min(Q, 64);

    // Workspace for one Q panel: (L x QPANEL), col-major
    AlignedBuffer<T> Work_panel((size_t)L * (size_t)QPANEL, 64);

    const T invV = T(1) / T(nvecs);

    BlockTimers t;
    py::gil_scoped_release nogil;

    for (int q0 = 0; q0 < Q; q0 += QPANEL) {
        check_for_interrupt();
        const int q = std::min(QPANEL, Q - q0);

        // GEMM: Work_panel(L x q) = (Geno^T)(L x N) * Xz_panel(N x q)
        const T* rhs = Xptr + (size_t)q0 * (size_t)N; // K-major: advance q0 columns with ld N
        auto t2 = std::chrono::high_resolution_clock::now();
        gemm_col_major_tn<T>(/*m=*/L, /*n=*/q, /*k=*/N,
                             /*A=*/Geno.data(), /*lda=*/N,  // A^T is LxN
                             /*B=*/rhs,        /*ldb=*/N,  // N x q
                             /*C=*/Work_panel.ptr, /*ldc=*/L,
                             /*alpha=*/T(1), /*beta=*/T(0));
        auto t3 = std::chrono::high_resolution_clock::now();
        t.add_gemm(std::chrono::duration<double,std::milli>(t3 - t2).count());

        // Reduce: meansq[blk_start:blk_end, k] += (wcol^2)/V
        auto t4 = std::chrono::high_resolution_clock::now();
        for (int tcol = 0; tcol < q; ++tcol) {
            const int g = q0 + tcol;            // global column within [0, B*V)
            const int k = g / nvecs;           // **K-major** bin index
            const T* __restrict wcol = Work_panel.ptr + (size_t)tcol * (size_t)L;

            // base pointer for this bin's column in global meansq
            T* __restrict out = Mptr + ((size_t)blk_start * (size_t)B + (size_t)k);

            #pragma omp simd
            for (int i = 0; i < L; ++i) {
                const T z2 = wcol[i] * wcol[i] * invV;
                out[(size_t)i * (size_t)B] += z2; // stride by B across rows
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        t.add_reduce(std::chrono::duration<double,std::milli>(t5 - t4).count());
    }

    t.dump(blk_start, blk_end, B, nvecs);
}



// ------------------------------- PyBind module -------------------------------

PYBIND11_MODULE(gwldcore, m) {
    m.doc() = "C++ core for SUMMIT GW LD score (bed parser + BLAS-safe GEMMs)";

    // Phase 1 (chunked) float32
    m.def("phase1_compute_Xz_bed_chunk",
        &phase1_compute_Xz_bed_chunk_impl<float>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("blk_start"), py::arg("blk_end"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("annot_blk"),
        py::arg("inv_right"),
        py::arg("v_start"),
        py::arg("v_count"),
        py::arg("kmax_hint"),                     // NEW
        py::arg("rand_dist") = "rademacher",
        py::arg("seed") = py::none(),
        py::arg("Xz2d_chunk"),
        py::arg("project_right") = false,
        py::arg("C") = py::none(),
        py::arg("R") = py::none());

    // Phase 1 (chunked) float64
    m.def("phase1_compute_Xz_bed_chunk",
        &phase1_compute_Xz_bed_chunk_impl<double>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("blk_start"), py::arg("blk_end"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("annot_blk"),
        py::arg("inv_right"),
        py::arg("v_start"),
        py::arg("v_count"),
        py::arg("kmax_hint"),                     // NEW
        py::arg("rand_dist") = "rademacher",
        py::arg("seed") = py::none(),
        py::arg("Xz2d_chunk"),
        py::arg("project_right") = false,
        py::arg("C") = py::none(),
        py::arg("R") = py::none());


    // Phase 2 float32
    m.def("phase2_compute_XtXz_bed",
          &phase2_compute_XtXz_bed_impl<float>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("blk_start"), py::arg("blk_end"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("inv_left"),
          py::arg("nvecs"),
          py::arg("vchunk"),
          py::arg("Xz2d"),
          py::arg("meansq"),
          py::arg("C") = py::none(),
          py::arg("R") = py::none(),
          py::arg("N_denom") = 0);

    // Phase 2 float64
    m.def("phase2_compute_XtXz_bed",
          &phase2_compute_XtXz_bed_impl<double>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("blk_start"), py::arg("blk_end"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("inv_left"),
          py::arg("nvecs"),
          py::arg("vchunk"),
          py::arg("Xz2d"),
          py::arg("meansq"),
          py::arg("C") = py::none(),
          py::arg("R") = py::none(),
          py::arg("N_denom") = 0);
}
