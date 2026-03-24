// src/gwldcore.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
#include <mutex>
#include <unordered_map>
#include <atomic>

#include <limits>

#if defined(__linux__)
  #include <sched.h>
  #include <unistd.h>
#endif


#include "blas_compat.hpp"
#include "arch_compat.hpp"
#include "genotype.hpp"

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
    explicit AlignedBuffer(size_t count, size_t align = 64) { allocate(count, align); }
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

// --- verbosity gate --------------------------------------------
static std::atomic<bool> g_verbose{false};
static inline bool verbose_enabled() { return g_verbose.load(std::memory_order_relaxed); }
static inline void set_verbose(bool v) { g_verbose.store(v, std::memory_order_relaxed); }

// --- timers ------------------------------------------------------------------
struct P1Timers {
    double t_packZ_ms = 0.0;
    double t_packA_ms = 0.0;
    double t_gemm_ms  = 0.0;
    double t_scatt_ms = 0.0;

    void dump(int blk_start, int blk_end, int B, int vcount) const {
        if (!verbose_enabled()) return;
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
        if (!verbose_enabled()) return;
        std::fprintf(stderr,
            "[phase2] block [%d:%d) B=%d V=%d  pack=%.2f ms  gemm=%.2f ms  reduce=%.2f ms\n",
            blk_start, blk_end, B, nvecs, t_pack_ms, t_gemm_ms, t_reduce_ms);
    }
};

static inline int clampi(int x, int lo, int hi) {
    return std::max(lo, std::min(hi, x));
}
static inline int round_up(int x, int m) {
    return ((x + m - 1) / m) * m;
}
static inline int ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}


// ------------------------------- Small helpers -------------------------------
static void prefetch_bed_block_py(const std::string& bed_prefix,
                                  const std::string& fam_path,
                                  int blk_start, int blk_end,
                                  int ahead_blocks)
{
#if defined(__linux__)
    py::gil_scoped_release nogil;
    const std::string bed_path = bed_prefix + ".bed";
    prefetch_bed_block(bed_path, fam_path, blk_start, blk_end, ahead_blocks);
#else
    (void)bed_prefix; (void)fam_path; (void)blk_start; (void)blk_end; (void)ahead_blocks;
#endif
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

inline void cblas_taxpy(int n, float  a, const float*  x, int incx, float*  y, int incy){ cblas_saxpy(n,a,x,incx,y,incy); }
inline void cblas_taxpy(int n, double a, const double* x, int incx, double* y, int incy){ cblas_daxpy(n,a,x,incx,y,incy); }

// Parse optional row_sel (indices of individuals to keep), cached per-thread.
// This avoids rebuilding the same rows vector every block call.
static const std::vector<int>& parse_row_sel(py::object row_sel_obj, int64_t N_total) {
    struct Cache {
        PyObject* key = nullptr;  // row_sel_obj.ptr() or nullptr for None
        int64_t   N_total = -1;
        std::vector<int> rows;
    };
    static thread_local Cache C;

    PyObject* k = row_sel_obj.is_none() ? nullptr : row_sel_obj.ptr();

    if (C.key == k && C.N_total == N_total && !C.rows.empty()) {
        return C.rows;
    }

    C.key = k;
    C.N_total = N_total;
    C.rows.clear();
    // C.rows.shrink_to_fit(); // optional; you can remove if you prefer keeping capacity

    if (row_sel_obj.is_none()) {
        C.rows.resize((size_t)N_total);
        for (int64_t i = 0; i < N_total; ++i) C.rows[(size_t)i] = (int)i;
        return C.rows;
    }

    py::array idx = row_sel_obj.cast<py::array>();
    py::buffer_info bi = idx.request();
    C.rows.resize((size_t)bi.shape[0]);

    if (bi.format == py::format_descriptor<int32_t>::format()) {
        auto p = static_cast<const int32_t*>(bi.ptr);
        for (ssize_t i = 0; i < bi.shape[0]; ++i) C.rows[(size_t)i] = (int)p[i];
    } else {
        auto p = static_cast<const int64_t*>(bi.ptr);
        for (ssize_t i = 0; i < bi.shape[0]; ++i) C.rows[(size_t)i] = (int)p[i];
    }
    return C.rows;
}


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
                                      bool project_right = false,
                                      py::object C_opt = py::none(),
                                      py::object R_opt = py::none())
{
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    // ---- rows & block genotype ----
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    std::vector<T> Geno; // (N x L), column-major
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    if (project_right && !C_opt.is_none() && !R_opt.is_none()) {
    py::array_t<T, py::array::f_style | py::array::forcecast> C = C_opt.cast<py::array_t<T>>();
    py::array_t<T, py::array::f_style | py::array::forcecast> R = R_opt.cast<py::array_t<T>>();
    auto Ci = C.request();
    auto Ri = R.request();
    const int p = (int)Ci.shape[1];
    if ((int)Ci.shape[0] != N || (int)Ri.shape[0] != p || (int)Ri.shape[1] != N)
        throw std::runtime_error("C/R shape mismatch in phase1");

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
                if (a == T(0)) continue;

                const double ad = (double)a;
                if (!(ad > 0.0) || !std::isfinite(ad)) continue;

                const double invd = (double)inv[i];
                if (!std::isfinite(invd) || invd <= 0.0) continue;

                const double sc = invd * std::sqrt(ad);
                if (!std::isfinite(sc)) continue;

                if constexpr (std::is_same_v<T, float>) {
                    if (sc > (double)std::numeric_limits<float>::max()) continue;
                }

                ++cnt;
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
                if (a == T(0)) continue;

                const double ad = (double)a;
                if (!(ad > 0.0) || !std::isfinite(ad)) continue;

                const double invd = (double)inv[i];
                if (!std::isfinite(invd) || invd <= 0.0) continue;

                const double sc = invd * std::sqrt(ad);
                if (!std::isfinite(sc)) continue;

                if constexpr (std::is_same_v<T, float>) {
                    if (sc > (double)std::numeric_limits<float>::max()) continue;
                }

                const int pos = base + f++;
                rowind_buf.ptr[(size_t)pos] = i;
                scale_buf.ptr [(size_t)pos] = (T)sc;
            }
        }
    }

    // ---- First-touch Xz2d (only once at start-of-tile) ----
    if (blk_start == 0) {
        const size_t Q = (size_t)B * (size_t)v_count;
        const size_t elems_per_page = (size_t)((4096 / sizeof(T)) ? (4096 / sizeof(T)) : 512);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t g = 0; g < (ptrdiff_t)Q; ++g) {
            T* col = Xptr + (size_t)g * (size_t)ldc;
            for (size_t r = 0; r < (size_t)N; r += elems_per_page) {
                col[r] += T(0); // write to establish page ownership
            }
        }
    }

    // ---- constants / timers ----
    const char* s = std::getenv("SUMMIT_P1_NTILE");
    int NTILE = s ? std::max(64, std::atoi(s)) : (std::is_same_v<T,double> ? 256 : 512);
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
                    const T   ssc = scale_buf.ptr [(size_t)k0 + (size_t)c];
                    const T* src  = Geno.data() + (size_t)snp * (size_t)N + (size_t)n0;
                    T*       dst  = A_tile.ptr  + (size_t)c   * (size_t)Nt;
                    #pragma omp simd
                    for (int r = 0; r < Nt; ++r) dst[r] = src[r] * ssc;
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
                                 py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d,      // (N x (B*V)), K-major
                                 py::array_t<T, py::array::c_style | py::array::forcecast> meansq,    // (M x B), row-major
                                 py::object C_opt, py::object R_opt,
                                 int N_denom)
{
    // -------------------- tiny helpers --------------------
    auto round_down = [](int x, int m) { return (m > 0) ? (x / m) * m : x; };

    // -------------------- paths and shape checks --------------------
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    // Use cached rows (no copy = no alloc)
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    // Read & standardize geno block -> Geno (N x L), col-major
    int N = 0, L = 0;
    std::vector<T> Geno;
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // Optional projection: Geno <- Geno - C*(R*Geno)
    if (!C_opt.is_none() && !R_opt.is_none()) {
        py::array_t<T, py::array::f_style | py::array::forcecast> C =
            C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        py::array_t<T, py::array::f_style | py::array::forcecast> R =
            R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Ci = C.request();
        auto Ri = R.request();
        const int p = (int)Ci.shape[1];
        if ((int)Ci.shape[0] != N || (int)Ri.shape[0] != p || (int)Ri.shape[1] != N)
            throw std::runtime_error("C/R shape mismatch");

        const T* Cptr = static_cast<const T*>(Ci.ptr);
        const T* Rptr = static_cast<const T*>(Ri.ptr);

        AlignedBuffer<T> tmpG((size_t)p * (size_t)L, 64);

        // tmpG = R * Geno    (p x L)
        gemm_col_major_nn<T>(p, L, N,
                             Rptr, p,
                             Geno.data(), N,
                             tmpG.ptr, p,
                             T(1), T(0));

        // Geno = Geno - C * tmpG   (N x L)
        gemm_col_major_nn<T>(N, L, p,
                             Cptr, N,
                             tmpG.ptr, p,
                             Geno.data(), N,
                             T(-1), T(1));
    }

    // inv_left (L,)
    auto Ii = inv_left.request();
    if (Ii.ndim != 1 || (int)Ii.shape[0] != L)
        throw std::runtime_error("inv_left shape mismatch");
    const T* inv = static_cast<const T*>(Ii.ptr);

    // Xz: (N x (B*V)), K-major
    auto Xi = Xz2d.request();
    if (Xi.ndim != 2 || (int)Xi.shape[0] != N)
        throw std::runtime_error("Xz2d shape mismatch");
    T* Xptr = static_cast<T*>(Xi.ptr);
    const int BV = (int)Xi.shape[1];
    if (nvecs <= 0 || (BV % nvecs) != 0)
        throw std::runtime_error("Xz2d col count must be multiple of nvecs");
    const int B = BV / nvecs;

    // meansq: (M x B), row-major
    auto Mi = meansq.request();
    if (Mi.ndim != 2 || (int)Mi.shape[1] != B)
        throw std::runtime_error("meansq shape mismatch");
    T* Mptr = static_cast<T*>(Mi.ptr);
    const int M = (int)Mi.shape[0];
    if (blk_end > M)
        throw std::runtime_error("meansq rows smaller than SNP count");

    // -------------------- denom & constants --------------------
    double denom = (double)N_denom - 1.0;
    if (denom <= 0.0) denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    const double invV = 1.0 / (double)nvecs;

    // -------------------- Tunable QPANEL --------------------
    const int Q = BV;
    int QPANEL = 32768;
    if (const char* s = std::getenv("SUMMIT_P2_QPANEL")) {
        QPANEL = std::atoi(s);
    }
    if (QPANEL <= 0) QPANEL = Q;              // 0 => no cap
    QPANEL = std::min(QPANEL, Q);
    QPANEL = round_down(QPANEL, 64);
    if (QPANEL < 64) QPANEL = std::min(Q, 64);

    // Workspace Work = (L x QPANEL), col-major, TLS reuse
    static thread_local AlignedBuffer<T> Work_panel_tls;
    const size_t needW = (size_t)L * (size_t)QPANEL;
    if (Work_panel_tls.n < needW) Work_panel_tls.allocate(needW, 64);
    T* Work = Work_panel_tls.ptr;

    // -------------------- reduction tuning --------------------
    // IBLK = how many SNP-rows (within this L-SNP block) you reduce at a time.
    // Big enough to amortize overhead, small enough to keep acc[] hot in cache.
    int IBLK = 2048;
    if (const char* s = std::getenv("SUMMIT_P2_IBLK")) {
        IBLK = std::atoi(s);
    }
    IBLK = clampi(IBLK, 512, 16384);
    IBLK = round_up(IBLK, 512);
    
    int REDUCE_THREADS = 1;
    if (const char* s = std::getenv("SUMMIT_P2_REDUCE_THREADS")) {
        REDUCE_THREADS = std::atoi(s);
    }
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = omp_get_max_threads();

    // No point asking for more threads than slabs:
    const int nslabs = ceil_div_i(L, IBLK);
    REDUCE_THREADS = clampi(REDUCE_THREADS, 1, std::max(1, nslabs));

    BlockTimers t;
    py::gil_scoped_release nogil;

    // -------------------- main loop over panels --------------------
    for (int q0 = 0; q0 < Q; q0 += QPANEL) {
        check_for_interrupt();

        const int q = std::min(QPANEL, Q - q0);
        const T* rhs = Xptr + (size_t)q0 * (size_t)N;

        // GEMM: Work(L x q) = Geno^T(L x N) * Xz(N x q)
        auto t2 = std::chrono::high_resolution_clock::now();
        gemm_col_major_tn<T>(L, q, N,
                             Geno.data(), N,
                             rhs,        N,
                             Work,       L,
                             T(1), T(0));
        auto t3 = std::chrono::high_resolution_clock::now();
        t.add_gemm(std::chrono::duration<double,std::milli>(t3 - t2).count());

        // Compute which bins this panel touches (<= B segments, no heap alloc).
        int seg_k[512];
        int seg_tcol0[512];
        int seg_len[512];
        if (B > 512) throw std::runtime_error("B too large for fixed seg buffers (increase 512).");
        int segments = 0;
        {
            int g = q0;
            const int g_end = q0 + q;
            while (g < g_end) {
                const int k = g / nvecs;
                const int v_in = g - k * nvecs;
                const int len = std::min(g_end - g, nvecs - v_in);
                seg_k[segments] = k;
                seg_tcol0[segments] = g - q0;
                seg_len[segments] = len;
                ++segments;
                g += len;
            }
        }

        auto t4 = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(REDUCE_THREADS)
#endif
        for (int i0 = 0; i0 < L; i0 += IBLK) {
            const int ib = std::min(IBLK, L - i0);

            // Per-thread accumulator (TLS) to avoid repeated heap allocs.
            static thread_local AlignedBuffer<double> acc_tls;
            if (acc_tls.n < (size_t)IBLK) acc_tls.allocate((size_t)IBLK, 64);
            double* acc = acc_tls.ptr;

            // For this SNP-slab, update all bins touched by the current panel.
            for (int sidx = 0; sidx < segments; ++sidx) {
                const int k    = seg_k[sidx];
                const int tcol = seg_tcol0[sidx];
                const int len  = seg_len[sidx];

                // acc[ii] = sum_{c in segment} Work(i0+ii, tcol+c)^2
                std::fill(acc, acc + ib, 0.0);

                for (int c = 0; c < len; ++c) {
                    const T* __restrict wcol =
                        Work + (size_t)(tcol + c) * (size_t)L + (size_t)i0;
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (int ii = 0; ii < ib; ++ii) {
                        const double w = (double)wcol[ii];
                        acc[ii] += w * w;
                    }
                }

                // Write-out with exact scaling factor
                T* __restrict out =
                    Mptr + ((size_t)(blk_start + i0) * (size_t)B + (size_t)k);

#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int ii = 0; ii < ib; ++ii) {
                    const double inv_i = (double)inv[i0 + ii];
                    const double inv2  = inv_i * inv_i * inv_denom2;
                    double upd = acc[ii] * invV * inv2;

                    // Guard: drop NaN/inf and drop values that would overflow float32.
                    if (!std::isfinite(upd)) {
                        continue;
                    }
                    if constexpr (std::is_same_v<T, float>) {
                        const double fmax = (double)std::numeric_limits<float>::max();
                        if (upd > fmax) {
                            continue;
                        }
                    }

                    // (Optional) tiny negative due to FP noise → clamp
                    if (upd < 0.0) upd = 0.0;

                    out[(size_t)ii * (size_t)B] += (T)upd;
                }
            }
        }

        auto t5 = std::chrono::high_resolution_clock::now();
        t.add_reduce(std::chrono::duration<double,std::milli>(t5 - t4).count());
    }

    t.dump(blk_start, blk_end, B, nvecs);
}

static inline int64_t llabs_i64(int64_t x) { return x >= 0 ? x : -x; }

template <typename T>
static void fill_random_probes_same_as_phase1(int L,
                                              int v_start,
                                              int v_count,
                                              const std::string& rand_dist,
                                              uint64_t root_seed,
                                              int block_start,
                                              std::vector<T>& Z)
{
    std::mt19937_64 rng(make_seed(root_seed, /*block=*/block_start, /*v0=*/v_start));
    std::normal_distribution<T> gN(0, (T)1);
    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_spherical  = (rand_dist == "spherical");

    Z.assign((size_t)L * (size_t)v_count, T(0));
    for (int c = 0; c < v_count; ++c) {
        long double ss = 0.0L;
        for (int r = 0; r < L; ++r) {
            T z = is_rademacher ? ((rng() & 1) ? T(+1) : T(-1)) : gN(rng);
            Z[(size_t)r + (size_t)c * (size_t)L] = z;
            if (is_spherical) ss += (long double)z * (long double)z;
        }
        if (is_spherical) {
            T scale = ss > 0.0L ? (T)std::sqrt((long double)L / ss) : T(1);
            for (int r = 0; r < L; ++r) {
                Z[(size_t)r + (size_t)c * (size_t)L] *= scale;
            }
        }
    }
}


template <typename T>
inline void gemv_col_major_t(int m, int n,
                             const T* A, int lda,
                             const T* x, int incx,
                             T* y, int incy,
                             T alpha = T(1), T beta = T(0)) {
    if constexpr (std::is_same_v<T,double>) {
        cblas_dgemv(CblasColMajor, CblasTrans,
                    m, n, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        cblas_sgemv(CblasColMajor, CblasTrans,
                    m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}

inline void cblas_tger(int m, int n,
                       float alpha,
                       const float* x, int incx,
                       const float* y, int incy,
                       float* A, int lda) {
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

inline void cblas_tger(int m, int n,
                       double alpha,
                       const double* x, int incx,
                       const double* y, int incy,
                       double* A, int lda) {
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <typename T>
static void project_target_block_inplace(std::vector<T>& G, int N, int L,
                                         const T* Cptr, const T* Rptr, int p,
                                         AlignedBuffer<T>& tmp_buf) {
    if (p <= 0) return;
    if (tmp_buf.n < (size_t)p * (size_t)L) {
        tmp_buf.allocate((size_t)p * (size_t)L, 64);
    }
    T* tmp = tmp_buf.ptr;

    gemm_col_major_nn<T>(/*m=*/p, /*n=*/L, /*k=*/N,
                         /*A=*/Rptr, /*lda=*/p,
                         /*B=*/G.data(), /*ldb=*/N,
                         /*C=*/tmp, /*ldc=*/p,
                         /*alpha=*/T(1), /*beta=*/T(0));

    gemm_col_major_nn<T>(/*m=*/N, /*n=*/L, /*k=*/p,
                         /*A=*/Cptr, /*lda=*/N,
                         /*B=*/tmp, /*ldb=*/p,
                         /*C=*/G.data(), /*ldc=*/N,
                         /*alpha=*/T(-1), /*beta=*/T(1));
}

template <typename T>
static void prepare_dense_P_block(
    int blk_start,
    int L,
    int B,
    const T* ann_blk_rowmajor,   // annot_all[blk_start:, :]
    const T* inv_blk,            // inv_all[blk_start:]
    int v_start,
    int v_count,
    const std::string& rand_dist,
    uint64_t root_seed,
    std::vector<T>& P            // output: (L x (B*v_count)), col-major
) {
    const int Q = B * v_count;
    P.assign((size_t)L * (size_t)Q, T(0));
    if (L <= 0 || B <= 0 || v_count <= 0) return;

    std::vector<T> Z;
    fill_random_probes_same_as_phase1<T>(
        L, v_start, v_count, rand_dist, root_seed, blk_start, Z
    ); // Z: (L x v_count), col-major

    // weights S: (L x B) conceptually, but fill directly into P
    for (int k = 0; k < B; ++k) {
        const int q0 = k * v_count;
        for (int vv = 0; vv < v_count; ++vv) {
            const T* zcol = Z.data() + (size_t)vv * (size_t)L;
            T* pcol = P.data() + (size_t)(q0 + vv) * (size_t)L;

            for (int i = 0; i < L; ++i) {
                const double a = (double)ann_blk_rowmajor[(size_t)i * (size_t)B + (size_t)k];
                const double invj = (double)inv_blk[(size_t)i];
                if (!(a > 0.0) || !std::isfinite(a) || !std::isfinite(invj) || invj <= 0.0) {
                    pcol[(size_t)i] = T(0);
                    continue;
                }
                const double s = invj * std::sqrt(a);
                const double val = s * (double)zcol[(size_t)i];
                if (!std::isfinite(val)) {
                    pcol[(size_t)i] = T(0);
                    continue;
                }
                if constexpr (std::is_same_v<T,float>) {
                    const double fmax = (double)std::numeric_limits<float>::max();
                    if (std::fabs(val) > fmax) {
                        pcol[(size_t)i] = T(0);
                        continue;
                    }
                }
                pcol[(size_t)i] = (T)val;
            }
        }
    }
}

template <typename T>
static void prepare_dense_Wex_block(
    int L,
    int B,
    const T* ann_blk_rowmajor,
    const T* inv_blk,
    std::vector<T>& Wex          // output: (L x B), col-major
) {
    Wex.assign((size_t)L * (size_t)B, T(0));
    for (int k = 0; k < B; ++k) {
        T* col = Wex.data() + (size_t)k * (size_t)L;
        for (int i = 0; i < L; ++i) {
            const double a = (double)ann_blk_rowmajor[(size_t)i * (size_t)B + (size_t)k];
            const double invj = (double)inv_blk[(size_t)i];
            if (!(a > 0.0) || !std::isfinite(a) || !std::isfinite(invj) || invj <= 0.0) {
                col[(size_t)i] = T(0);
                continue;
            }
            const double val = invj * invj * a;
            if (!std::isfinite(val)) {
                col[(size_t)i] = T(0);
                continue;
            }
            if constexpr (std::is_same_v<T,float>) {
                const double fmax = (double)std::numeric_limits<float>::max();
                if (std::fabs(val) > fmax) {
                    col[(size_t)i] = T(0);
                    continue;
                }
            }
            col[(size_t)i] = (T)val;
        }
    }
}

enum class PartialMode : int {
    LEFT  = 0,
    RIGHT = 1,
    SELF  = 2
};

template <typename T>
static void accum_partial_rp_block(
    PartialMode mode,
    const std::vector<T>& Xs,  // (N x Ls), col-major
    int N, int Ls,
    const std::vector<T>& P,   // (Ls x Q), col-major
    int Q,
    const std::vector<T>& Xt_left, // (N x Lt), col-major
    int Lt,
    const int64_t* bp_s,
    const int64_t* bp_t,
    int64_t window_bp,
    T* Ycol,    // (Lt x Q), col-major
    int ldY     // = Lt
) {
    if (Ls <= 0 || Lt <= 0 || Q <= 0) return;

    AlignedBuffer<T> Ucur((size_t)N * (size_t)Q, 64);
    std::fill(Ucur.ptr, Ucur.ptr + Ucur.n, T(0));

    std::vector<T> yrow((size_t)Q, T(0));

    auto init_from_slice = [&](int lo, int hi) {
        std::fill(Ucur.ptr, Ucur.ptr + Ucur.n, T(0));
        const int len = hi - lo;
        if (len <= 0) return;
        gemm_col_major_nn<T>(
            /*m=*/N, /*n=*/Q, /*k=*/len,
            /*A=*/Xs.data() + (size_t)lo * (size_t)N, /*lda=*/N,
            /*B=*/P.data()  + (size_t)lo,             /*ldb=*/Ls,
            /*C=*/Ucur.ptr,                           /*ldc=*/N,
            /*alpha=*/T(1), /*beta=*/T(0)
        );
    };

    int lo = 0, hi = 0;

    if (mode == PartialMode::LEFT) {
        lo = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[0] - window_bp) - bp_s);
        hi = Ls;
        init_from_slice(lo, hi);

        for (int i = 0; i < Lt; ++i) {
            gemv_col_major_t<T>(
                /*m=*/N, /*n=*/Q,
                /*A=*/Ucur.ptr, /*lda=*/N,
                /*x=*/Xt_left.data() + (size_t)i * (size_t)N, /*incx=*/1,
                /*y=*/yrow.data(), /*incy=*/1,
                /*alpha=*/T(1), /*beta=*/T(0)
            );
            for (int q = 0; q < Q; ++q) {
                Ycol[(size_t)q * (size_t)ldY + (size_t)i] += yrow[(size_t)q];
            }

            if (i + 1 < Lt) {
                int lo_next = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[i + 1] - window_bp) - bp_s);
                for (int j = lo; j < lo_next; ++j) {
                    cblas_tger(
                        N, Q, T(-1),
                        Xs.data() + (size_t)j * (size_t)N, 1,
                        P.data() + (size_t)j, Ls,
                        Ucur.ptr, N
                    );
                }
                lo = lo_next;
            }
        }
        return;
    }

    if (mode == PartialMode::RIGHT) {
        lo = 0;
        hi = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[0] + window_bp) - bp_s);
        init_from_slice(lo, hi);

        for (int i = 0; i < Lt; ++i) {
            gemv_col_major_t<T>(
                /*m=*/N, /*n=*/Q,
                /*A=*/Ucur.ptr, /*lda=*/N,
                /*x=*/Xt_left.data() + (size_t)i * (size_t)N, /*incx=*/1,
                /*y=*/yrow.data(), /*incy=*/1,
                /*alpha=*/T(1), /*beta=*/T(0)
            );
            for (int q = 0; q < Q; ++q) {
                Ycol[(size_t)q * (size_t)ldY + (size_t)i] += yrow[(size_t)q];
            }

            if (i + 1 < Lt) {
                int hi_next = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[i + 1] + window_bp) - bp_s);
                for (int j = hi; j < hi_next; ++j) {
                    cblas_tger(
                        N, Q, T(+1),
                        Xs.data() + (size_t)j * (size_t)N, 1,
                        P.data() + (size_t)j, Ls,
                        Ucur.ptr, N
                    );
                }
                hi = hi_next;
            }
        }
        return;
    }

    // SELF
    lo = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[0] - window_bp) - bp_s);
    hi = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[0] + window_bp) - bp_s);
    init_from_slice(lo, hi);

    for (int i = 0; i < Lt; ++i) {
        gemv_col_major_t<T>(
            /*m=*/N, /*n=*/Q,
            /*A=*/Ucur.ptr, /*lda=*/N,
            /*x=*/Xt_left.data() + (size_t)i * (size_t)N, /*incx=*/1,
            /*y=*/yrow.data(), /*incy=*/1,
            /*alpha=*/T(1), /*beta=*/T(0)
        );
        for (int q = 0; q < Q; ++q) {
            Ycol[(size_t)q * (size_t)ldY + (size_t)i] += yrow[(size_t)q];
        }

        if (i + 1 < Lt) {
            int lo_next = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[i + 1] - window_bp) - bp_s);
            int hi_next = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[i + 1] + window_bp) - bp_s);

            for (int j = lo; j < lo_next; ++j) {
                cblas_tger(
                    N, Q, T(-1),
                    Xs.data() + (size_t)j * (size_t)N, 1,
                    P.data() + (size_t)j, Ls,
                    Ucur.ptr, N
                );
            }
            for (int j = hi; j < hi_next; ++j) {
                cblas_tger(
                    N, Q, T(+1),
                    Xs.data() + (size_t)j * (size_t)N, 1,
                    P.data() + (size_t)j, Ls,
                    Ucur.ptr, N
                );
            }

            lo = lo_next;
            hi = hi_next;
        }
    }
}

template <typename T>
static void accum_partial_exact_block(
    PartialMode mode,
    const T* Cross,     // (Ls x Lt), col-major
    int Ls, int Lt,
    const std::vector<T>& Wex,  // (Ls x B), col-major
    int B,
    const int64_t* bp_s,
    const int64_t* bp_t,
    int64_t window_bp,
    T* Ecol,            // (Lt x B), col-major
    int ldE             // = Lt
) {
    if (Ls <= 0 || Lt <= 0 || B <= 0) return;

    int lo = 0, hi = 0;

    auto accum_row = [&](int i, int lo_i, int hi_i) {
        if (lo_i >= hi_i) return;
        const T* ccol = Cross + (size_t)i * (size_t)Ls;
        for (int k = 0; k < B; ++k) {
            const T* wcol = Wex.data() + (size_t)k * (size_t)Ls;
            double acc = 0.0;
            for (int j = lo_i; j < hi_i; ++j) {
                const double x = (double)ccol[(size_t)j];
                acc += x * x * (double)wcol[(size_t)j];
            }
            Ecol[(size_t)k * (size_t)ldE + (size_t)i] += (T)acc;
        }
    };

    if (mode == PartialMode::LEFT) {
        lo = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[0] - window_bp) - bp_s);
        hi = Ls;
        for (int i = 0; i < Lt; ++i) {
            accum_row(i, lo, hi);
            if (i + 1 < Lt) {
                lo = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[i + 1] - window_bp) - bp_s);
            }
        }
        return;
    }

    if (mode == PartialMode::RIGHT) {
        lo = 0;
        hi = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[0] + window_bp) - bp_s);
        for (int i = 0; i < Lt; ++i) {
            accum_row(i, lo, hi);
            if (i + 1 < Lt) {
                hi = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[i + 1] + window_bp) - bp_s);
            }
        }
        return;
    }

    // SELF
    lo = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[0] - window_bp) - bp_s);
    hi = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[0] + window_bp) - bp_s);
    for (int i = 0; i < Lt; ++i) {
        accum_row(i, lo, hi);
        if (i + 1 < Lt) {
            lo = (int)(std::lower_bound(bp_s, bp_s + Ls, bp_t[i + 1] - window_bp) - bp_s);
            hi = (int)(std::upper_bound(bp_s, bp_s + Ls, bp_t[i + 1] + window_bp) - bp_s);
        }
    }
}

template <typename T>
void phase2_compute_local_rp_sliding_bed_impl(
    const std::string &bed_prefix,
    const std::string &fam_path,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> block_starts, // one chromosome only
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> block_ends,
    py::object row_sel_obj,
    int ddof,
    py::array_t<T, py::array::c_style | py::array::forcecast> inv_all,             // (M,)
    py::array_t<T, py::array::c_style | py::array::forcecast> annot_all,           // (M x B), row-major
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> bp_all,        // (M,)
    int64_t window_bp,
    int v_start,
    int v_count,
    const std::string &rand_dist,
    py::object seed_obj,
    py::array_t<T, py::array::c_style | py::array::forcecast> meansq_accum,        // (M x B), row-major
    py::object C_opt,
    py::object R_opt,
    int N_denom)
{
    if (v_count <= 0) return;

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);

    auto BSi = block_starts.request();
    auto BEi = block_ends.request();
    if (BSi.ndim != 1 || BEi.ndim != 1 || BSi.shape[0] != BEi.shape[0]) {
        throw std::runtime_error("block_starts / block_ends shape mismatch");
    }
    const int nb = (int)BSi.shape[0];
    const int32_t* blk_s = static_cast<const int32_t*>(BSi.ptr);
    const int32_t* blk_e = static_cast<const int32_t*>(BEi.ptr);

    auto Ii = inv_all.request();
    auto Ai = annot_all.request();
    auto Bi = bp_all.request();
    auto Mi = meansq_accum.request();

    if (Ii.ndim != 1 || (int64_t)Ii.shape[0] != M_total) {
        throw std::runtime_error("inv_all shape mismatch");
    }
    if (Ai.ndim != 2 || (int64_t)Ai.shape[0] != M_total) {
        throw std::runtime_error("annot_all shape mismatch");
    }
    if (Bi.ndim != 1 || (int64_t)Bi.shape[0] != M_total) {
        throw std::runtime_error("bp_all shape mismatch");
    }
    if (Mi.ndim != 2 || (int64_t)Mi.shape[0] != M_total) {
        throw std::runtime_error("meansq_accum shape mismatch");
    }

    const T* invp = static_cast<const T*>(Ii.ptr);
    const T* annp = static_cast<const T*>(Ai.ptr);
    const int B = (int)Ai.shape[1];
    const int64_t* bpp = static_cast<const int64_t*>(Bi.ptr);
    T* Mptr = static_cast<T*>(Mi.ptr);

    if ((int)Mi.shape[1] != B) {
        throw std::runtime_error("meansq_accum second dimension must match annot_all second dimension");
    }

    const int Q = B * v_count;
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    py::array_t<T, py::array::f_style | py::array::forcecast> Carr;
    py::array_t<T, py::array::f_style | py::array::forcecast> Rarr;
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;

    if (have_proj) {
        Carr = C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        Rarr = R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Cinfo = Carr.request();
        auto Rinfo = Rarr.request();

        p = (int)Cinfo.shape[1];
        if ((int)Cinfo.shape[0] != (int)rows.size() ||
            (int)Rinfo.shape[0] != p ||
            (int)Rinfo.shape[1] != (int)rows.size()) {
            throw std::runtime_error("C/R shape mismatch in phase2_compute_local_rp_sliding_bed");
        }
        Cptr = static_cast<const T*>(Cinfo.ptr);
        Rptr = static_cast<const T*>(Rinfo.ptr);
    }

    const uint64_t root_seed =
        seed_obj.is_none() ? uint64_t(std::random_device{}()) : seed_obj.cast<uint64_t>();

    double denom = (double)N_denom - 1.0;
    if (denom <= 0.0) denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    int maxL = 0;
    for (int bi = 0; bi < nb; ++bi) {
        const int s = blk_s[bi];
        const int e = blk_e[bi];
        if (s < 0 || e <= s || e > M_total) {
            throw std::runtime_error("Invalid block_starts / block_ends");
        }
        maxL = std::max(maxL, e - s);
    }
    if (maxL <= 0) return;

    // Determine N by reading one tiny block lazily later? We need full sketch alloc.
    int N = 0, Lchk = 0;
    std::vector<T> Xtmp;
    read_block_standardized<T>(bed_path, fam_path, blk_s[0], blk_e[0], rows, ddof, Xtmp, N, Lchk);
    if (N <= 0) return;

    AlignedBuffer<T> ProjTmpBuf(have_proj ? ((size_t)p * (size_t)maxL) : 0, 64);
    AlignedBuffer<T> Sfull((size_t)N * (size_t)Q, 64);     // (N x Q), col-major
    AlignedBuffer<T> Yfull((size_t)maxL * (size_t)Q, 64);  // (Lt x Q), col-major
    std::fill(Sfull.ptr, Sfull.ptr + Sfull.n, T(0));

    py::gil_scoped_release nogil;

    std::vector<int64_t> blk_bp0((size_t)nb), blk_bp1((size_t)nb);
    for (int bi = 0; bi < nb; ++bi) {
        blk_bp0[(size_t)bi] = bpp[blk_s[bi]];
        blk_bp1[(size_t)bi] = bpp[blk_e[bi] - 1];
    }

    int active_lo = 0;
    int active_hi = -1;
    int full_lo_prev = 0;
    int full_hi_prev = -1;

    auto update_full_block = [&](int bi, T sign) {
        const int sstart = blk_s[bi];
        const int send   = blk_e[bi];
        const int Ls = send - sstart;
        if (Ls <= 0) return;

        int Ns = 0, Ls_chk = 0;
        std::vector<T> Xs_raw;
        read_block_standardized<T>(bed_path, fam_path, sstart, send, rows, ddof, Xs_raw, Ns, Ls_chk);
        if (Ns != N || Ls_chk != Ls) {
            throw std::runtime_error("Unexpected source block dimensions in full update");
        }

        const T* ann_blk = annp + (size_t)sstart * (size_t)B;
        const T* inv_blk = invp + (size_t)sstart;
        std::vector<T> P;
        prepare_dense_P_block<T>(
            sstart, Ls, B, ann_blk, inv_blk,
            v_start, v_count, rand_dist, root_seed, P
        );

        AlignedBuffer<T> U((size_t)N * (size_t)Q, 64);
        gemm_col_major_nn<T>(
            /*m=*/N, /*n=*/Q, /*k=*/Ls,
            /*A=*/Xs_raw.data(), /*lda=*/N,
            /*B=*/P.data(),      /*ldb=*/Ls,
            /*C=*/U.ptr,         /*ldc=*/N,
            /*alpha=*/T(1), /*beta=*/T(0)
        );

        const int nelt = N * Q;
        cblas_taxpy(nelt, sign, U.ptr, 1, Sfull.ptr, 1);
    };

    for (int ti = 0; ti < nb; ++ti) {
        check_for_interrupt();

        const int tstart = blk_s[ti];
        const int tend   = blk_e[ti];
        const int Lt = tend - tstart;
        if (Lt <= 0) continue;

        const int64_t t_bp0 = blk_bp0[(size_t)ti];
        const int64_t t_bp1 = blk_bp1[(size_t)ti];

        while (active_lo < nb && blk_bp1[(size_t)active_lo] < t_bp0 - window_bp) {
            ++active_lo;
        }
        while (active_hi + 1 < nb && blk_bp0[(size_t)(active_hi + 1)] <= t_bp1 + window_bp) {
            ++active_hi;
        }

        int full_lo = 0;
        while (full_lo < nb && blk_bp0[(size_t)full_lo] < t_bp1 - window_bp) {
            ++full_lo;
        }
        int full_hi = -1;
        while (full_hi + 1 < nb && blk_bp1[(size_t)(full_hi + 1)] <= t_bp0 + window_bp) {
            ++full_hi;
        }

        if (full_lo < active_lo) full_lo = active_lo;
        if (full_hi > active_hi) full_hi = active_hi;
        if (full_lo > full_hi) {
            full_lo = 1;
            full_hi = 0; // empty interval
        }

        // Update full sketch monotonically
        if (full_hi_prev >= full_lo_prev) {
            for (int b = full_lo_prev; b < std::min(full_lo, full_hi_prev + 1); ++b) {
                update_full_block(b, T(-1));
            }
        }
        if (full_hi >= full_lo) {
            const int add_from = std::max(full_hi_prev + 1, full_lo);
            for (int b = add_from; b <= full_hi; ++b) {
                update_full_block(b, T(+1));
            }
        }

        full_lo_prev = full_lo;
        full_hi_prev = full_hi;

        // Read target block once
        int Nt = 0, Lt_chk = 0;
        std::vector<T> Xt_raw;
        read_block_standardized<T>(bed_path, fam_path, tstart, tend, rows, ddof, Xt_raw, Nt, Lt_chk);
        if (Nt != N || Lt_chk != Lt) {
            throw std::runtime_error("Unexpected target block dimensions");
        }

        std::vector<T> Xt_left = Xt_raw;
        if (have_proj) {
            project_target_block_inplace<T>(Xt_left, N, Lt, Cptr, Rptr, p, ProjTmpBuf);
        }

        // Y = Xt_left^T Sfull
        gemm_col_major_tn<T>(
            /*m=*/Lt, /*n=*/Q, /*k=*/N,
            /*A=*/Xt_left.data(), /*lda=*/N,
            /*B=*/Sfull.ptr,      /*ldb=*/N,
            /*C=*/Yfull.ptr,      /*ldc=*/Lt,
            /*alpha=*/T(1), /*beta=*/T(0)
        );

        const int64_t* bp_t = bpp + tstart;

        const bool self_full = (full_lo <= ti && ti <= full_hi);

        // Left partial blocks
        const int left_end = std::min(full_lo - 1, ti - 1);
        for (int bi = active_lo; bi <= left_end; ++bi) {
            const int sstart = blk_s[bi];
            const int send   = blk_e[bi];
            const int Ls = send - sstart;

            int Ns = 0, Ls_chk = 0;
            std::vector<T> Xs_raw;
            read_block_standardized<T>(bed_path, fam_path, sstart, send, rows, ddof, Xs_raw, Ns, Ls_chk);
            if (Ns != N || Ls_chk != Ls) {
                throw std::runtime_error("Unexpected partial-left source block dimensions");
            }

            const T* ann_blk = annp + (size_t)sstart * (size_t)B;
            const T* inv_blk = invp + (size_t)sstart;
            std::vector<T> P;
            prepare_dense_P_block<T>(
                sstart, Ls, B, ann_blk, inv_blk,
                v_start, v_count, rand_dist, root_seed, P
            );

            accum_partial_rp_block<T>(
                PartialMode::LEFT,
                Xs_raw, N, Ls,
                P, Q,
                Xt_left, Lt,
                bpp + sstart,
                bp_t,
                window_bp,
                Yfull.ptr,
                Lt
            );
        }

        // Self partial if needed
        if (!self_full && active_lo <= ti && ti <= active_hi) {
            const int sstart = blk_s[ti];
            const int send   = blk_e[ti];
            const int Ls = send - sstart;

            const T* ann_blk = annp + (size_t)sstart * (size_t)B;
            const T* inv_blk = invp + (size_t)sstart;
            std::vector<T> P;
            prepare_dense_P_block<T>(
                sstart, Ls, B, ann_blk, inv_blk,
                v_start, v_count, rand_dist, root_seed, P
            );

            accum_partial_rp_block<T>(
                PartialMode::SELF,
                Xt_raw, N, Ls,
                P, Q,
                Xt_left, Lt,
                bp_t,
                bp_t,
                window_bp,
                Yfull.ptr,
                Lt
            );
        }

        // Right partial blocks
        const int right_beg = std::max(full_hi + 1, ti + 1);
        for (int bi = right_beg; bi <= active_hi; ++bi) {
            const int sstart = blk_s[bi];
            const int send   = blk_e[bi];
            const int Ls = send - sstart;

            int Ns = 0, Ls_chk = 0;
            std::vector<T> Xs_raw;
            read_block_standardized<T>(bed_path, fam_path, sstart, send, rows, ddof, Xs_raw, Ns, Ls_chk);
            if (Ns != N || Ls_chk != Ls) {
                throw std::runtime_error("Unexpected partial-right source block dimensions");
            }

            const T* ann_blk = annp + (size_t)sstart * (size_t)B;
            const T* inv_blk = invp + (size_t)sstart;
            std::vector<T> P;
            prepare_dense_P_block<T>(
                sstart, Ls, B, ann_blk, inv_blk,
                v_start, v_count, rand_dist, root_seed, P
            );

            accum_partial_rp_block<T>(
                PartialMode::RIGHT,
                Xs_raw, N, Ls,
                P, Q,
                Xt_left, Lt,
                bpp + sstart,
                bp_t,
                window_bp,
                Yfull.ptr,
                Lt
            );
        }

        // Finalize subtraction onto pre-divide scale:
        // meansq_accum -= inv_i^2 / D^2 * sum_{v in tile} y_{ikv}^2
        for (int i = 0; i < Lt; ++i) {
            const double invi = (double)invp[(size_t)tstart + (size_t)i];
            if (!std::isfinite(invi) || invi <= 0.0) continue;
            const double left_scale = invi * invi * inv_denom2;

            T* out = Mptr + ((size_t)(tstart + i) * (size_t)B);

            for (int k = 0; k < B; ++k) {
                double rp = 0.0;
                const int q0 = k * v_count;
                for (int vv = 0; vv < v_count; ++vv) {
                    const double y = (double)Yfull.ptr[(size_t)(q0 + vv) * (size_t)Lt + (size_t)i];
                    rp += y * y;
                }
                const double upd = -left_scale * rp;
                if (!std::isfinite(upd)) continue;
                if constexpr (std::is_same_v<T,float>) {
                    const double fmax = (double)std::numeric_limits<float>::max();
                    if (std::fabs(upd) > fmax) continue;
                }
                out[(size_t)k] += (T)upd;
            }
        }
    }
}

template <typename T>
void phase2_compute_local_exact_bed_impl(
    const std::string &bed_prefix,
    const std::string &fam_path,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> block_starts, // one chromosome only
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> block_ends,
    py::object row_sel_obj,
    int ddof,
    py::array_t<T, py::array::c_style | py::array::forcecast> inv_all,             // (M,)
    py::array_t<T, py::array::c_style | py::array::forcecast> annot_all,           // (M x B), row-major
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> bp_all,        // (M,)
    int64_t window_bp,
    py::array_t<T, py::array::c_style | py::array::forcecast> meansq_accum,        // (M x B), row-major
    double exact_scale,
    py::object C_opt,
    py::object R_opt,
    int N_denom)
{
    if (exact_scale == 0.0) return;

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);

    auto BSi = block_starts.request();
    auto BEi = block_ends.request();
    if (BSi.ndim != 1 || BEi.ndim != 1 || BSi.shape[0] != BEi.shape[0]) {
        throw std::runtime_error("block_starts / block_ends shape mismatch");
    }
    const int nb = (int)BSi.shape[0];
    const int32_t* blk_s = static_cast<const int32_t*>(BSi.ptr);
    const int32_t* blk_e = static_cast<const int32_t*>(BEi.ptr);

    auto Ii = inv_all.request();
    auto Ai = annot_all.request();
    auto Bi = bp_all.request();
    auto Mi = meansq_accum.request();

    if (Ii.ndim != 1 || (int64_t)Ii.shape[0] != M_total) {
        throw std::runtime_error("inv_all shape mismatch");
    }
    if (Ai.ndim != 2 || (int64_t)Ai.shape[0] != M_total) {
        throw std::runtime_error("annot_all shape mismatch");
    }
    if (Bi.ndim != 1 || (int64_t)Bi.shape[0] != M_total) {
        throw std::runtime_error("bp_all shape mismatch");
    }
    if (Mi.ndim != 2 || (int64_t)Mi.shape[0] != M_total) {
        throw std::runtime_error("meansq_accum shape mismatch");
    }

    const T* invp = static_cast<const T*>(Ii.ptr);
    const T* annp = static_cast<const T*>(Ai.ptr);
    const int B = (int)Ai.shape[1];
    const int64_t* bpp = static_cast<const int64_t*>(Bi.ptr);
    T* Mptr = static_cast<T*>(Mi.ptr);

    if ((int)Mi.shape[1] != B) {
        throw std::runtime_error("meansq_accum second dimension must match annot_all second dimension");
    }

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    py::array_t<T, py::array::f_style | py::array::forcecast> Carr;
    py::array_t<T, py::array::f_style | py::array::forcecast> Rarr;
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;

    if (have_proj) {
        Carr = C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        Rarr = R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Cinfo = Carr.request();
        auto Rinfo = Rarr.request();

        p = (int)Cinfo.shape[1];
        if ((int)Cinfo.shape[0] != (int)rows.size() ||
            (int)Rinfo.shape[0] != p ||
            (int)Rinfo.shape[1] != (int)rows.size()) {
            throw std::runtime_error("C/R shape mismatch in phase2_compute_local_exact_bed");
        }
        Cptr = static_cast<const T*>(Cinfo.ptr);
        Rptr = static_cast<const T*>(Rinfo.ptr);
    }

    double denom = (double)N_denom - 1.0;
    if (denom <= 0.0) denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    int maxL = 0;
    for (int bi = 0; bi < nb; ++bi) {
        const int s = blk_s[bi];
        const int e = blk_e[bi];
        if (s < 0 || e <= s || e > M_total) {
            throw std::runtime_error("Invalid block_starts / block_ends");
        }
        maxL = std::max(maxL, e - s);
    }
    if (maxL <= 0) return;

    AlignedBuffer<T> ProjTmpBuf(have_proj ? ((size_t)p * (size_t)maxL) : 0, 64);
    AlignedBuffer<T> CrossBuf((size_t)maxL * (size_t)maxL, 64);

    py::gil_scoped_release nogil;

    std::vector<int64_t> blk_bp0((size_t)nb), blk_bp1((size_t)nb);
    for (int bi = 0; bi < nb; ++bi) {
        blk_bp0[(size_t)bi] = bpp[blk_s[bi]];
        blk_bp1[(size_t)bi] = bpp[blk_e[bi] - 1];
    }

    int active_lo = 0;
    int active_hi = -1;

    for (int ti = 0; ti < nb; ++ti) {
        check_for_interrupt();

        const int tstart = blk_s[ti];
        const int tend   = blk_e[ti];
        const int Lt = tend - tstart;
        if (Lt <= 0) continue;

        const int64_t t_bp0 = blk_bp0[(size_t)ti];
        const int64_t t_bp1 = blk_bp1[(size_t)ti];

        while (active_lo < nb && blk_bp1[(size_t)active_lo] < t_bp0 - window_bp) {
            ++active_lo;
        }
        while (active_hi + 1 < nb && blk_bp0[(size_t)(active_hi + 1)] <= t_bp1 + window_bp) {
            ++active_hi;
        }

        int full_lo = 0;
        while (full_lo < nb && blk_bp0[(size_t)full_lo] < t_bp1 - window_bp) {
            ++full_lo;
        }
        int full_hi = -1;
        while (full_hi + 1 < nb && blk_bp1[(size_t)(full_hi + 1)] <= t_bp0 + window_bp) {
            ++full_hi;
        }

        if (full_lo < active_lo) full_lo = active_lo;
        if (full_hi > active_hi) full_hi = active_hi;
        if (full_lo > full_hi) {
            full_lo = 1;
            full_hi = 0;
        }

        const bool self_full = (full_lo <= ti && ti <= full_hi);

        // Read target block once
        int N = 0, Lt_chk = 0;
        std::vector<T> Xt_raw;
        read_block_standardized<T>(bed_path, fam_path, tstart, tend, rows, ddof, Xt_raw, N, Lt_chk);
        if (Lt_chk != Lt) {
            throw std::runtime_error("Unexpected target block dimensions");
        }

        std::vector<T> Xt_left = Xt_raw;
        if (have_proj) {
            project_target_block_inplace<T>(Xt_left, N, Lt, Cptr, Rptr, p, ProjTmpBuf);
        }

        std::vector<T> Ecol((size_t)Lt * (size_t)B, T(0)); // col-major Lt x B
        const int64_t* bp_t = bpp + tstart;

        for (int bi = active_lo; bi <= active_hi; ++bi) {
            check_for_interrupt();

            const int sstart = blk_s[bi];
            const int send   = blk_e[bi];
            const int Ls = send - sstart;

            int Ns = 0, Ls_chk = 0;
            std::vector<T> Xs_raw;
            read_block_standardized<T>(bed_path, fam_path, sstart, send, rows, ddof, Xs_raw, Ns, Ls_chk);
            if (Ns != N || Ls_chk != Ls) {
                throw std::runtime_error("Unexpected source block dimensions in exact-local");
            }

            std::vector<T> Wex;
            prepare_dense_Wex_block<T>(
                Ls, B,
                annp + (size_t)sstart * (size_t)B,
                invp + (size_t)sstart,
                Wex
            );

            T* Cross = CrossBuf.ptr;
            // Cross = Xs_raw^T Xt_left : (Ls x Lt)
            gemm_col_major_tn<T>(
                /*m=*/Ls, /*n=*/Lt, /*k=*/N,
                /*A=*/Xs_raw.data(),  /*lda=*/N,
                /*B=*/Xt_left.data(), /*ldb=*/N,
                /*C=*/Cross,          /*ldc=*/Ls,
                /*alpha=*/T(1), /*beta=*/T(0)
            );

            const int64_t s_bp0 = blk_bp0[(size_t)bi];
            const int64_t s_bp1 = blk_bp1[(size_t)bi];

            if (full_lo <= bi && bi <= full_hi) {
                // square in place
                for (int col = 0; col < Lt; ++col) {
                    T* ccol = Cross + (size_t)col * (size_t)Ls;
                    #pragma omp simd
                    for (int r = 0; r < Ls; ++r) {
                        const T x = ccol[(size_t)r];
                        ccol[(size_t)r] = x * x;
                    }
                }
                gemm_col_major_tn<T>(
                    /*m=*/Lt, /*n=*/B, /*k=*/Ls,
                    /*A=*/Cross,      /*lda=*/Ls,
                    /*B=*/Wex.data(), /*ldb=*/Ls,
                    /*C=*/Ecol.data(),/*ldc=*/Lt,
                    /*alpha=*/T(1), /*beta=*/T(1)
                );
                continue;
            }

            if (bi == ti && !self_full) {
                accum_partial_exact_block<T>(
                    PartialMode::SELF,
                    Cross, Ls, Lt, Wex, B,
                    bp_t, bp_t, window_bp,
                    Ecol.data(), Lt
                );
                continue;
            }

            if (bi < ti) {
                accum_partial_exact_block<T>(
                    PartialMode::LEFT,
                    Cross, Ls, Lt, Wex, B,
                    bpp + sstart, bp_t, window_bp,
                    Ecol.data(), Lt
                );
                continue;
            }

            if (bi > ti) {
                accum_partial_exact_block<T>(
                    PartialMode::RIGHT,
                    Cross, Ls, Lt, Wex, B,
                    bpp + sstart, bp_t, window_bp,
                    Ecol.data(), Lt
                );
                continue;
            }
        }

        // Finalize addition onto pre-divide scale:
        // meansq_accum += exact_scale * inv_i^2 / D^2 * exact_local
        for (int i = 0; i < Lt; ++i) {
            const double invi = (double)invp[(size_t)tstart + (size_t)i];
            if (!std::isfinite(invi) || invi <= 0.0) continue;
            const double left_scale = invi * invi * inv_denom2;

            T* out = Mptr + ((size_t)(tstart + i) * (size_t)B);
            for (int k = 0; k < B; ++k) {
                const double upd = exact_scale * left_scale * (double)Ecol[(size_t)k * (size_t)Lt + (size_t)i];
                if (!std::isfinite(upd)) continue;
                if constexpr (std::is_same_v<T,float>) {
                    const double fmax = (double)std::numeric_limits<float>::max();
                    if (std::fabs(upd) > fmax) continue;
                }
                out[(size_t)k] += (T)upd;
            }
        }
    }
}

void set_num_threads(int n) {
    if (n > 0) {
        omp_set_num_threads(n);
    }
}

// ------------------------------- PyBind module -------------------------------
PYBIND11_MODULE(gwldcore, m) {
    m.doc() = "C++ core for SUMMIT GW LD score (bed parser + BLAS-safe GEMMs)";

    m.def("set_verbose", &set_verbose, py::arg("enabled"),
      "Enable/disable verbose timing prints");

    m.def("set_num_threads", &set_num_threads, py::arg("n"),
      "Set the number of OpenMP threads used inside gwldcore.");
    
    m.def("prefetch_bed_block",
      &prefetch_bed_block_py,
      py::arg("bed_prefix"),
      py::arg("fam_path"),
      py::arg("blk_start"),
      py::arg("blk_end"),
      py::arg("ahead_blocks") = 1,
      "Prefetch .bed bytes for [blk_start, blk_end) and optionally ahead blocks (Linux mmap+madvise).");

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
        py::arg("kmax_hint"),
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
        py::arg("kmax_hint"),
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

    m.def("phase2_compute_local_rp_sliding_bed",
        &phase2_compute_local_rp_sliding_bed_impl<float>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("block_starts"),
        py::arg("block_ends"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("inv_all"),
        py::arg("annot_all"),
        py::arg("bp_all"),
        py::arg("window_bp"),
        py::arg("v_start"),
        py::arg("v_count"),
        py::arg("rand_dist") = "rademacher",
        py::arg("seed") = py::none(),
        py::arg("meansq_accum"),
        py::arg("C") = py::none(),
        py::arg("R") = py::none(),
        py::arg("N_denom") = 0);

    m.def("phase2_compute_local_rp_sliding_bed",
        &phase2_compute_local_rp_sliding_bed_impl<double>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("block_starts"),
        py::arg("block_ends"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("inv_all"),
        py::arg("annot_all"),
        py::arg("bp_all"),
        py::arg("window_bp"),
        py::arg("v_start"),
        py::arg("v_count"),
        py::arg("rand_dist") = "rademacher",
        py::arg("seed") = py::none(),
        py::arg("meansq_accum"),
        py::arg("C") = py::none(),
        py::arg("R") = py::none(),
        py::arg("N_denom") = 0);

    m.def("phase2_compute_local_exact_bed",
        &phase2_compute_local_exact_bed_impl<float>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("block_starts"),
        py::arg("block_ends"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("inv_all"),
        py::arg("annot_all"),
        py::arg("bp_all"),
        py::arg("window_bp"),
        py::arg("meansq_accum"),
        py::arg("exact_scale"),
        py::arg("C") = py::none(),
        py::arg("R") = py::none(),
        py::arg("N_denom") = 0);

    m.def("phase2_compute_local_exact_bed",
        &phase2_compute_local_exact_bed_impl<double>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("block_starts"),
        py::arg("block_ends"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("inv_all"),
        py::arg("annot_all"),
        py::arg("bp_all"),
        py::arg("window_bp"),
        py::arg("meansq_accum"),
        py::arg("exact_scale"),
        py::arg("C") = py::none(),
        py::arg("R") = py::none(),
        py::arg("N_denom") = 0);
}
