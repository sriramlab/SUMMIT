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
    C.rows.shrink_to_fit(); // optional; you can remove if you prefer keeping capacity

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

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

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

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < L; ++i) {
        const T s = inv[i] / denom;
        T* col = Geno.data() + (size_t)i * (size_t)N;
        #pragma omp simd
        for (int r = 0; r < N; ++r) col[r] *= s;
    }

    // Tile planning over Q = B*V (columns contiguous in K-major)
    const int Q = BV;
    int QPANEL = 4096;
    if (const char* qp = std::getenv("SUMMIT_P2_QP")) {
        int val = std::atoi(qp);
        if (val > 0) QPANEL = val;
    }
    if (QPANEL > Q) QPANEL = Q;
    QPANEL = ((QPANEL + 63) / 64) * 64;
    if (QPANEL > Q) QPANEL = Q;
    if (QPANEL < 64) QPANEL = std::min(Q, 64);

    // Workspace for one Q panel: (L x QPANEL), col-major; thread-local version
    static thread_local AlignedBuffer<T> Work_panel_tls;
    const size_t needW = (size_t)L * (size_t)QPANEL;
    if (Work_panel_tls.n < needW) {
        Work_panel_tls.allocate(needW, 64);
    }
    T* Work = Work_panel_tls.ptr;


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
                             /*C=*/Work, /*ldc=*/L,
                             /*alpha=*/T(1), /*beta=*/T(0));
        auto t3 = std::chrono::high_resolution_clock::now();
        t.add_gemm(std::chrono::duration<double,std::milli>(t3 - t2).count());

        // Reduce: meansq[blk_start:blk_end, k] += (wcol^2)/V
        auto t4 = std::chrono::high_resolution_clock::now();
        // Reduce: meansq[blk_start:blk_end, k] += (sum over v in this panel segment of w^2)/V
        // K-major implies columns are ordered by k then v; within a q-panel, each bin k appears
        // as contiguous segments of length <= nvecs.
        const int k0 = q0 / nvecs;
        const int k1 = (q0 + q - 1) / nvecs;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int k = k0; k <= k1; ++k) {
        for (int i = 0; i < L; ++i) {
            const int g0 = std::max(q0, k * nvecs);
            const int g1 = std::min(q0 + q, (k + 1) * nvecs);
            const int seg_len = g1 - g0;
            const int tcol0 = g0 - q0;

            long double acc = 0.0L;
            const T* base = Work + (size_t)tcol0 * (size_t)L + (size_t)i;
            for (int c = 0; c < seg_len; ++c) {
            T w = base[(size_t)c * (size_t)L];
            acc += (long double)w * (long double)w;
            }

            T* out = Mptr + ((size_t)blk_start * (size_t)B + (size_t)k);
            out[(size_t)i * (size_t)B] += (T)(acc * (long double)invV);
        }
        }


        auto t5 = std::chrono::high_resolution_clock::now();
        t.add_reduce(std::chrono::duration<double,std::milli>(t5 - t4).count());
    }

    t.dump(blk_start, blk_end, B, nvecs);
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
}
