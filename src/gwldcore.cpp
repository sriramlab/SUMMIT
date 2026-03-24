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

struct Phase1CSRKey {
    const void* ann_ptr = nullptr;
    const void* inv_ptr = nullptr;
    int L = 0;
    int B = 0;
    uint8_t itemsize = 0;

    bool operator==(const Phase1CSRKey& o) const {
        return ann_ptr == o.ann_ptr &&
               inv_ptr == o.inv_ptr &&
               L == o.L &&
               B == o.B &&
               itemsize == o.itemsize;
    }
};

struct Phase1CSRKeyHash {
    std::size_t operator()(const Phase1CSRKey& k) const noexcept {
        std::size_t h = std::hash<const void*>{}(k.ann_ptr);
        h ^= std::hash<const void*>{}(k.inv_ptr) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.L) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.B) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}((int)k.itemsize) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

template <typename T>
struct Phase1CSRData {
    std::vector<int> colptr;
    AlignedBuffer<int> rowind;
    AlignedBuffer<T> scale;
    std::size_t bytes = 0;
    uint64_t age = 0;
};

template <typename T>
struct Phase1CSRStore {
    std::unordered_map<Phase1CSRKey,
                       std::shared_ptr<Phase1CSRData<T>>,
                       Phase1CSRKeyHash> map;
    std::size_t bytes = 0;
    uint64_t clock = 0;
};

template <typename T>
static Phase1CSRStore<T>& phase1_csr_store() {
    static thread_local Phase1CSRStore<T> store;
    return store;
}

static std::size_t phase1_csr_cache_cap_bytes() {
    const char* s = std::getenv("SUMMIT_P1_CSR_CACHE_MB");
    if (!s || !*s) return 256ULL * 1024ULL * 1024ULL;  // default 256 MB
    long long mb = std::atoll(s);
    if (mb <= 0) return 0;
    return (std::size_t)mb * 1024ULL * 1024ULL;
}

template <typename T>
static std::shared_ptr<Phase1CSRData<T>>
build_phase1_csr(const T* ann, const T* inv, int L, int B)
{
    auto out = std::make_shared<Phase1CSRData<T>>();
    out->colptr.assign((size_t)B + 1, 0);

    for (int k = 0; k < B; ++k) {
        int cnt = 0;
        const T* colk = ann + (size_t)k; // row-major view: a[i*B + k]
        for (int i = 0; i < L; ++i) {
            const T a = colk[(size_t)i * (size_t)B];
            if (a == T(0)) continue;

            const double ad = (double)a;
            if (!(ad > 0.0) || !std::isfinite(ad)) continue;

            const double invd = (double)inv[(size_t)i];
            if (!std::isfinite(invd) || invd <= 0.0) continue;

            const double sc = invd * std::sqrt(ad);
            if (!std::isfinite(sc)) continue;

            if constexpr (std::is_same_v<T, float>) {
                if (sc > (double)std::numeric_limits<float>::max()) continue;
            }

            ++cnt;
        }
        out->colptr[(size_t)k + 1] = cnt;
    }

    for (int k = 0; k < B; ++k) {
        out->colptr[(size_t)k + 1] += out->colptr[(size_t)k];
    }

    const int nnz = out->colptr[(size_t)B];
    out->rowind.allocate((size_t)nnz, 64);
    out->scale.allocate((size_t)nnz, 64);

    std::vector<int> fill((size_t)B, 0);
    for (int k = 0; k < B; ++k) {
        const int base = out->colptr[(size_t)k];
        int& f = fill[(size_t)k];
        const T* colk = ann + (size_t)k;

        for (int i = 0; i < L; ++i) {
            const T a = colk[(size_t)i * (size_t)B];
            if (a == T(0)) continue;

            const double ad = (double)a;
            if (!(ad > 0.0) || !std::isfinite(ad)) continue;

            const double invd = (double)inv[(size_t)i];
            if (!std::isfinite(invd) || invd <= 0.0) continue;

            const double sc = invd * std::sqrt(ad);
            if (!std::isfinite(sc)) continue;

            if constexpr (std::is_same_v<T, float>) {
                if (sc > (double)std::numeric_limits<float>::max()) continue;
            }

            const int pos = base + f++;
            out->rowind.ptr[(size_t)pos] = i;
            out->scale.ptr[(size_t)pos] = (T)sc;
        }
    }

    out->bytes =
        out->colptr.size() * sizeof(int) +
        (std::size_t)nnz * (sizeof(int) + sizeof(T));

    return out;
}

template <typename T>
static std::shared_ptr<Phase1CSRData<T>>
get_or_build_phase1_csr(const T* ann, const T* inv, int L, int B)
{
    const std::size_t cap = phase1_csr_cache_cap_bytes();

    Phase1CSRKey key;
    key.ann_ptr = ann;
    key.inv_ptr = inv;
    key.L = L;
    key.B = B;
    key.itemsize = (uint8_t)sizeof(T);

    if (cap > 0) {
        auto& store = phase1_csr_store<T>();
        auto it = store.map.find(key);
        if (it != store.map.end()) {
            it->second->age = ++store.clock;
            return it->second;
        }

        auto built = build_phase1_csr<T>(ann, inv, L, B);
        built->age = ++store.clock;

        if (built->bytes > cap) {
            return built; // too large to cache, but still use this instance
        }

        while (store.bytes + built->bytes > cap && !store.map.empty()) {
            auto victim = store.map.begin();
            for (auto it2 = store.map.begin(); it2 != store.map.end(); ++it2) {
                if (it2->second->age < victim->second->age) victim = it2;
            }
            store.bytes -= victim->second->bytes;
            store.map.erase(victim);
        }

        store.bytes += built->bytes;
        store.map.emplace(key, built);
        return built;
    }

    return build_phase1_csr<T>(ann, inv, L, B);
}

static void clear_phase1_csr_cache() {
    {
        auto& s = phase1_csr_store<float>();
        s.map.clear();
        s.bytes = 0;
        s.clock = 0;
    }
    {
        auto& s = phase1_csr_store<double>();
        s.map.clear();
        s.bytes = 0;
        s.clock = 0;
    }
}

template <typename T>
static void project_panel_inplace(T* X, int N, int Q,
                                  const T* Cptr, const T* Rptr, int p,
                                  AlignedBuffer<T>& tmp_buf)
{
    if (p <= 0 || N <= 0 || Q <= 0) return;

    const size_t need = (size_t)p * (size_t)Q;
    if (tmp_buf.n < need) {
        tmp_buf.allocate(need, 64);
    }
    T* tmp = tmp_buf.ptr;

    // tmp = R * X   (p x Q)
    gemm_col_major_nn<T>(p, Q, N,
                         Rptr, p,
                         X,    N,
                         tmp,  p,
                         T(1), T(0));

    // X = X - C * tmp   (N x Q)
    gemm_col_major_nn<T>(N, Q, p,
                         Cptr, N,
                         tmp,  p,
                         X,    N,
                         T(-1), T(1));
}

template <typename T>
void apply_grm_bed_panel_impl(
    const std::string& bed_prefix,
    const std::string& fam_path,
    int nsnps,
    int step_size,
    py::object row_sel_obj,
    int ddof,
    py::array_t<T, py::array::c_style | py::array::forcecast> inv_all,     // (M,)
    py::array_t<T, py::array::f_style | py::array::forcecast> panel_in,    // (N x Q), sample-space probes
    py::array_t<T, py::array::f_style | py::array::forcecast> panel_out,   // (N x Q), output K * panel_in
    py::object C_opt,
    py::object R_opt)
{
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);

    if (nsnps < 0 || nsnps > (int)M_total) {
        throw std::runtime_error("nsnps is out of range in apply_grm_bed_panel");
    }
    if (step_size <= 0) {
        throw std::runtime_error("step_size must be > 0 in apply_grm_bed_panel");
    }

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    const int N_rows = (int)rows.size();
    if (N_rows <= 0) {
        throw std::runtime_error("No rows selected in apply_grm_bed_panel");
    }

    auto Iinfo = inv_all.request();
    if (Iinfo.ndim != 1 || (int)Iinfo.shape[0] < nsnps) {
        throw std::runtime_error("inv_all shape mismatch in apply_grm_bed_panel");
    }
    const T* invp = static_cast<const T*>(Iinfo.ptr);

    auto Xin = panel_in.request();
    auto Xout = panel_out.request();

    if (Xin.ndim != 2 || Xout.ndim != 2) {
        throw std::runtime_error("panel_in/panel_out must be 2D in apply_grm_bed_panel");
    }
    if ((int)Xin.shape[0] != N_rows || (int)Xout.shape[0] != N_rows) {
        throw std::runtime_error("panel_in/panel_out row dimension mismatch in apply_grm_bed_panel");
    }
    if ((int)Xin.shape[1] != (int)Xout.shape[1]) {
        throw std::runtime_error("panel_in/panel_out column dimension mismatch in apply_grm_bed_panel");
    }

    const int Q = (int)Xin.shape[1];
    const T* inptr = static_cast<const T*>(Xin.ptr);
    T* outptr = static_cast<T*>(Xout.ptr);

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;

    py::array_t<T, py::array::f_style | py::array::forcecast> Carr;
    py::array_t<T, py::array::f_style | py::array::forcecast> Rarr;

    if (have_proj) {
        Carr = C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        Rarr = R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();

        auto Cinfo = Carr.request();
        auto Rinfo = Rarr.request();

        p = (int)Cinfo.shape[1];
        if ((int)Cinfo.shape[0] != N_rows ||
            (int)Rinfo.shape[0] != p ||
            (int)Rinfo.shape[1] != N_rows) {
            throw std::runtime_error("C/R shape mismatch in apply_grm_bed_panel");
        }

        Cptr = static_cast<const T*>(Cinfo.ptr);
        Rptr = static_cast<const T*>(Rinfo.ptr);
    }

    AlignedBuffer<T> Vin((size_t)N_rows * (size_t)Q, 64);
    std::memcpy(Vin.ptr, inptr, sizeof(T) * (size_t)N_rows * (size_t)Q);

    std::fill(outptr, outptr + (size_t)N_rows * (size_t)Q, T(0));

    AlignedBuffer<T> proj_tmp(have_proj ? (size_t)p * (size_t)Q : 0, 64);
    AlignedBuffer<T> coef_buf((size_t)std::min(step_size, nsnps) * (size_t)Q, 64);

    py::gil_scoped_release nogil;

    // Right-side projection: because the BED-standardized SNPs are already mean-zero,
    // only the explicit covariate projection is needed here.
    if (have_proj) {
        project_panel_inplace<T>(Vin.ptr, N_rows, Q, Cptr, Rptr, p, proj_tmp);
    }

    const double invM = 1.0 / (double)nsnps;

    for (int s = 0; s < nsnps; s += step_size) {
        check_for_interrupt();

        const int e = std::min(nsnps, s + step_size);
        const int L = e - s;
        if (L <= 0) continue;

        int N_blk = 0, L_blk = 0;
        std::vector<T> Geno;
        read_block_standardized<T>(bed_path, fam_path, s, e, rows, ddof, Geno, N_blk, L_blk);

        if (N_blk != N_rows || L_blk != L) {
            throw std::runtime_error("Unexpected block dimensions in apply_grm_bed_panel");
        }

        T* coef = coef_buf.ptr;

        // coef = G^T * Vin   (L x Q)
        gemm_col_major_tn<T>(L, Q, N_rows,
                             Geno.data(), N_rows,
                             Vin.ptr,     N_rows,
                             coef,        L,
                             T(1), T(0));

        // Scale each SNP-row by inv_j^2 / M.
        std::vector<T> row_scale((size_t)L, T(0));
        for (int j = 0; j < L; ++j) {
            const double invj = (double)invp[(size_t)(s + j)];
            double w = invj * invj * invM;
            if (!std::isfinite(w) || w <= 0.0) w = 0.0;
            row_scale[(size_t)j] = (T)w;
        }

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int q = 0; q < Q; ++q) {
            T* ccol = coef + (size_t)q * (size_t)L;
            for (int j = 0; j < L; ++j) {
                ccol[(size_t)j] *= row_scale[(size_t)j];
            }
        }

        // out += G * coef   (N x Q)
        gemm_col_major_nn<T>(N_rows, Q, L,
                             Geno.data(), N_rows,
                             coef,        L,
                             outptr,      N_rows,
                             T(1), T(1));
    }

    // Left-side projection to obtain K * v = (1/M) P G D^2 G^T P v.
    if (have_proj) {
        project_panel_inplace<T>(outptr, N_rows, Q, Cptr, Rptr, p, proj_tmp);
    }
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
                                      int v_start,
                                      int v_count,
                                      int /*kmax_hint*/,
                                      const std::string &rand_dist,
                                      py::object seed_obj,
                                      py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d_chunk, // (N x (B*v_count)), Fortran
                                      bool project_right = false,
                                      py::object C_opt = py::none(),
                                      py::object R_opt = py::none())
{
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    std::vector<T> Geno;
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

        AlignedBuffer<T> tmpG((size_t)p * (size_t)L, 64);
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno.data(), N, tmpG.ptr, p, T(1), T(0));
        gemm_col_major_nn<T>(N, L, p, Cptr, N, tmpG.ptr, p, Geno.data(), N, T(-1), T(1));
    }

    auto Ainfo = annot_blk.request();
    auto Iinfo = inv_right.request();
    if (Ainfo.ndim != 2 || Iinfo.ndim != 1) throw std::runtime_error("annot/inv shapes");
    const int B = (int)Ainfo.shape[1];
    if ((int)Ainfo.shape[0] != L || (int)Iinfo.shape[0] != L) throw std::runtime_error("L mismatch");

    const T* ann = static_cast<const T*>(Ainfo.ptr);
    const T* inv = static_cast<const T*>(Iinfo.ptr);

    auto Xinfo = Xz2d_chunk.request();
    if (Xinfo.ndim != 2 || (int)Xinfo.shape[0] != N || (int)Xinfo.shape[1] != B * v_count)
        throw std::runtime_error("Xz2d_chunk shape must be (N, B*v_count)");
    T* Xptr = static_cast<T*>(Xinfo.ptr);
    const int ldc = N;

    const bool have_root = !seed_obj.is_none();
    const uint64_t root_seed = have_root ? seed_obj.cast<uint64_t>() : std::random_device{}();
    std::mt19937_64 rng(make_seed(root_seed, blk_start, v_start));
    std::normal_distribution<T> gN(0, (T)1);
    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_spherical  = (rand_dist == "spherical");

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
            for (int r = 0; r < L; ++r) {
                Z[(size_t)r + (size_t)c * (size_t)L] *= scale;
            }
        }
    }

    auto csr = get_or_build_phase1_csr<T>(ann, inv, L, B);
    const std::vector<int>& colptr = csr->colptr;
    const int* rowind = csr->rowind.ptr;
    const T* scale = csr->scale.ptr;

    if (blk_start == 0) {
        const size_t Q = (size_t)B * (size_t)v_count;
        const size_t elems_per_page = (size_t)((4096 / sizeof(T)) ? (4096 / sizeof(T)) : 512);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t g = 0; g < (ptrdiff_t)Q; ++g) {
            T* col = Xptr + (size_t)g * (size_t)ldc;
            for (size_t r = 0; r < (size_t)N; r += elems_per_page) {
                col[r] += T(0);
            }
        }
    }

    const char* s = std::getenv("SUMMIT_P1_NTILE");
    int NTILE = s ? std::max(64, std::atoi(s)) : (std::is_same_v<T,double> ? 256 : 512);
    P1Timers t1;

    for (int k = 0; k < B; ++k) {
        check_for_interrupt();

        const int k0 = colptr[(size_t)k];
        const int k1 = colptr[(size_t)k + 1];
        const int K  = k1 - k0;
        if (K == 0) continue;

        AlignedBuffer<T> Bcol((size_t)K * (size_t)v_count, 64);

        auto z0 = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < v_count; ++c) {
            const T* zc = Z.data() + (size_t)c * (size_t)L;
            T* dst = Bcol.ptr + (size_t)c * (size_t)K;
            for (int r = 0; r < K; ++r) {
                const int snp = rowind[(size_t)k0 + (size_t)r];
                dst[(size_t)r] = zc[(size_t)snp];
            }
        }
        auto z1 = std::chrono::high_resolution_clock::now();
        t1.t_packZ_ms += std::chrono::duration<double, std::milli>(z1 - z0).count();

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            AlignedBuffer<T> A_tile((size_t)NTILE * (size_t)K, 64);
            AlignedBuffer<T> C_tile((size_t)NTILE * (size_t)v_count, 64);

            double packA_ms = 0.0, gemm_ms = 0.0, scatt_ms = 0.0;

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int n0 = 0; n0 < N; n0 += NTILE) {
                const int Nt = std::min(N - n0, NTILE);

                auto a0 = std::chrono::high_resolution_clock::now();
                for (int c = 0; c < K; ++c) {
                    const int snp = rowind[(size_t)k0 + (size_t)c];
                    const T ssc = scale[(size_t)k0 + (size_t)c];
                    const T* src = Geno.data() + (size_t)snp * (size_t)N + (size_t)n0;
                    T* dst = A_tile.ptr + (size_t)c * (size_t)Nt;
#pragma omp simd
                    for (int r = 0; r < Nt; ++r) {
                        dst[r] = src[r] * ssc;
                    }
                }
                auto a1 = std::chrono::high_resolution_clock::now();
                packA_ms += std::chrono::duration<double, std::milli>(a1 - a0).count();

                auto g0 = std::chrono::high_resolution_clock::now();
                gemm_col_major_nn<T>(
                    Nt, v_count, K,
                    A_tile.ptr, Nt,
                    Bcol.ptr, K,
                    C_tile.ptr, Nt,
                    T(1), T(0)
                );
                auto g1 = std::chrono::high_resolution_clock::now();
                gemm_ms += std::chrono::duration<double, std::milli>(g1 - g0).count();

                auto s0 = std::chrono::high_resolution_clock::now();
                const size_t base_col = (size_t)k * (size_t)v_count;
                for (int c = 0; c < v_count; ++c) {
                    const T* __restrict src_col = C_tile.ptr + (size_t)c * (size_t)Nt;
                    T* __restrict dst_col = Xptr + ((base_col + (size_t)c) * (size_t)ldc) + (size_t)n0;
                    cblas_taxpy(Nt, T(1), src_col, 1, dst_col, 1);
                }
                auto s1 = std::chrono::high_resolution_clock::now();
                scatt_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
            }

#ifdef _OPENMP
            #pragma omp atomic
#endif
            t1.t_packA_ms += packA_ms;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            t1.t_gemm_ms += gemm_ms;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            t1.t_scatt_ms += scatt_ms;
        }
    }

    t1.dump(blk_start, blk_end, B, v_count);
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

template <typename T>
void phase2_accum_XtXz_bed_impl(const std::string &bed_prefix,
                                const std::string &fam_path,
                                int blk_start, int blk_end,
                                py::object row_sel_obj,
                                int ddof,
                                py::array_t<T, py::array::c_style | py::array::forcecast> inv_left, // (L,)
                                int tile_nvecs,
                                py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d,      // (N x (B*Vtile)), K-major
                                py::array_t<T, py::array::c_style | py::array::forcecast> meansq_accum, // (M x B), row-major
                                py::object C_opt,
                                py::object R_opt,
                                int N_denom)
{
    auto round_down = [](int x, int m) { return (m > 0) ? (x / m) * m : x; };

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    std::vector<T> Geno;
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    if (!C_opt.is_none() && !R_opt.is_none()) {
        py::array_t<T, py::array::f_style | py::array::forcecast> C =
            C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        py::array_t<T, py::array::f_style | py::array::forcecast> R =
            R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Ci = C.request();
        auto Ri = R.request();
        const int p = (int)Ci.shape[1];
        if ((int)Ci.shape[0] != N || (int)Ri.shape[0] != p || (int)Ri.shape[1] != N)
            throw std::runtime_error("C/R shape mismatch in phase2_accum_XtXz_bed");

        const T* Cptr = static_cast<const T*>(Ci.ptr);
        const T* Rptr = static_cast<const T*>(Ri.ptr);

        AlignedBuffer<T> tmpG((size_t)p * (size_t)L, 64);
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno.data(), N, tmpG.ptr, p, T(1), T(0));
        gemm_col_major_nn<T>(N, L, p, Cptr, N, tmpG.ptr, p, Geno.data(), N, T(-1), T(1));
    }

    auto Ii = inv_left.request();
    if (Ii.ndim != 1 || (int)Ii.shape[0] != L)
        throw std::runtime_error("inv_left shape mismatch in phase2_accum_XtXz_bed");
    const T* inv = static_cast<const T*>(Ii.ptr);

    auto Xi = Xz2d.request();
    if (Xi.ndim != 2 || (int)Xi.shape[0] != N)
        throw std::runtime_error("Xz2d shape mismatch in phase2_accum_XtXz_bed");
    T* Xptr = static_cast<T*>(Xi.ptr);

    const int Q = (int)Xi.shape[1];
    if (tile_nvecs <= 0 || (Q % tile_nvecs) != 0)
        throw std::runtime_error("Xz2d col count must be a multiple of tile_nvecs");
    const int B = Q / tile_nvecs;

    auto Mi = meansq_accum.request();
    if (Mi.ndim != 2 || (int)Mi.shape[1] != B)
        throw std::runtime_error("meansq_accum shape mismatch in phase2_accum_XtXz_bed");
    if (blk_end > (int)Mi.shape[0])
        throw std::runtime_error("meansq_accum rows smaller than SNP count");
    T* Mptr = static_cast<T*>(Mi.ptr);

    double denom = (double)N_denom - 1.0;
    if (denom <= 0.0) denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    std::vector<double> left_scale((size_t)L, 0.0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < L; ++i) {
        const double inv_i = (double)inv[(size_t)i];
        if (std::isfinite(inv_i) && inv_i > 0.0) {
            left_scale[(size_t)i] = inv_i * inv_i * inv_denom2;
        }
    }

    int QPANEL = 32768;
    if (const char* s = std::getenv("SUMMIT_P2_QPANEL")) {
        QPANEL = std::atoi(s);
    }
    if (QPANEL <= 0) QPANEL = Q;
    QPANEL = std::min(QPANEL, Q);
    QPANEL = round_down(QPANEL, 64);
    if (QPANEL < 64) QPANEL = std::min(Q, 64);

    static thread_local AlignedBuffer<T> Work_panel_tls;
    const size_t needW = (size_t)L * (size_t)QPANEL;
    if (Work_panel_tls.n < needW) Work_panel_tls.allocate(needW, 64);
    T* Work = Work_panel_tls.ptr;

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
#ifdef _OPENMP
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = omp_get_max_threads();
#else
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = 1;
#endif

    const int nslabs = ceil_div_i(L, IBLK);
    REDUCE_THREADS = clampi(REDUCE_THREADS, 1, std::max(1, nslabs));

    BlockTimers t;
    py::gil_scoped_release nogil;

    for (int q0 = 0; q0 < Q; q0 += QPANEL) {
        check_for_interrupt();

        const int q = std::min(QPANEL, Q - q0);
        const T* rhs = Xptr + (size_t)q0 * (size_t)N;

        auto t2 = std::chrono::high_resolution_clock::now();
        gemm_col_major_tn<T>(L, q, N,
                             Geno.data(), N,
                             rhs,        N,
                             Work,       L,
                             T(1), T(0));
        auto t3 = std::chrono::high_resolution_clock::now();
        t.add_gemm(std::chrono::duration<double, std::milli>(t3 - t2).count());

        std::vector<int> seg_k;
        std::vector<int> seg_tcol0;
        std::vector<int> seg_len;
        seg_k.reserve((size_t)B + 1);
        seg_tcol0.reserve((size_t)B + 1);
        seg_len.reserve((size_t)B + 1);

        {
            int g = q0;
            const int g_end = q0 + q;
            while (g < g_end) {
                const int k = g / tile_nvecs;
                const int v_in = g - k * tile_nvecs;
                const int len = std::min(g_end - g, tile_nvecs - v_in);
                seg_k.push_back(k);
                seg_tcol0.push_back(g - q0);
                seg_len.push_back(len);
                g += len;
            }
        }

        auto t4 = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(REDUCE_THREADS)
#endif
        for (int i0 = 0; i0 < L; i0 += IBLK) {
            const int ib = std::min(IBLK, L - i0);

            static thread_local AlignedBuffer<double> acc_tls;
            if (acc_tls.n < (size_t)IBLK) acc_tls.allocate((size_t)IBLK, 64);
            double* acc = acc_tls.ptr;

            for (std::size_t sidx = 0; sidx < seg_k.size(); ++sidx) {
                const int k    = seg_k[sidx];
                const int tcol = seg_tcol0[sidx];
                const int len  = seg_len[sidx];

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

                T* __restrict out =
                    Mptr + ((size_t)(blk_start + i0) * (size_t)B + (size_t)k);

#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int ii = 0; ii < ib; ++ii) {
                    double upd = acc[ii] * left_scale[(size_t)(i0 + ii)];

                    if (!std::isfinite(upd)) continue;
                    if constexpr (std::is_same_v<T, float>) {
                        const double fmax = (double)std::numeric_limits<float>::max();
                        if (upd > fmax) continue;
                    }
                    if (upd < 0.0) upd = 0.0;

                    out[(size_t)ii * (size_t)B] += (T)upd;
                }
            }
        }

        auto t5 = std::chrono::high_resolution_clock::now();
        t.add_reduce(std::chrono::duration<double, std::milli>(t5 - t4).count());
    }

    t.dump(blk_start, blk_end, B, tile_nvecs);
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
static void compute_raw_sd_block_from_bed(
    std::ifstream& fin,
    int64_t N_total,
    const std::vector<int>& row_byte,
    const std::vector<unsigned char>& row_shift,
    int blk_start,
    int blk_end,
    int ddof,
    T* out_raw_sd,
    std::vector<unsigned char>& io_buf
) {
    const int L = blk_end - blk_start;
    if (L <= 0) return;

    const int bytes_per_snp = (int)((N_total + 3) >> 2);
    const size_t need = (size_t)L * (size_t)bytes_per_snp;
    io_buf.resize(need);

    fin.clear();
    fin.seekg(std::streamoff(3) + std::streamoff(blk_start) * (std::streamoff)bytes_per_snp, std::ios::beg);
    fin.read(reinterpret_cast<char*>(io_buf.data()), (std::streamsize)need);
    if (!fin) {
        throw std::runtime_error("Failed reading BED bytes for raw SD calculation");
    }

    const T qnan = std::numeric_limits<T>::quiet_NaN();

    for (int j = 0; j < L; ++j) {
        const unsigned char* snp = io_buf.data() + (size_t)j * (size_t)bytes_per_snp;

        double sum = 0.0;
        double sumsq = 0.0;
        int cnt = 0;

        for (size_t r = 0; r < row_byte.size(); ++r) {
            const unsigned char code =
                (unsigned char)((snp[(size_t)row_byte[r]] >> row_shift[r]) & 0x03u);

            double x = 0.0;
            switch (code) {
                case 0u: x = 0.0; break; // homozygous major
                case 2u: x = 1.0; break; // heterozygous
                case 3u: x = 2.0; break; // homozygous minor
                case 1u: default: continue; // missing
            }

            sum += x;
            sumsq += x * x;
            ++cnt;
        }

        if (cnt == 0 || cnt <= ddof) {
            out_raw_sd[(size_t)j] = qnan;
            continue;
        }

        double num = sumsq - (sum * sum) / (double)cnt;
        if (num < 0.0 && num > -1e-12) num = 0.0;

        if (num < 0.0 || !std::isfinite(num)) {
            out_raw_sd[(size_t)j] = qnan;
            continue;
        }

        out_raw_sd[(size_t)j] = (T)std::sqrt(num / (double)(cnt - ddof));
    }
}

template <typename T>
py::tuple precompute_residual_variances_bed_impl(
    const std::string& bed_prefix,
    const std::string& fam_path,
    int nsnps,
    int step_size,
    py::object row_sel_obj,
    int ddof,
    double eps,
    bool compute_mu22,
    py::object annot_all_obj,
    py::object C_opt,
    py::object R_opt)
{
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);

    if (nsnps < 0 || nsnps > (int)M_total) {
        throw std::runtime_error("nsnps is out of range in precompute_residual_variances_bed");
    }
    if (step_size <= 0) {
        throw std::runtime_error("step_size must be > 0 in precompute_residual_variances_bed");
    }

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    const int N_rows = (int)rows.size();
    if (N_rows <= 1) {
        throw std::runtime_error("Too few rows in precompute_residual_variances_bed");
    }

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
        if ((int)Cinfo.shape[0] != N_rows ||
            (int)Rinfo.shape[0] != p ||
            (int)Rinfo.shape[1] != N_rows) {
            throw std::runtime_error("C/R shape mismatch in precompute_residual_variances_bed");
        }
        if (N_rows - p <= 1) {
            throw std::runtime_error("N_eff must be > 1 in precompute_residual_variances_bed");
        }

        Cptr = static_cast<const T*>(Cinfo.ptr);
        Rptr = static_cast<const T*>(Rinfo.ptr);
    }

    const double proj_df = have_proj ? (double)(N_rows - p - 1) : (double)(N_rows - 1);
    if (proj_df <= 0.0) {
        throw std::runtime_error("Projected residual df must be > 0 in precompute_residual_variances_bed");
    }

    py::array_t<T> inv_out(nsnps);
    auto Iinfo = inv_out.request();
    T* invp = static_cast<T*>(Iinfo.ptr);

    int B = 0;
    py::array_t<T, py::array::c_style | py::array::forcecast> Aarr;
    const T* annp = nullptr;

    AlignedBuffer<double> S_mat;
    AlignedBuffer<double> X2_panel;
    AlignedBuffer<double> W_panel;
    AlignedBuffer<double> S_sum;
    std::vector<double> Msum;

    int JPANEL = 32;
    if (const char* s = std::getenv("SUMMIT_RESVAR_MU22_JPANEL")) {
        const int v = std::atoi(s);
        if (v > 0) JPANEL = v;
    }

    if (compute_mu22) {
        if (annot_all_obj.is_none()) {
            throw std::runtime_error("annot_all must be provided when compute_mu22=True");
        }

        Aarr = annot_all_obj.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
        auto Ainfo = Aarr.request();
        if (Ainfo.ndim != 2 || (int)Ainfo.shape[0] != nsnps) {
            throw std::runtime_error("annot_all shape mismatch in precompute_residual_variances_bed");
        }

        annp = static_cast<const T*>(Ainfo.ptr);
        B = (int)Ainfo.shape[1];
        if (B <= 0) {
            throw std::runtime_error("annot_all must have at least one column");
        }

        S_mat.allocate((size_t)N_rows * (size_t)B, 64);
        std::fill(S_mat.ptr, S_mat.ptr + S_mat.n, 0.0);

        JPANEL = std::max(1, std::min(JPANEL, std::max(1, step_size)));
        X2_panel.allocate((size_t)N_rows * (size_t)JPANEL, 64);
        W_panel.allocate((size_t)JPANEL * (size_t)B, 64);
        S_sum.allocate((size_t)B * (size_t)B, 64);
        Msum.assign((size_t)B, 0.0);
    }

    AlignedBuffer<T> proj_tmp(have_proj ? (size_t)p * (size_t)std::min(step_size, nsnps) : 0, 64);

    {
        py::gil_scoped_release nogil;

        for (int s = 0; s < nsnps; s += step_size) {
            check_for_interrupt();

            const int e = std::min(nsnps, s + step_size);
            const int L = e - s;
            if (L <= 0) continue;

            int N_blk = 0, L_blk = 0;
            std::vector<T> Geno;
            read_block_standardized<T>(bed_path, fam_path, s, e, rows, ddof, Geno, N_blk, L_blk);

            if (N_blk != N_rows || L_blk != L) {
                throw std::runtime_error("Unexpected block dimensions in precompute_residual_variances_bed");
            }

            if (have_proj) {
                project_target_block_inplace<T>(Geno, N_blk, L_blk, Cptr, Rptr, p, proj_tmp);
            }

            std::vector<double> inv2_local((size_t)L, 0.0);

#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < L; ++j) {
                const T* col = Geno.data() + (size_t)j * (size_t)N_blk;

                double ss = 0.0;
                for (int i = 0; i < N_blk; ++i) {
                    const double x = (double)col[(size_t)i];
                    ss += x * x;
                }

                const double var = ss / proj_df;
                double safe = 0.0;
                if (std::isnan(var)) {
                    safe = std::numeric_limits<double>::quiet_NaN();
                } else {
                    safe = (var > eps ? var : eps);
                }

                const double invj = 1.0 / std::sqrt(safe);
                invp[(size_t)(s + j)] = (T)invj;
                inv2_local[(size_t)j] = invj * invj;
            }

            if (compute_mu22) {
                const T* ann_blk = annp + (size_t)s * (size_t)B;

                for (int j = 0; j < L; ++j) {
                    const T* arow = ann_blk + (size_t)j * (size_t)B;
                    for (int k = 0; k < B; ++k) {
                        Msum[(size_t)k] += (double)arow[(size_t)k];
                    }
                }

                for (int j0 = 0; j0 < L; j0 += JPANEL) {
                    const int jb = std::min(JPANEL, L - j0);

                    for (int jj = 0; jj < jb; ++jj) {
                        const int j = j0 + jj;
                        const T* gcol = Geno.data() + (size_t)j * (size_t)N_blk;
                        double* xcol = X2_panel.ptr + (size_t)jj * (size_t)N_blk;

                        for (int i = 0; i < N_blk; ++i) {
                            const double y = (double)gcol[(size_t)i];
                            xcol[(size_t)i] = y * y;
                        }
                    }

                    for (int k = 0; k < B; ++k) {
                        double* wcol = W_panel.ptr + (size_t)k * (size_t)jb;
                        for (int jj = 0; jj < jb; ++jj) {
                            const int j = j0 + jj;
                            const double a = (double)ann_blk[(size_t)j * (size_t)B + (size_t)k];
                            wcol[(size_t)jj] = a * inv2_local[(size_t)j];
                        }
                    }

                    gemm_col_major_nn<double>(
                        /*m=*/N_blk, /*n=*/B, /*k=*/jb,
                        /*A=*/X2_panel.ptr, /*lda=*/N_blk,
                        /*B=*/W_panel.ptr,  /*ldb=*/jb,
                        /*C=*/S_mat.ptr,    /*ldc=*/N_blk,
                        /*alpha=*/1.0, /*beta=*/1.0
                    );
                }
            }
        }

        if (compute_mu22) {
            gemm_col_major_tn<double>(
                /*m=*/B, /*n=*/B, /*k=*/N_rows,
                /*A=*/S_mat.ptr, /*lda=*/N_rows,
                /*B=*/S_mat.ptr, /*ldb=*/N_rows,
                /*C=*/S_sum.ptr, /*ldc=*/B,
                /*alpha=*/1.0, /*beta=*/0.0
            );
        }
    }

    if (!compute_mu22) {
        return py::make_tuple(inv_out, py::none());
    }

    py::array_t<double> mu22_out({B, B});
    auto Mout = mu22_out.request();
    double* out = static_cast<double*>(Mout.ptr);

    for (int a = 0; a < B; ++a) {
        for (int b = 0; b < B; ++b) {
            const double denom = (double)N_rows * Msum[(size_t)a] * Msum[(size_t)b];
            double v = 0.0;
            if (denom != 0.0) {
                v = S_sum.ptr[(size_t)a + (size_t)b * (size_t)B] / denom;
                if (!std::isfinite(v)) v = 0.0;
            }
            out[(size_t)a * (size_t)B + (size_t)b] = v;
        }
    }

    return py::make_tuple(inv_out, mu22_out);
}


template <typename Tann>
py::tuple compute_block_corrections_binary_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> meansq_raw,
    py::array_t<double, py::array::c_style | py::array::forcecast> mu22,
    py::array_t<Tann,   py::array::c_style | py::array::forcecast> annot_all,
    int N,
    double d)
{
    auto Minfo = meansq_raw.request();
    auto Uinfo = mu22.request();
    auto Ainfo = annot_all.request();

    if (Minfo.ndim != 2) {
        throw std::runtime_error("meansq_raw must be 2D in compute_block_corrections_binary");
    }
    if (Uinfo.ndim != 2) {
        throw std::runtime_error("mu22 must be 2D in compute_block_corrections_binary");
    }
    if (Ainfo.ndim != 2) {
        throw std::runtime_error("annot_all must be 2D in compute_block_corrections_binary");
    }

    const int M = (int)Minfo.shape[0];
    const int B = (int)Minfo.shape[1];

    if ((int)Uinfo.shape[0] != B || (int)Uinfo.shape[1] != B) {
        throw std::runtime_error("mu22 shape mismatch in compute_block_corrections_binary");
    }
    if ((int)Ainfo.shape[0] != M || (int)Ainfo.shape[1] != B) {
        throw std::runtime_error("annot_all shape mismatch in compute_block_corrections_binary");
    }
    if (N <= 1) {
        throw std::runtime_error("N must be > 1 in compute_block_corrections_binary");
    }

    const double* mptr = static_cast<const double*>(Minfo.ptr);
    const double* uptr = static_cast<const double*>(Uinfo.ptr);
    const Tann* aptr   = static_cast<const Tann*>(Ainfo.ptr);

    std::vector<double> M_a((size_t)B, 0.0);
    std::vector<double> S((size_t)B * (size_t)B, 0.0);

    for (int i = 0; i < M; ++i) {
        const Tann* arow = aptr + (size_t)i * (size_t)B;
        const double* mrow = mptr + (size_t)i * (size_t)B;

        for (int a = 0; a < B; ++a) {
            if (arow[(size_t)a] == (Tann)0) continue;

            M_a[(size_t)a] += 1.0;
            double* Srow = S.data() + (size_t)a * (size_t)B;
            for (int b = 0; b < B; ++b) {
                Srow[(size_t)b] += mrow[(size_t)b];
            }
        }
    }

    py::array_t<double> R2_out({B, B});
    py::array_t<double> rho2_out({B, B});
    py::array_t<double> bias_out({B, B});
    py::array_t<double> delta_out({B, B});

    auto R2i = R2_out.request();
    auto Ri  = rho2_out.request();
    auto Bi  = bias_out.request();
    auto Di  = delta_out.request();

    double* R2p = static_cast<double*>(R2i.ptr);
    double* Rp  = static_cast<double*>(Ri.ptr);
    double* Bp  = static_cast<double*>(Bi.ptr);
    double* Dp  = static_cast<double*>(Di.ptr);

    std::fill(R2p, R2p + (size_t)B * (size_t)B, 0.0);
    std::fill(Rp,  Rp  + (size_t)B * (size_t)B, 0.0);
    std::fill(Bp,  Bp  + (size_t)B * (size_t)B, 0.0);
    std::fill(Dp,  Dp  + (size_t)B * (size_t)B, 0.0);

    const double Nf = (double)N;
    const double denom_rho = Nf * (Nf - 1.0);

    for (int a = 0; a < B; ++a) {
        const double Ma = M_a[(size_t)a];
        if (Ma <= 0.0) continue;

        for (int b = 0; b < B; ++b) {
            const double Mb = M_a[(size_t)b];
            if (Mb <= 0.0) continue;

            const double S_ab = S[(size_t)a * (size_t)B + (size_t)b];
            if (S_ab == 0.0) continue;

            const double R2_ab = S_ab / (Ma * Mb);
            const double mu_ab = uptr[(size_t)a * (size_t)B + (size_t)b];
            const double rho2_ab = (d * d * R2_ab - Nf * mu_ab) / denom_rho;
            const double bias_ab = S_ab - (Ma * Mb * rho2_ab);
            const double delta_ab = mu_ab - (1.0 + 2.0 * rho2_ab);

            R2p[(size_t)a * (size_t)B + (size_t)b] = R2_ab;
            Rp [(size_t)a * (size_t)B + (size_t)b] = rho2_ab;
            Bp [(size_t)a * (size_t)B + (size_t)b] = bias_ab;
            Dp [(size_t)a * (size_t)B + (size_t)b] = delta_ab;
        }
    }

    return py::make_tuple(R2_out, rho2_out, bias_out, delta_out);
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

    m.def("clear_phase1_csr_cache",
          &clear_phase1_csr_cache,
          "Clear the thread-local phase1 CSR cache.");

    m.def("precompute_residual_variances_bed",
        &precompute_residual_variances_bed_impl<float>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("nsnps"),
        py::arg("step_size"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("eps") = 1e-10,
        py::arg("compute_mu22") = false,
        py::arg("annot_all") = py::none(),
        py::arg("C") = py::none(),
        py::arg("R") = py::none());

    m.def("precompute_residual_variances_bed",
        &precompute_residual_variances_bed_impl<double>,
        py::arg("bed_prefix"),
        py::arg("fam_path"),
        py::arg("nsnps"),
        py::arg("step_size"),
        py::arg("row_sel") = py::none(),
        py::arg("ddof") = 1,
        py::arg("eps") = 1e-10,
        py::arg("compute_mu22") = false,
        py::arg("annot_all") = py::none(),
        py::arg("C") = py::none(),
        py::arg("R") = py::none());

    m.def("phase2_accum_XtXz_bed",
          &phase2_accum_XtXz_bed_impl<float>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("blk_start"),
          py::arg("blk_end"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("inv_left"),
          py::arg("tile_nvecs"),
          py::arg("Xz2d"),
          py::arg("meansq_accum"),
          py::arg("C") = py::none(),
          py::arg("R") = py::none(),
          py::arg("N_denom") = 0);

    m.def("phase2_accum_XtXz_bed",
          &phase2_accum_XtXz_bed_impl<double>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("blk_start"),
          py::arg("blk_end"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("inv_left"),
          py::arg("tile_nvecs"),
          py::arg("Xz2d"),
          py::arg("meansq_accum"),
          py::arg("C") = py::none(),
          py::arg("R") = py::none(),
          py::arg("N_denom") = 0);

    m.def("compute_block_corrections_binary",
        &compute_block_corrections_binary_impl<float>,
        py::arg("meansq_raw"),
        py::arg("mu22"),
        py::arg("annot_all"),
        py::arg("N"),
        py::arg("d"));

    m.def("compute_block_corrections_binary",
        &compute_block_corrections_binary_impl<double>,
        py::arg("meansq_raw"),
        py::arg("mu22"),
        py::arg("annot_all"),
        py::arg("N"),
        py::arg("d"));

    m.def("apply_grm_bed_panel",
          &apply_grm_bed_panel_impl<float>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("nsnps"),
          py::arg("step_size"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("inv_all"),
          py::arg("panel_in"),
          py::arg("panel_out"),
          py::arg("C") = py::none(),
          py::arg("R") = py::none());

    m.def("apply_grm_bed_panel",
          &apply_grm_bed_panel_impl<double>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("nsnps"),
          py::arg("step_size"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("inv_all"),
          py::arg("panel_in"),
          py::arg("panel_out"),
          py::arg("C") = py::none(),
          py::arg("R") = py::none());

}
