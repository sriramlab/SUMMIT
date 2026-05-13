// gwldcore.cpp
#include "nb_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

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

// --- Interrupt handling (Ctrl-C) --------------------------------------------
static inline void check_for_interrupt() { nb_check_for_interrupt(); }
static inline bool interrupt_pending_noexcept() {
    nb::gil_scoped_acquire gil;
    return PyErr_CheckSignals() != 0;
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

template <typename T>
static T* read_block_standardized_aligned(const std::string& bed_path,
                                          const std::string& fam_path,
                                          int blk_start, int blk_end,
                                          const std::vector<int>& rows,
                                          int ddof,
                                          ImputeMode impute_mode,
                                          uint64_t impute_seed,
                                          AlignedBuffer<T>& Geno,
                                          int& N, int& L)
{
    const int needN = (int)rows.size();
    const int needL = std::max(0, blk_end - blk_start);
    const size_t need = (size_t)std::max(0, needN) * (size_t)std::max(0, needL);
    if (Geno.n < need) Geno.allocate(need, 64);
    read_block_standardized_into<T>(bed_path, fam_path, blk_start, blk_end,
                                    rows, ddof, impute_mode, impute_seed,
                                    Geno.ptr, Geno.n, N, L);
    return Geno.ptr;
}

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

static inline double env_double_or_default(const char* key, double defv) {
    const char* s = std::getenv(key);
    if (!s || !*s) return defv;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || !std::isfinite(v)) return defv;
    return v;
}

static inline double phase2_upd_rel_factor() {
    static const double v = env_double_or_default("SUMMIT_P2_UPD_REL_FACTOR", 1000.0);
    return v;
}

static inline double phase2_upd_mean_floor_ld() {
    static const double v = env_double_or_default("SUMMIT_P2_UPD_MEAN_FLOOR_LD", 10.0);
    return std::max(0.0, v);
}

static inline double phase2_upd_running_limit(double accepted_sum,
                                              unsigned long long accepted_count,
                                              double mean_floor_update,
                                              double rel_factor) {
    if (!(rel_factor > 0.0)) return std::numeric_limits<double>::infinity();
    double ref = mean_floor_update;
    if (accepted_count > 0) {
        const double mean = accepted_sum / (double) accepted_count;
        if (std::isfinite(mean) && mean > ref) ref = mean;
    }
    return rel_factor * ref;
}

template <typename T>
static inline bool phase2_reject_update(double upd, double running_limit) {
    if (!std::isfinite(upd)) return true;
    if constexpr (std::is_same_v<T, float>) {
        const double fmax = (double) std::numeric_limits<float>::max();
        if (upd > fmax) return true;
    }
    return (std::isfinite(running_limit) && upd > running_limit);
}

// ------------------------------- Small helpers -------------------------------
static void prefetch_bed_block_py(const std::string& bed_prefix,
                                  const std::string& fam_path,
                                  int blk_start, int blk_end,
                                  int ahead_blocks)
{
#if defined(__linux__)
    nb::gil_scoped_release nogil;
    const std::string bed_path = bed_prefix + ".bed";
    prefetch_bed_block(bed_path, fam_path, blk_start, blk_end, ahead_blocks);
#else
    (void)bed_prefix; (void)fam_path; (void)blk_start; (void)blk_end; (void)ahead_blocks;
#endif
}

static inline uint64_t mix64(uint64_t x) {
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
                              const T* A, int lda,
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

static const std::vector<int>& parse_row_sel(nb::object row_sel_obj, int64_t N_total) {
    return parse_row_sel_nb(std::move(row_sel_obj), N_total);
}

static inline ImputeMode parse_impute_mode(const std::string& s) {
    if (s == "hwe") return ImputeMode::Hwe;
    if (s == "mean") return ImputeMode::Mean;
    throw std::runtime_error("impute_mode must be 'mean' or 'hwe'");
}

template <typename Tacc>
static inline int compute_mailman_qpanel_table(int64_t table_size, int q_total, int segment_size) {
    if (q_total <= 0) return 1;
    if (const char* s = std::getenv("SUMMIT_MAILMAN_QPANEL")) {
        const int v = std::atoi(s);
        if (v > 0) return std::max(1, std::min(v, q_total));
    }

    long long mb = 8;  // per-thread lookup-table workspace target
    if (const char* s = std::getenv("SUMMIT_MAILMAN_WORK_MB")) {
        const long long v = std::atoll(s);
        if (v > 0) mb = v;
    }

    size_t budget = (size_t)mb * 1024ULL * 1024ULL;
    size_t bytes_per_col = (size_t)table_size * sizeof(Tacc)
                         + (size_t)std::max(1, segment_size) * sizeof(Tacc)
                         + sizeof(double);
    if (bytes_per_col == 0) return std::max(1, q_total);

    int q = (int)(budget / bytes_per_col);
    q = std::max(1, std::min(q, q_total));
    if (q >= 64) q = (q / 64) * 64;
    if (q < 1) q = 1;
    return q;
}

template <typename Tacc, typename Trhs>
static inline int compute_mailman_qpanel_balanced(int64_t table_size,
                                                  int q_total,
                                                  int segment_size,
                                                  int64_t n_rows,
                                                  bool need_project_copy,
                                                  int p = 0)
{
    int q = compute_mailman_qpanel_table<Tacc>(table_size, q_total, segment_size);

    long long rhs_mb = 256;
    if (const char* s = std::getenv("SUMMIT_MAILMAN_RHS_MB")) {
        const long long v = std::atoll(s);
        if (v > 0) rhs_mb = v;
    }

    const size_t rhs_budget = (size_t)rhs_mb * 1024ULL * 1024ULL;
    size_t rhs_bytes_per_col = (size_t)n_rows * sizeof(Trhs);
    if (need_project_copy) rhs_bytes_per_col += (size_t)n_rows * sizeof(Trhs);
    if (p > 0) rhs_bytes_per_col += (size_t)p * sizeof(Trhs);

    if (rhs_bytes_per_col > 0) {
        const int q_rhs = std::max(1, (int)(rhs_budget / rhs_bytes_per_col));
        q = std::min(q, q_rhs);
    }

    q = std::max(1, std::min(q, q_total));
    if (q >= 64) q = (q / 64) * 64;
    if (q < 1) q = 1;
    return q;
}

template <typename T>
static inline void compute_col_sums(const T* X, int N, int Q, std::vector<double>& sums) {
    sums.assign((size_t)Q, 0.0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < Q; ++c) {
        const T* col = X + (size_t)c * (size_t)N;
        double acc = 0.0;
        for (int i = 0; i < N; ++i) acc += (double)col[(size_t)i];
        sums[(size_t)c] = acc;
    }
}

template <typename T>
static void project_panel_inplace(T* X, int N, int Q,
                                  const T* Cptr, const T* Rptr, int p,
                                  AlignedBuffer<T>& tmp_buf);

template <typename T>
static inline void transpose_colsums_fpanel_to_rowmajor(const T* src_f,
                                                        int N,
                                                        int q,
                                                        T* dst_rm,
                                                        double* sums)
{
    if (q <= 0 || N <= 0) return;
    constexpr int CB = 32;
    constexpr int RB = 64;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c0 = 0; c0 < q; c0 += CB) {
        const int cb = std::min(CB, q - c0);
        double local[CB];
        for (int c = 0; c < cb; ++c) local[c] = 0.0;
        for (int i0 = 0; i0 < N; i0 += RB) {
            const int ib = std::min(RB, N - i0);
            for (int i = 0; i < ib; ++i) {
                T* dst = dst_rm + (size_t)(i0 + i) * (size_t)q + (size_t)c0;
                const T* src = src_f + (size_t)(i0 + i) + (size_t)c0 * (size_t)N;
                for (int c = 0; c < cb; ++c) {
                    const T v = src[(size_t)c * (size_t)N];
                    dst[(size_t)c] = v;
                    local[c] += (double)v;
                }
            }
        }
        for (int c = 0; c < cb; ++c) sums[(size_t)(c0 + c)] = local[c];
    }
}

template <typename T>
static inline T* prepare_mailman_rhs_panel_from_f(const T* Xf,
                                                  int N,
                                                  int Q,
                                                  int q0,
                                                  int q,
                                                  const T* Cptr,
                                                  const T* Rptr,
                                                  int p,
                                                  AlignedBuffer<T>& panel_col,
                                                  AlignedBuffer<T>& panel_row,
                                                  AlignedBuffer<T>& proj_tmp,
                                                  std::vector<double>& sums)
{
    (void)Q;
    const T* src_panel = Xf + (size_t)q0 * (size_t)N;
    if (p > 0) {
        const size_t need_col = (size_t)N * (size_t)q;
        if (panel_col.n < need_col) panel_col.allocate(need_col, 64);
        std::memcpy(panel_col.ptr, src_panel, need_col * sizeof(T));
        project_panel_inplace<T>(panel_col.ptr, N, q, Cptr, Rptr, p, proj_tmp);
        src_panel = panel_col.ptr;
    }
    const size_t need_row = (size_t)N * (size_t)q;
    if (panel_row.n < need_row) panel_row.allocate(need_row, 64);
    sums.assign((size_t)q, 0.0);
    transpose_colsums_fpanel_to_rowmajor<T>(src_panel, N, q, panel_row.ptr, sums.data());
    return panel_row.ptr;
}


template <typename T>
static inline void copy_rowmajor_panel_to_colmajor(const T* src_rm,
                                                   int N,
                                                   int Qfull,
                                                   int q0,
                                                   int q,
                                                   T* dst_cm)
{
    if (q <= 0 || N <= 0) return;
    constexpr int CB = 32;
    constexpr int RB = 64;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c0 = 0; c0 < q; c0 += CB) {
        const int cb = std::min(CB, q - c0);
        for (int i0 = 0; i0 < N; i0 += RB) {
            const int ib = std::min(RB, N - i0);
            for (int i = 0; i < ib; ++i) {
                const T* src = src_rm + (size_t)(i0 + i) * (size_t)Qfull + (size_t)(q0 + c0);
                T* dst = dst_cm + (size_t)(i0 + i) + (size_t)c0 * (size_t)N;
                for (int c = 0; c < cb; ++c) {
                    dst[(size_t)c * (size_t)N] = src[(size_t)c];
                }
            }
        }
    }
}

template <typename T>
static inline void copy_colmajor_panel_to_rowmajor_and_sums(const T* src_cm,
                                                            int N,
                                                            int Qfull,
                                                            int q0,
                                                            int q,
                                                            T* dst_rm,
                                                            double* sums)
{
    if (q <= 0 || N <= 0) return;
    constexpr int CB = 32;
    constexpr int RB = 64;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c0 = 0; c0 < q; c0 += CB) {
        const int cb = std::min(CB, q - c0);
        double local[CB];
        for (int c = 0; c < cb; ++c) local[c] = 0.0;
        for (int i0 = 0; i0 < N; i0 += RB) {
            const int ib = std::min(RB, N - i0);
            for (int i = 0; i < ib; ++i) {
                T* dst = dst_rm + (size_t)(i0 + i) * (size_t)Qfull + (size_t)(q0 + c0);
                const T* src = src_cm + (size_t)(i0 + i) + (size_t)c0 * (size_t)N;
                for (int c = 0; c < cb; ++c) {
                    const T v = src[(size_t)c * (size_t)N];
                    dst[(size_t)c] = v;
                    local[c] += (double)v;
                }
            }
        }
        for (int c = 0; c < cb; ++c) sums[(size_t)(q0 + c0 + c)] = local[c];
    }
}

template <typename T>
static inline void compute_col_sums_rowmajor_core(const T* Xrm,
                                                  int N,
                                                  int Q,
                                                  double* sums)
{
    if (Q <= 0 || N <= 0) return;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < Q; ++c) {
        double acc = 0.0;
        for (int i = 0; i < N; ++i) {
            acc += (double)Xrm[(size_t)i * (size_t)Q + (size_t)c];
        }
        sums[(size_t)c] = acc;
    }
}

template <typename T>
static void project_rowmajor_inplace_and_sums_core(T* Xrm,
                                                   int N,
                                                   int Q,
                                                   const T* Cptr,
                                                   const T* Rptr,
                                                   int p,
                                                   std::vector<double>& sums,
                                                   AlignedBuffer<T>& panel_col,
                                                   AlignedBuffer<T>& proj_tmp)
{
    sums.assign((size_t)Q, 0.0);
    if (Q <= 0 || N <= 0) return;
    if (p <= 0 || !Cptr || !Rptr) {
        compute_col_sums_rowmajor_core<T>(Xrm, N, Q, sums.data());
        return;
    }

    long long rhs_mb = 256;
    if (const char* s = std::getenv("SUMMIT_MAILMAN_RHS_MB")) {
        const long long v = std::atoll(s);
        if (v > 0) rhs_mb = v;
    }
    int qpanel = Q;
    const size_t rhs_budget = (size_t)rhs_mb * 1024ULL * 1024ULL;
    const size_t bytes_per_col = (size_t)N * sizeof(T) + (size_t)p * sizeof(T);
    if (bytes_per_col > 0) {
        int q_rhs = std::max(1, (int)(rhs_budget / bytes_per_col));
        qpanel = std::min(qpanel, q_rhs);
    }
    if (qpanel >= 64) qpanel = (qpanel / 64) * 64;
    if (qpanel < 1) qpanel = 1;

    for (int q0 = 0; q0 < Q; q0 += qpanel) {
        const int q = std::min(qpanel, Q - q0);
        const size_t need_col = (size_t)N * (size_t)q;
        const size_t need_tmp = (size_t)p * (size_t)q;
        if (panel_col.n < need_col) panel_col.allocate(need_col, 64);
        if (proj_tmp.n < need_tmp) proj_tmp.allocate(need_tmp, 64);

        copy_rowmajor_panel_to_colmajor<T>(Xrm, N, Q, q0, q, panel_col.ptr);
        project_panel_inplace<T>(panel_col.ptr, N, q, Cptr, Rptr, p, proj_tmp);
        copy_colmajor_panel_to_rowmajor_and_sums<T>(panel_col.ptr, N, Q, q0, q, Xrm, sums.data());
    }
}

template <typename CodeT, typename T, typename Tacc>
static inline void mailman_pre_accum_groups_rowmajor(const CodeT* packed,
                                                     int segment_size_actual,
                                                     int N,
                                                     int Q,
                                                     const T* op,
                                                     int ldo,
                                                     const double* sum_rhs,
                                                     const int* grp_k,
                                                     const int* grp_tcol0,
                                                     const int* grp_len,
                                                     const double* grp_sumsq,
                                                     int n_groups,
                                                     const double* left_scale,
                                                     const double* means,
                                                     int base_j,
                                                     int blk_start,
                                                     int B,
                                                     T* meansq_accum,
                                                     Tacc* work_table,
                                                     Tacc* row_buf,
                                                     double upd_rel_factor,
                                                     double upd_mean_floor_ld,
                                                     std::atomic<unsigned long long>* dropped_updates)
{
    const int64_t table_size = compute_mailman_table_size(segment_size_actual);
    for (int i = 0; i < N; ++i) {
        Tacc* __restrict tab = work_table + (size_t)packed[(size_t)i] * (size_t)Q;
        const T* __restrict x = op + (size_t)i * (size_t)ldo;
#ifdef _OPENMP
        #pragma omp simd
#endif
        for (int c = 0; c < Q; ++c) tab[(size_t)c] += (Tacc)x[(size_t)c];
    }

    int64_t d = table_size;
    for (int snp_in_seg = 0; snp_in_seg < segment_size_actual; ++snp_in_seg) {
        d /= 3;
#ifdef _OPENMP
        #pragma omp simd
#endif
        for (int c = 0; c < Q; ++c) row_buf[(size_t)c] = Tacc(0);

        for (int64_t i = 0; i < d; ++i) {
            Tacc* __restrict row0 = work_table + (size_t)i * (size_t)Q;
            Tacc* __restrict row1 = work_table + (size_t)(i + d) * (size_t)Q;
            Tacc* __restrict row2 = work_table + (size_t)(i + 2 * d) * (size_t)Q;
#ifdef _OPENMP
            #pragma omp simd
#endif
            for (int c = 0; c < Q; ++c) {
                const Tacc z1 = row1[(size_t)c];
                const Tacc z2 = row2[(size_t)c];
                row1[(size_t)c] = Tacc(0);
                row2[(size_t)c] = Tacc(0);
                row0[(size_t)c] += z1 + z2;
                row_buf[(size_t)c] += z1 + Tacc(2) * z2;
            }
        }

        const int j = base_j + snp_in_seg;
        const double alpha = left_scale[(size_t)j];
        if (alpha <= 0.0 || !std::isfinite(alpha)) continue;

        const double mean = means[(size_t)j];
        const double beta = -2.0 * mean * alpha;
        const double gamma = mean * mean * alpha;
        T* out = meansq_accum + ((size_t)(blk_start + j) * (size_t)B);
        double accepted_sum = 0.0;
        unsigned long long accepted_count = 0;
        unsigned long long local_dropped = 0;

        for (int g = 0; g < n_groups; ++g) {
            const int k = grp_k[g];
            const int tcol = grp_tcol0[g];
            const int len = grp_len[g];
            const double mean_floor_update = upd_mean_floor_ld * (double) len;
            const Tacc* src = row_buf + (size_t)tcol;
            const double* svec = sum_rhs + (size_t)tcol;
            double sq = 0.0;
            double dot = 0.0;
#ifdef _OPENMP
            #pragma omp simd reduction(+:sq,dot)
#endif
            for (int c = 0; c < len; ++c) {
                const double w = (double)src[(size_t)c];
                sq += w * w;
                dot += w * svec[(size_t)c];
            }
            double upd = alpha * sq + beta * dot + gamma * grp_sumsq[g];
            const double running_limit = phase2_upd_running_limit(
                accepted_sum, accepted_count, mean_floor_update, upd_rel_factor);
            if (phase2_reject_update<T>(upd, running_limit)) {
                ++local_dropped;
                continue;
            }
            if (upd < 0.0) upd = 0.0;
            out[(size_t)k] += (T)upd;
            accepted_sum += upd;
            ++accepted_count;
        }
        if (local_dropped && dropped_updates)
            dropped_updates->fetch_add(local_dropped, std::memory_order_relaxed);
    }

#ifdef _OPENMP
    #pragma omp simd
#endif
    for (int c = 0; c < Q; ++c) work_table[(size_t)c] = Tacc(0);
}

template <typename CodeT, typename T, typename Tacc>
static inline void mailman_pre_multiply_rowmajor(const CodeT* packed,
                                                 int segment_size_actual,
                                                 int N,
                                                 int Q,
                                                 const T* op,
                                                 int ldo,
                                                 Tacc* result,
                                                 Tacc* work_table)
{
    const int64_t table_size = compute_mailman_table_size(segment_size_actual);
    for (int i = 0; i < N; ++i) {
        Tacc* __restrict tab = work_table + (size_t)packed[(size_t)i] * (size_t)Q;
        const T* __restrict x = op + (size_t)i * (size_t)ldo;
#ifdef _OPENMP
        #pragma omp simd
#endif
        for (int c = 0; c < Q; ++c) tab[(size_t)c] += (Tacc)x[(size_t)c];
    }
    int64_t d = table_size;
    for (int snp_in_seg = 0; snp_in_seg < segment_size_actual; ++snp_in_seg) {
        d /= 3;
        Tacc* __restrict out = result + (size_t)snp_in_seg * (size_t)Q;
#ifdef _OPENMP
        #pragma omp simd
#endif
        for (int c = 0; c < Q; ++c) out[(size_t)c] = Tacc(0);
        for (int64_t i = 0; i < d; ++i) {
            Tacc* __restrict row0 = work_table + (size_t)i * (size_t)Q;
            Tacc* __restrict row1 = work_table + (size_t)(i + d) * (size_t)Q;
            Tacc* __restrict row2 = work_table + (size_t)(i + 2 * d) * (size_t)Q;
#ifdef _OPENMP
            #pragma omp simd
#endif
            for (int c = 0; c < Q; ++c) {
                const Tacc z1 = row1[(size_t)c];
                const Tacc z2 = row2[(size_t)c];
                row1[(size_t)c] = Tacc(0);
                row2[(size_t)c] = Tacc(0);
                row0[(size_t)c] += z1 + z2;
                out[(size_t)c] += z1 + Tacc(2) * z2;
            }
        }
    }
#ifdef _OPENMP
    #pragma omp simd
#endif
    for (int c = 0; c < Q; ++c) work_table[(size_t)c] = Tacc(0);
}

template <typename CodeT, typename T>
static inline void mailman_post_multiply_colmajor_subset(const CodeT* packed,
                                                         int segment_size_actual,
                                                         int row_start,
                                                         int row_count,
                                                         int Q,
                                                         const double* op,
                                                         int ldop,
                                                         T* result,
                                                         int ldres,
                                                         double* work_table)
{
    const int64_t table_size = compute_mailman_table_size(segment_size_actual);
    std::memset(work_table, 0, (size_t)table_size * (size_t)Q * sizeof(double));
    int64_t prefix = 1;
    for (int i = segment_size_actual - 1; i >= 0; --i) {
        const double* op_row = op + (size_t)i * (size_t)ldop;
        for (int64_t j = 0; j < prefix; ++j) {
            const int64_t off0 = j * (int64_t)Q;
            const int64_t off1 = (prefix + j) * (int64_t)Q;
            const int64_t off2 = (2 * prefix + j) * (int64_t)Q;
            for (int c = 0; c < Q; ++c) {
                const double base = work_table[(size_t)off0 + (size_t)c];
                work_table[(size_t)off1 + (size_t)c] = base + op_row[(size_t)c];
                work_table[(size_t)off2 + (size_t)c] = base + 2.0 * op_row[(size_t)c];
            }
        }
        prefix *= 3;
    }
    for (int i = 0; i < row_count; ++i) {
        const CodeT code = packed[(size_t)(row_start + i)];
        const double* src = work_table + (size_t)code * (size_t)Q;
        T* dst = result + (size_t)(row_start + i);
        for (int c = 0; c < Q; ++c) dst[(size_t)c * (size_t)ldres] += (T)src[(size_t)c];
    }
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
    if (!s || !*s) return 256ULL * 1024ULL * 1024ULL;
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

    out->bytes = out->colptr.size() * sizeof(int) + (std::size_t)nnz * (sizeof(int) + sizeof(T));
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

        if (built->bytes > cap) return built;

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
        s.map.clear(); s.bytes = 0; s.clock = 0;
    }
    {
        auto& s = phase1_csr_store<double>();
        s.map.clear(); s.bytes = 0; s.clock = 0;
    }
}

template <typename T>
static void project_panel_inplace(T* X, int N, int Q,
                                  const T* Cptr, const T* Rptr, int p,
                                  AlignedBuffer<T>& tmp_buf)
{
    if (p <= 0 || N <= 0 || Q <= 0) return;

    const size_t need = (size_t)p * (size_t)Q;
    if (tmp_buf.n < need) tmp_buf.allocate(need, 64);
    T* tmp = tmp_buf.ptr;

    gemm_col_major_nn<T>(p, Q, N, Rptr, p, X, N, tmp, p, T(1), T(0));
    gemm_col_major_nn<T>(N, Q, p, Cptr, N, tmp, p, X, N, T(-1), T(1));
}

template <typename T>
void apply_grm_bed_panel_impl(
    const std::string& bed_prefix,
    const std::string& fam_path,
    int nsnps,
    int step_size,
    nb::object row_sel_obj,
    int ddof,
    nb_vec1_ro<T> inv_all,
    nb_mat2f_ro<T> panel_in,
    nb_mat2f_rw<T> panel_out,
    nb::object C_opt,
    nb::object R_opt,
    const std::string& impute_mode_str,
    nb::object impute_seed_obj)
{
    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (nsnps < 0 || nsnps > (int) M_total)
        throw std::runtime_error("nsnps is out of range in apply_grm_bed_panel");
    if (step_size <= 0)
        throw std::runtime_error("step_size must be > 0 in apply_grm_bed_panel");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    const int N_rows = (int) rows.size();
    if (N_rows <= 0)
        throw std::runtime_error("No rows selected in apply_grm_bed_panel");

    if ((int) inv_all.shape(0) < nsnps)
        throw std::runtime_error("inv_all shape mismatch in apply_grm_bed_panel");
    const T* invp = inv_all.data();

    if ((int) panel_in.shape(0) != N_rows || (int) panel_out.shape(0) != N_rows)
        throw std::runtime_error("panel row mismatch in apply_grm_bed_panel");
    if ((int) panel_in.shape(1) != (int) panel_out.shape(1))
        throw std::runtime_error("panel col mismatch in apply_grm_bed_panel");

    const int Q = (int) panel_in.shape(1);
    const T* inptr = panel_in.data();
    T* outptr = panel_out.data();

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;

    if (have_proj) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N_rows || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N_rows)
            throw std::runtime_error("C/R shape mismatch in apply_grm_bed_panel");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    }

    AlignedBuffer<T> Vin((size_t) N_rows * (size_t) Q, 64);
    std::memcpy(Vin.ptr, inptr, sizeof(T) * (size_t) N_rows * (size_t) Q);
    std::fill(outptr, outptr + (size_t) N_rows * (size_t) Q, T(0));

    AlignedBuffer<T> proj_tmp(have_proj ? (size_t) p * (size_t) Q : 0, 64);
    AlignedBuffer<T> coef_buf((size_t) std::min(step_size, nsnps) * (size_t) Q, 64);

    nb::gil_scoped_release nogil;

    if (have_proj)
        project_panel_inplace<T>(Vin.ptr, N_rows, Q, Cptr, Rptr, p, proj_tmp);

    const double invM = 1.0 / (double) nsnps;

    for (int s = 0; s < nsnps; s += step_size) {
        check_for_interrupt();
        const int e = std::min(nsnps, s + step_size);
        const int L = e - s;
        if (L <= 0)
            continue;

        int N_blk = 0, L_blk = 0;
        static thread_local AlignedBuffer<T> Geno_tls;
        T* Geno = read_block_standardized_aligned<T>(bed_path, fam_path, s, e,
                                                     rows, ddof,
                                                     impute_mode, impute_seed,
                                                     Geno_tls, N_blk, L_blk);
        if (N_blk != N_rows || L_blk != L)
            throw std::runtime_error("Unexpected block dimensions in apply_grm_bed_panel");

        T* coef = coef_buf.ptr;
        gemm_col_major_tn<T>(L, Q, N_rows, Geno, N_rows,
                             Vin.ptr, N_rows, coef, L, T(1), T(0));

        static thread_local std::vector<T> row_scale_tls;
        if (row_scale_tls.size() < (size_t) L) row_scale_tls.resize((size_t) L);
        T* row_scale = row_scale_tls.data();
        for (int j = 0; j < L; ++j) {
            const double invj = (double) invp[(size_t) (s + j)];
            double w = invj * invj * invM;
            if (!std::isfinite(w) || w <= 0.0)
                w = 0.0;
            row_scale[(size_t) j] = (T) w;
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int q = 0; q < Q; ++q) {
            T* ccol = coef + (size_t) q * (size_t) L;
            for (int j = 0; j < L; ++j)
                ccol[(size_t) j] *= row_scale[(size_t) j];
        }

        gemm_col_major_nn<T>(N_rows, Q, L, Geno, N_rows,
                             coef, L, outptr, N_rows, T(1), T(1));
    }

    if (have_proj)
        project_panel_inplace<T>(outptr, N_rows, Q, Cptr, Rptr, p, proj_tmp);
}

template <typename T>
void apply_grm_bed_panel_mailman_impl(
    const std::string& bed_prefix,
    const std::string& fam_path,
    int nsnps,
    int step_size,
    nb::object row_sel_obj,
    int ddof,
    nb_vec1_ro<T> inv_all,
    nb_mat2f_ro<T> panel_in,
    nb_mat2f_rw<T> panel_out,
    nb::object C_opt,
    nb::object R_opt,
    nb::object impute_seed_obj)
{
    using Tacc = std::conditional_t<std::is_same_v<T, double>, double, float>;

    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (nsnps < 0 || nsnps > (int) M_total)
        throw std::runtime_error("nsnps is out of range in apply_grm_bed_panel_mailman");
    if (step_size <= 0)
        throw std::runtime_error("step_size must be > 0 in apply_grm_bed_panel_mailman");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    const int N_rows = (int) rows.size();
    if (N_rows <= 0)
        throw std::runtime_error("No rows selected in apply_grm_bed_panel_mailman");

    if ((int) inv_all.shape(0) < nsnps)
        throw std::runtime_error("inv_all shape mismatch in apply_grm_bed_panel_mailman");
    const T* invp = inv_all.data();

    if ((int) panel_in.shape(0) != N_rows || (int) panel_out.shape(0) != N_rows)
        throw std::runtime_error("panel row mismatch in apply_grm_bed_panel_mailman");
    if ((int) panel_in.shape(1) != (int) panel_out.shape(1))
        throw std::runtime_error("panel col mismatch in apply_grm_bed_panel_mailman");

    const int Q = (int) panel_in.shape(1);
    const T* inptr = panel_in.data();
    T* outptr = panel_out.data();

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;
    if (have_proj) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N_rows || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N_rows)
            throw std::runtime_error("C/R shape mismatch in apply_grm_bed_panel_mailman");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    }

    std::fill(outptr, outptr + (size_t) N_rows * (size_t) Q, T(0));
    AlignedBuffer<T> rhs_panel_col;
    AlignedBuffer<T> rhs_panel_row;
    AlignedBuffer<T> proj_tmp(have_proj ? (size_t) p * (size_t) std::max(1, Q) : 0, 64);
    std::vector<double> sum_Vin;

    nb::gil_scoped_release nogil;

    T* Vin_row_ptr = prepare_mailman_rhs_panel_from_f<T>(
        inptr, N_rows, Q, 0, Q, Cptr, Rptr, p,
        rhs_panel_col, rhs_panel_row, proj_tmp, sum_Vin);

    const double invM = 1.0 / (double) nsnps;

    MailmanPackedBlock pack;
    std::vector<double> coef;
    std::vector<double> mean_corr;

    for (int s = 0; s < nsnps; s += step_size) {
        check_for_interrupt();
        const int e = std::min(nsnps, s + step_size);
        const int L = e - s;
        if (L <= 0)
            continue;

        read_block_mailman_hwe(bed_path, fam_path, s, e, rows, ddof, impute_seed, pack);
        if (pack.N != N_rows || pack.L != L)
            throw std::runtime_error("Unexpected pack dimensions in apply_grm_bed_panel_mailman");

        const int qpanel = compute_mailman_qpanel_balanced<Tacc, T>(pack.table_size, Q, pack.segment_size, N_rows, false, 0);
        coef.resize((size_t) L * (size_t) Q);

        for (int q0 = 0; q0 < Q; q0 += qpanel) {
            const int q = std::min(qpanel, Q - q0);
            const T* rhs = Vin_row_ptr + (size_t) q0;
            const double* sum_rhs = sum_Vin.data() + (size_t) q0;
#ifdef _OPENMP
            #pragma omp parallel
#endif
            {
                static thread_local AlignedBuffer<Tacc> work_table_tls;
                static thread_local AlignedBuffer<Tacc> raw_seg_tls;

                const size_t need_table = (size_t) pack.table_size * (size_t) q;
                const size_t need_raw = (size_t) pack.segment_size * (size_t) q;

                if (work_table_tls.n < need_table) {
                    work_table_tls.allocate(need_table, 64);
                    std::memset(work_table_tls.ptr, 0, need_table * sizeof(Tacc));
                }
                if (raw_seg_tls.n < need_raw)
                    raw_seg_tls.allocate(need_raw, 64);

                Tacc* work_table = work_table_tls.ptr;
                Tacc* raw_seg = raw_seg_tls.ptr;
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (int64_t seg = 0; seg < pack.n_segments; ++seg) {
                    const int base = (int) (seg * (int64_t) pack.segment_size);
                    const int actual = std::min(pack.segment_size, L - base);
                    if (pack.use_u16) {
                        mailman_pre_multiply_rowmajor<uint16_t, T, Tacc>(pack.packed16.data() + (size_t) seg * (size_t) N_rows,
                                                                         actual, N_rows, q, rhs, Q,
                                                                         raw_seg, work_table);
                    } else {
                        mailman_pre_multiply_rowmajor<uint32_t, T, Tacc>(pack.packed32.data() + (size_t) seg * (size_t) N_rows,
                                                                         actual, N_rows, q, rhs, Q,
                                                                         raw_seg, work_table);
                    }
                    for (int r = 0; r < actual; ++r) {
                        const int j = base + r;
                        const double mean = pack.mean[(size_t) j];
                        const double inv_std = pack.inv_std[(size_t) j];
                        double* dst = coef.data() + (size_t) j * (size_t) Q + (size_t) q0;
                        const Tacc* src = raw_seg + (size_t) r * (size_t) q;
#ifdef _OPENMP
                        #pragma omp simd
#endif
                        for (int c = 0; c < q; ++c)
                            dst[(size_t) c] = ((double) src[(size_t) c] - mean * sum_rhs[(size_t) c]) * inv_std;
                    }
                }
            }
        }

        mean_corr.resize((size_t) Q);
        std::fill(mean_corr.begin(), mean_corr.end(), 0.0);
        for (int j = 0; j < L; ++j) {
            const double scale = (double) invp[(size_t) (s + j)] * (double) invp[(size_t) (s + j)] * invM * pack.inv_std[(size_t) j];
            const double mean = pack.mean[(size_t) j];
            double* row = coef.data() + (size_t) j * (size_t) Q;
#ifdef _OPENMP
            #pragma omp simd
#endif
            for (int c = 0; c < Q; ++c) {
                row[(size_t) c] *= scale;
                mean_corr[(size_t) c] += mean * row[(size_t) c];
            }
        }

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
            const int Tn = omp_get_num_threads();
#else
            const int tid = 0;
            const int Tn = 1;
#endif
            const int base_rows = N_rows / Tn;
            const int rem = N_rows % Tn;
            const int my_start = tid * base_rows + std::min(tid, rem);
            const int my_count = base_rows + (tid < rem ? 1 : 0);

            static thread_local AlignedBuffer<double> work_table_tls;
            for (int q0 = 0; q0 < Q; q0 += qpanel) {
                const int q = std::min(qpanel, Q - q0);
                const size_t need_table = (size_t) pack.table_size * (size_t) q;
                if (work_table_tls.n < need_table)
                    work_table_tls.allocate(need_table, 64);
                double* work_table = work_table_tls.ptr;
                std::memset(work_table, 0, need_table * sizeof(double));

                for (int64_t seg = 0; seg < pack.n_segments; ++seg) {
                    const int base = (int) (seg * (int64_t) pack.segment_size);
                    const int actual = std::min(pack.segment_size, L - base);
                    if (pack.use_u16) {
                        mailman_post_multiply_colmajor_subset<uint16_t>(pack.packed16.data() + (size_t) seg * (size_t) N_rows,
                                                                        actual,
                                                                        my_start,
                                                                        my_count,
                                                                        q,
                                                                        coef.data() + (size_t) base * (size_t) Q + (size_t) q0,
                                                                        Q,
                                                                        outptr + (size_t) q0 * (size_t) N_rows,
                                                                        N_rows,
                                                                        work_table);
                    } else {
                        mailman_post_multiply_colmajor_subset<uint32_t>(pack.packed32.data() + (size_t) seg * (size_t) N_rows,
                                                                        actual,
                                                                        my_start,
                                                                        my_count,
                                                                        q,
                                                                        coef.data() + (size_t) base * (size_t) Q + (size_t) q0,
                                                                        Q,
                                                                        outptr + (size_t) q0 * (size_t) N_rows,
                                                                        N_rows,
                                                                        work_table);
                    }
                }
                for (int c = 0; c < q; ++c) {
                    T* col = outptr + (size_t) (q0 + c) * (size_t) N_rows + (size_t) my_start;
                    const T corr = (T) mean_corr[(size_t) (q0 + c)];
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (int i = 0; i < my_count; ++i)
                        col[(size_t) i] -= corr;
                }
            }
        }
    }

    if (have_proj)
        project_panel_inplace<T>(outptr, N_rows, Q, Cptr, Rptr, p, proj_tmp);
}

template <typename T>
static void compute_inv_resvar_from_geno_block(
    const T* Geno,
    int N,
    int L,
    double eps,
    nb::object C_opt,
    nb::object R_opt,
    nb::object gram_opt,
    T* inv_out);

template <typename T>
void phase1_compute_Xz_bed_chunk_impl(const std::string &bed_prefix,
                                      const std::string &fam_path,
                                      int blk_start, int blk_end,
                                      nb::object row_sel_obj,
                                      int ddof,
                                      nb_mat2c_ro<T> annot_blk,
                                      nb::object inv_right_obj,
                                      int v_start,
                                      int v_count,
                                      int kmax_hint,
                                      const std::string &rand_dist,
                                      nb::object seed_obj,
                                      nb_mat2f_rw<T> Xz2d_chunk,
                                      nb::object bin_init_mask_obj,
                                      bool project_right,
                                      nb::object C_opt,
                                      nb::object R_opt,
                                      const std::string& impute_mode_str,
                                      nb::object impute_seed_obj,
                                      nb::object resvar_gram_obj,
                                      nb::object inv_out_obj,
                                      double resvar_eps)
{
    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total)
        throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    static thread_local AlignedBuffer<T> Geno_tls;
    T* Geno = read_block_standardized_aligned<T>(bed_path, fam_path, blk_start, blk_end,
                                                 rows, ddof,
                                                 impute_mode, impute_seed,
                                                 Geno_tls, N, L);
    if (L == 0)
        return;

    if (project_right && !C_opt.is_none() && !R_opt.is_none()) {
        if (!inv_out_obj.is_none())
            throw std::runtime_error("Fused residual variance is incompatible with project_right=True");
        auto C = nb::cast<nb_mat2f_ro<T>>(C_opt);
        auto R = nb::cast<nb_mat2f_ro<T>>(R_opt);
        const int p = (int) C.shape(1);
        if ((int) C.shape(0) != N || (int) R.shape(0) != p || (int) R.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in phase1");
        const T* Cptr = C.data();
        const T* Rptr = R.data();
        static thread_local AlignedBuffer<T> tmpG_tls;
        const size_t need_tmp = (size_t) p * (size_t) L;
        if (tmpG_tls.n < need_tmp) tmpG_tls.allocate(need_tmp, 64);
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno, N, tmpG_tls.ptr, p, T(1), T(0));
        gemm_col_major_nn<T>(N, L, p, Cptr, N, tmpG_tls.ptr, p, Geno, N, T(-1), T(1));
    }

    const int B = (int) annot_blk.shape(1);
    if ((int) annot_blk.shape(0) != L)
        throw std::runtime_error("L mismatch");
    const T* ann = annot_blk.data();

    nb_vec1_ro<T> inv_right;
    nb_vec1_rw<T> inv_out;
    const T* inv = nullptr;
    if (!inv_out_obj.is_none()) {
        inv_out = nb::cast<nb_vec1_rw<T>>(inv_out_obj);
        if ((int) inv_out.shape(0) != L)
            throw std::runtime_error("inv_out shape mismatch in phase1 fused residual variance");
        compute_inv_resvar_from_geno_block<T>(
            Geno, N, L, resvar_eps,
            C_opt, R_opt, resvar_gram_obj,
            inv_out.data()
        );
        inv = inv_out.data();
    } else {
        inv_right = nb::cast<nb_vec1_ro<T>>(inv_right_obj);
        if ((int) inv_right.shape(0) != L)
            throw std::runtime_error("inv_right shape mismatch in phase1");
        inv = inv_right.data();
    }

    if ((int) Xz2d_chunk.shape(0) != N || (int) Xz2d_chunk.shape(1) != B * v_count)
        throw std::runtime_error("Xz2d_chunk shape must be (N, B*v_count)");
    T* Xptr = Xz2d_chunk.data();
    const int ldc = N;

    uint8_t* bin_init = nullptr;
    nb_vec1_rw<uint8_t> bin_init_mask;
    if (!bin_init_mask_obj.is_none()) {
        bin_init_mask = nb::cast<nb_vec1_rw<uint8_t>>(bin_init_mask_obj);
        if ((int) bin_init_mask.shape(0) != B)
            throw std::runtime_error("bin_init_mask shape mismatch in phase1");
        bin_init = bin_init_mask.data();
    }

    const bool have_root = !seed_obj.is_none();
    const uint64_t root_seed = have_root ? nb::cast<uint64_t>(seed_obj) : std::random_device{}();
    std::mt19937_64 rng(make_seed(root_seed, blk_start, v_start));
    std::normal_distribution<T> gN(0, (T) 1);
    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_spherical = (rand_dist == "spherical");

    static thread_local AlignedBuffer<T> Z_tls;
    const size_t needZ = (size_t) L * (size_t) v_count;
    if (Z_tls.n < needZ) Z_tls.allocate(needZ, 64);
    T* Zptr = Z_tls.ptr;
    for (int c = 0; c < v_count; ++c) {
        long double ss = 0.0L;
        T* zc = Zptr + (size_t) c * (size_t) L;
        for (int r = 0; r < L; ++r) {
            T z = is_rademacher ? ((rng() & 1) ? T(+1) : T(-1)) : gN(rng);
            zc[(size_t) r] = z;
            if (is_spherical)
                ss += (long double) z * (long double) z;
        }
        if (is_spherical) {
            const T zscale = ss > 0.0L ? (T) std::sqrt((long double) L / ss) : T(1);
            for (int r = 0; r < L; ++r)
                zc[(size_t) r] *= zscale;
        }
    }

    auto csr = get_or_build_phase1_csr<T>(ann, inv, L, B);
    const std::vector<int>& colptr = csr->colptr;
    const int* rowind = csr->rowind.ptr;
    const T* scale = csr->scale.ptr;

    int Kcap = std::max(1, kmax_hint);
    for (int k = 0; k < B; ++k)
        Kcap = std::max(Kcap, colptr[(size_t) k + 1] - colptr[(size_t) k]);

    const char* s = std::getenv("SUMMIT_P1_NTILE");
    int NTILE = s ? std::max(64, std::atoi(s)) : (std::is_same_v<T, double> ? 256 : 512);
    const bool do_timing = verbose_enabled();
    using Clock = std::chrono::high_resolution_clock;
    P1Timers t1;

    static thread_local AlignedBuffer<T> Bcol_shared_tls;
    const size_t needB = (size_t) Kcap * (size_t) v_count;
    if (Bcol_shared_tls.n < needB) Bcol_shared_tls.allocate(needB, 64);
    T* Bcol_shared = Bcol_shared_tls.ptr;

    int shared_k0 = 0;
    int sharedK = 0;
    size_t shared_base_col = 0;
    T shared_beta = T(1);
    bool interrupted = false;

    check_for_interrupt();

    {
        nb::gil_scoped_release nogil;
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            static thread_local AlignedBuffer<T> A_tile_tls;
            const size_t needA = (size_t) NTILE * (size_t) Kcap;
            if (A_tile_tls.n < needA) A_tile_tls.allocate(needA, 64);
            T* A_tile = A_tile_tls.ptr;
            double packZ_ms = 0.0;
            double packA_ms = 0.0;
            double gemm_ms = 0.0;

            for (int k = 0; k < B; ++k) {
#ifdef _OPENMP
                #pragma omp master
#endif
                {
                    if (!interrupted && interrupt_pending_noexcept()) {
                        interrupted = true;
                        sharedK = 0;
                    }
                    if (!interrupted) {
                        shared_k0 = colptr[(size_t) k];
                        const int k1 = colptr[(size_t) k + 1];
                        sharedK = k1 - shared_k0;
                        shared_base_col = (size_t) k * (size_t) v_count;
                        shared_beta = T(1);
                        if (sharedK > 0 && bin_init) {
                            shared_beta = bin_init[(size_t) k] ? T(1) : T(0);
                            bin_init[(size_t) k] = 1u;
                        }
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
                if (!interrupted && sharedK > 0) {
                    const auto z0 = do_timing ? Clock::now() : Clock::time_point{};
#ifdef _OPENMP
                    #pragma omp for schedule(static)
#endif
                    for (int c = 0; c < v_count; ++c) {
                        const T* zc = Zptr + (size_t) c * (size_t) L;
                        T* dst = Bcol_shared + (size_t) c * (size_t) sharedK;
                        for (int r = 0; r < sharedK; ++r) {
                            const size_t idx = (size_t) shared_k0 + (size_t) r;
                            dst[(size_t) r] = zc[(size_t) rowind[idx]] * scale[idx];
                        }
                    }
                    if (do_timing) {
                        const auto z1 = Clock::now();
                        packZ_ms += std::chrono::duration<double, std::milli>(z1 - z0).count();
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
                if (!interrupted && sharedK > 0) {
#ifdef _OPENMP
                    #pragma omp for schedule(static)
#endif
                    for (int n0 = 0; n0 < N; n0 += NTILE) {
                        const int Nt = std::min(N - n0, NTILE);
                        const auto a0 = do_timing ? Clock::now() : Clock::time_point{};
                        for (int c = 0; c < sharedK; ++c) {
                            const int snp = rowind[(size_t) shared_k0 + (size_t) c];
                            const T* src = Geno + (size_t) snp * (size_t) N + (size_t) n0;
                            T* dst = A_tile + (size_t) c * (size_t) Nt;
                            std::memcpy(dst, src, (size_t) Nt * sizeof(T));
                        }
                        if (do_timing) {
                            const auto a1 = Clock::now();
                            packA_ms += std::chrono::duration<double, std::milli>(a1 - a0).count();
                        }

                        const auto g0 = do_timing ? Clock::now() : Clock::time_point{};
                        T* Cdst = Xptr + shared_base_col * (size_t) ldc + (size_t) n0;
                        gemm_col_major_nn<T>(Nt, v_count, sharedK,
                                             A_tile, Nt,
                                             Bcol_shared, sharedK,
                                             Cdst, ldc,
                                             T(1), shared_beta);
                        if (do_timing) {
                            const auto g1 = Clock::now();
                            gemm_ms += std::chrono::duration<double, std::milli>(g1 - g0).count();
                        }
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }
            if (do_timing) {
#ifdef _OPENMP
                #pragma omp atomic
#endif
                t1.t_packZ_ms += packZ_ms;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                t1.t_packA_ms += packA_ms;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                t1.t_gemm_ms += gemm_ms;
            }
        }
    }

    if (interrupted)
        throw nb::python_error();

    t1.dump(blk_start, blk_end, B, v_count);
}

template <typename T>
void phase1_compute_Xz_bed_chunk_rowmajor_impl(const std::string &bed_prefix,
                                               const std::string &fam_path,
                                               int blk_start, int blk_end,
                                               nb::object row_sel_obj,
                                               int ddof,
                                               nb_mat2c_ro<T> annot_blk,
                                               nb::object inv_right_obj,
                                               int v_start,
                                               int v_count,
                                               int,
                                               const std::string &rand_dist,
                                               nb::object seed_obj,
                                               nb_mat2c_rw<T> Xz2d_chunk,
                                               bool project_right,
                                               nb::object C_opt,
                                               nb::object R_opt,
                                               const std::string& impute_mode_str,
                                               nb::object impute_seed_obj,
                                               nb::object resvar_gram_obj,
                                               nb::object inv_out_obj,
                                               double resvar_eps)
{
    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total)
        throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    static thread_local AlignedBuffer<T> Geno_tls;
    T* Geno = read_block_standardized_aligned<T>(bed_path, fam_path, blk_start, blk_end,
                                                 rows, ddof,
                                                 impute_mode, impute_seed,
                                                 Geno_tls, N, L);
    if (L == 0)
        return;

    if (project_right && !C_opt.is_none() && !R_opt.is_none()) {
        if (!inv_out_obj.is_none())
            throw std::runtime_error("Fused residual variance is incompatible with project_right=True");
        auto C = nb::cast<nb_mat2f_ro<T>>(C_opt);
        auto R = nb::cast<nb_mat2f_ro<T>>(R_opt);
        const int p = (int) C.shape(1);
        if ((int) C.shape(0) != N || (int) R.shape(0) != p || (int) R.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in phase1 rowmajor");
        const T* Cptr = C.data();
        const T* Rptr = R.data();
        static thread_local AlignedBuffer<T> tmpG_tls;
        const size_t need_tmp = (size_t) p * (size_t) L;
        if (tmpG_tls.n < need_tmp) tmpG_tls.allocate(need_tmp, 64);
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno, N, tmpG_tls.ptr, p, T(1), T(0));
        gemm_col_major_nn<T>(N, L, p, Cptr, N, tmpG_tls.ptr, p, Geno, N, T(-1), T(1));
    }

    const int B = (int) annot_blk.shape(1);
    if ((int) annot_blk.shape(0) != L)
        throw std::runtime_error("L mismatch");
    const T* ann = annot_blk.data();

    nb_vec1_ro<T> inv_right;
    nb_vec1_rw<T> inv_out;
    const T* inv = nullptr;
    if (!inv_out_obj.is_none()) {
        inv_out = nb::cast<nb_vec1_rw<T>>(inv_out_obj);
        if ((int) inv_out.shape(0) != L)
            throw std::runtime_error("inv_out shape mismatch in rowmajor phase1 fused residual variance");
        compute_inv_resvar_from_geno_block<T>(
            Geno, N, L, resvar_eps,
            C_opt, R_opt, resvar_gram_obj,
            inv_out.data()
        );
        inv = inv_out.data();
    } else {
        inv_right = nb::cast<nb_vec1_ro<T>>(inv_right_obj);
        if ((int) inv_right.shape(0) != L)
            throw std::runtime_error("inv_right shape mismatch in rowmajor phase1");
        inv = inv_right.data();
    }

    if ((int) Xz2d_chunk.shape(0) != N || (int) Xz2d_chunk.shape(1) != B * v_count)
        throw std::runtime_error("Xz2d_chunk shape must be (N, B*v_count) in rowmajor phase1");
    T* Xptr = Xz2d_chunk.data();
    const int row_stride = B * v_count;

    const bool have_root = !seed_obj.is_none();
    const uint64_t root_seed = have_root ? nb::cast<uint64_t>(seed_obj) : std::random_device{}();
    std::mt19937_64 rng(make_seed(root_seed, blk_start, v_start));
    std::normal_distribution<T> gN(0, (T) 1);
    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_spherical = (rand_dist == "spherical");

    std::vector<T> Z((size_t) L * (size_t) v_count, T(0));
    for (int c = 0; c < v_count; ++c) {
        long double ss = 0.0L;
        for (int r = 0; r < L; ++r) {
            T z = is_rademacher ? ((rng() & 1) ? T(+1) : T(-1)) : gN(rng);
            Z[(size_t) r + (size_t) c * (size_t) L] = z;
            if (is_spherical)
                ss += (long double) z * (long double) z;
        }
        if (is_spherical) {
            T zscale = ss > 0.0L ? (T) std::sqrt((long double) L / ss) : T(1);
            for (int r = 0; r < L; ++r)
                Z[(size_t) r + (size_t) c * (size_t) L] *= zscale;
        }
    }

    auto csr = get_or_build_phase1_csr<T>(ann, inv, L, B);
    const std::vector<int>& colptr = csr->colptr;
    const int* rowind = csr->rowind.ptr;
    const T* scale = csr->scale.ptr;

    const char* s = std::getenv("SUMMIT_P1_NTILE");
    int NTILE = s ? std::max(64, std::atoi(s)) : (std::is_same_v<T, double> ? 256 : 512);
    P1Timers t1;

    for (int k = 0; k < B; ++k) {
        check_for_interrupt();
        const int k0 = colptr[(size_t) k];
        const int k1 = colptr[(size_t) k + 1];
        const int K = k1 - k0;
        if (K == 0)
            continue;

        AlignedBuffer<T> Bcol((size_t) K * (size_t) v_count, 64);

        auto z0 = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < v_count; ++c) {
            const T* zc = Z.data() + (size_t) c * (size_t) L;
            T* dst = Bcol.ptr + (size_t) c * (size_t) K;
            for (int r = 0; r < K; ++r) {
                const int snp = rowind[(size_t) k0 + (size_t) r];
                dst[(size_t) r] = zc[(size_t) snp];
            }
        }
        auto z1 = std::chrono::high_resolution_clock::now();
        t1.t_packZ_ms += std::chrono::duration<double, std::milli>(z1 - z0).count();

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            AlignedBuffer<T> A_tile((size_t) NTILE * (size_t) K, 64);
            AlignedBuffer<T> C_tile((size_t) NTILE * (size_t) v_count, 64);
            double packA_ms = 0.0, gemm_ms = 0.0, scatt_ms = 0.0;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int n0 = 0; n0 < N; n0 += NTILE) {
                const int Nt = std::min(N - n0, NTILE);
                auto a0 = std::chrono::high_resolution_clock::now();
                for (int c = 0; c < K; ++c) {
                    const int snp = rowind[(size_t) k0 + (size_t) c];
                    const T ssc = scale[(size_t) k0 + (size_t) c];
                    const T* src = Geno + (size_t) snp * (size_t) N + (size_t) n0;
                    T* dst = A_tile.ptr + (size_t) c * (size_t) Nt;
#pragma omp simd
                    for (int r = 0; r < Nt; ++r)
                        dst[r] = src[r] * ssc;
                }
                auto a1 = std::chrono::high_resolution_clock::now();
                packA_ms += std::chrono::duration<double, std::milli>(a1 - a0).count();

                auto g0 = std::chrono::high_resolution_clock::now();
                gemm_col_major_nn<T>(Nt, v_count, K, A_tile.ptr, Nt, Bcol.ptr, K, C_tile.ptr, Nt, T(1), T(0));
                auto g1 = std::chrono::high_resolution_clock::now();
                gemm_ms += std::chrono::duration<double, std::milli>(g1 - g0).count();

                auto s0 = std::chrono::high_resolution_clock::now();
                const size_t base_col = (size_t) k * (size_t) v_count;
                for (int r = 0; r < Nt; ++r) {
                    T* dst_row = Xptr + ((size_t) (n0 + r) * (size_t) row_stride) + base_col;
                    const T* src = C_tile.ptr + (size_t) r;
#pragma omp simd
                    for (int c = 0; c < v_count; ++c)
                        dst_row[(size_t) c] += src[(size_t) c * (size_t) Nt];
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

template <typename T>
void phase2_compute_XtXz_bed_impl(const std::string &bed_prefix,
                                  const std::string &fam_path,
                                  int blk_start, int blk_end,
                                  nb::object row_sel_obj,
                                  int ddof,
                                  nb_vec1_ro<T> inv_left,
                                  int nvecs,
                                  int,
                                  nb_mat2f_ro<T> Xz2d,
                                  nb_mat2c_rw<T> meansq,
                                  nb::object C_opt, nb::object R_opt,
                                  int N_denom,
                                  const std::string& impute_mode_str,
                                  nb::object impute_seed_obj)
{
    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    auto round_down = [](int x, int m) { return (m > 0) ? (x / m) * m : x; };
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total)
        throw std::runtime_error("blk_end exceeds #SNPs in BIM");
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    static thread_local AlignedBuffer<T> Geno_tls;
    T* Geno = read_block_standardized_aligned<T>(bed_path, fam_path, blk_start, blk_end,
                                                 rows, ddof,
                                                 impute_mode, impute_seed,
                                                 Geno_tls, N, L);
    if (L == 0)
        return;

    if (!C_opt.is_none() && !R_opt.is_none()) {
        auto C = nb::cast<nb_mat2f_ro<T>>(C_opt);
        auto R = nb::cast<nb_mat2f_ro<T>>(R_opt);
        const int p = (int) C.shape(1);
        if ((int) C.shape(0) != N || (int) R.shape(0) != p || (int) R.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch");
        const T* Cptr = C.data();
        const T* Rptr = R.data();
        static thread_local AlignedBuffer<T> tmpG_tls;
        const size_t need_tmp = (size_t) p * (size_t) L;
        if (tmpG_tls.n < need_tmp) tmpG_tls.allocate(need_tmp, 64);
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno, N, tmpG_tls.ptr, p, T(1), T(0));
        gemm_col_major_nn<T>(N, L, p, Cptr, N, tmpG_tls.ptr, p, Geno, N, T(-1), T(1));
    }

    if ((int) inv_left.shape(0) != L)
        throw std::runtime_error("inv_left shape mismatch");
    const T* inv = inv_left.data();

    if ((int) Xz2d.shape(0) != N)
        throw std::runtime_error("Xz2d shape mismatch");
    const T* Xptr = Xz2d.data();
    const int BV = (int) Xz2d.shape(1);
    if (nvecs <= 0 || (BV % nvecs) != 0)
        throw std::runtime_error("Xz2d col count must be multiple of nvecs");
    const int B = BV / nvecs;

    if ((int) meansq.shape(1) != B)
        throw std::runtime_error("meansq shape mismatch");
    T* Mptr = meansq.data();
    const int M = (int) meansq.shape(0);
    if (blk_end > M)
        throw std::runtime_error("meansq rows smaller than SNP count");

    double denom = (double) N_denom - 1.0;
    if (denom <= 0.0)
        denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);
    const double invV = 1.0 / (double) nvecs;

    int Q = BV;
    int QPANEL = 32768;
    if (const char* s = std::getenv("SUMMIT_P2_QPANEL")) QPANEL = std::atoi(s);
    if (QPANEL <= 0) QPANEL = Q;
    QPANEL = std::min(QPANEL, Q);
    QPANEL = round_down(QPANEL, 64);
    if (QPANEL < 64) QPANEL = std::min(Q, 64);

    static thread_local AlignedBuffer<T> Work_panel_tls;
    const size_t needW = (size_t) L * (size_t) QPANEL;
    if (Work_panel_tls.n < needW) Work_panel_tls.allocate(needW, 64);
    T* Work = Work_panel_tls.ptr;

    int IBLK = 2048;
    if (const char* s = std::getenv("SUMMIT_P2_IBLK")) IBLK = std::atoi(s);
    IBLK = clampi(IBLK, 512, 16384);
    IBLK = round_up(IBLK, 512);

    int REDUCE_THREADS = 1;
    if (const char* s = std::getenv("SUMMIT_P2_REDUCE_THREADS")) REDUCE_THREADS = std::atoi(s);
#ifdef _OPENMP
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = omp_get_max_threads();
#else
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = 1;
#endif
    const int nslabs = ceil_div_i(L, IBLK);
    REDUCE_THREADS = clampi(REDUCE_THREADS, 1, std::max(1, nslabs));

    BlockTimers t;
    const double upd_rel_factor = phase2_upd_rel_factor();
    const double upd_mean_floor_ld = phase2_upd_mean_floor_ld();
    std::atomic<unsigned long long> dropped_updates{0};
    nb::gil_scoped_release nogil;

    for (int q0 = 0; q0 < Q; q0 += QPANEL) {
        check_for_interrupt();
        const int q = std::min(QPANEL, Q - q0);
        const T* rhs = Xptr + (size_t) q0 * (size_t) N;
        auto t2 = std::chrono::high_resolution_clock::now();
        gemm_col_major_tn<T>(L, q, N, Geno, N, rhs, N, Work, L, T(1), T(0));
        auto t3 = std::chrono::high_resolution_clock::now();
        t.add_gemm(std::chrono::duration<double, std::milli>(t3 - t2).count());

        int seg_k[512], seg_tcol0[512], seg_len[512], segments = 0;
        if (B > 512)
            throw std::runtime_error("B too large for fixed seg buffers");
        int g = q0, g_end = q0 + q;
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

        auto t4 = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(REDUCE_THREADS)
#endif
        for (int i0 = 0; i0 < L; i0 += IBLK) {
            const int ib = std::min(IBLK, L - i0);
            static thread_local AlignedBuffer<double> acc_tls;
            if (acc_tls.n < (size_t) IBLK) acc_tls.allocate((size_t) IBLK, 64);
            double* acc = acc_tls.ptr;
            unsigned long long local_dropped = 0;
            for (int sidx = 0; sidx < segments; ++sidx) {
                const int k = seg_k[sidx], tcol = seg_tcol0[sidx], len = seg_len[sidx];
                const double mean_floor_update = upd_mean_floor_ld * ((double) len * invV);
                double accepted_sum = 0.0;
                unsigned long long accepted_count = 0;
                std::fill(acc, acc + ib, 0.0);
                for (int c = 0; c < len; ++c) {
                    const T* wcol = Work + (size_t) (tcol + c) * (size_t) L + (size_t) i0;
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (int ii = 0; ii < ib; ++ii) {
                        const double w = (double) wcol[ii];
                        acc[ii] += w * w;
                    }
                }
                T* out = Mptr + ((size_t) (blk_start + i0) * (size_t) B + (size_t) k);
                for (int ii = 0; ii < ib; ++ii) {
                    const double inv_i = (double) inv[i0 + ii];
                    const double inv2 = inv_i * inv_i * inv_denom2;
                    double upd = acc[ii] * invV * inv2;
                    const double running_limit = phase2_upd_running_limit(
                        accepted_sum, accepted_count, mean_floor_update, upd_rel_factor);
                    if (phase2_reject_update<T>(upd, running_limit)) {
                        ++local_dropped;
                        continue;
                    }
                    if (upd < 0.0) upd = 0.0;
                    out[(size_t) ii * (size_t) B] += (T) upd;
                    accepted_sum += upd;
                    ++accepted_count;
                }
            }
            if (local_dropped)
                dropped_updates.fetch_add(local_dropped, std::memory_order_relaxed);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        t.add_reduce(std::chrono::duration<double, std::milli>(t5 - t4).count());
    }
    const unsigned long long ndrop = dropped_updates.load(std::memory_order_relaxed);
    if (ndrop > 0) {
        std::fprintf(stderr,
            "[phase2-warn] dropped %llu oversized/non-finite updates in block [%d:%d) "
            "(rel_factor=%.6g, mean_floor_ld=%.6g)\n",
            ndrop, blk_start, blk_end, upd_rel_factor, upd_mean_floor_ld);
    }
    t.dump(blk_start, blk_end, B, nvecs);
}

template <typename T>
void phase2_accum_XtXz_bed_impl(const std::string &bed_prefix,
                                const std::string &fam_path,
                                int blk_start, int blk_end,
                                nb::object row_sel_obj,
                                int ddof,
                                nb_vec1_ro<T> inv_left,
                                int tile_nvecs,
                                nb_mat2f_ro<T> Xz2d,
                                nb_mat2c_rw<T> meansq_accum,
                                nb::object C_opt,
                                nb::object R_opt,
                                int N_denom,
                                const std::string& impute_mode_str,
                                nb::object impute_seed_obj)
{
    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    auto round_down = [](int x, int m) { return (m > 0) ? (x / m) * m : x; };
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total)
        throw std::runtime_error("blk_end exceeds #SNPs in BIM");
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    static thread_local AlignedBuffer<T> Geno_tls;
    T* Geno = read_block_standardized_aligned<T>(bed_path, fam_path, blk_start, blk_end,
                                                 rows, ddof,
                                                 impute_mode, impute_seed,
                                                 Geno_tls, N, L);
    if (L == 0)
        return;

    if (!C_opt.is_none() && !R_opt.is_none()) {
        auto C = nb::cast<nb_mat2f_ro<T>>(C_opt);
        auto R = nb::cast<nb_mat2f_ro<T>>(R_opt);
        const int p = (int) C.shape(1);
        if ((int) C.shape(0) != N || (int) R.shape(0) != p || (int) R.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in phase2_accum_XtXz_bed");
        const T* Cptr = C.data();
        const T* Rptr = R.data();
        static thread_local AlignedBuffer<T> tmpG_tls;
        const size_t need_tmp = (size_t) p * (size_t) L;
        if (tmpG_tls.n < need_tmp) tmpG_tls.allocate(need_tmp, 64);
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno, N, tmpG_tls.ptr, p, T(1), T(0));
        gemm_col_major_nn<T>(N, L, p, Cptr, N, tmpG_tls.ptr, p, Geno, N, T(-1), T(1));
    }

    if ((int) inv_left.shape(0) != L)
        throw std::runtime_error("inv_left shape mismatch in phase2_accum_XtXz_bed");
    const T* inv = inv_left.data();

    if ((int) Xz2d.shape(0) != N)
        throw std::runtime_error("Xz2d shape mismatch in phase2_accum_XtXz_bed");
    const T* Xptr = Xz2d.data();
    const int Q = (int) Xz2d.shape(1);
    if (tile_nvecs <= 0 || (Q % tile_nvecs) != 0)
        throw std::runtime_error("Xz2d col count must be a multiple of tile_nvecs");
    const int B = Q / tile_nvecs;

    if ((int) meansq_accum.shape(1) != B)
        throw std::runtime_error("meansq_accum shape mismatch in phase2_accum_XtXz_bed");
    if (blk_end > (int) meansq_accum.shape(0))
        throw std::runtime_error("meansq_accum rows smaller than SNP count");
    T* Mptr = meansq_accum.data();

    double denom = (double) N_denom - 1.0;
    if (denom <= 0.0)
        denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    static thread_local std::vector<double> left_scale_tls;
    if (left_scale_tls.size() < (size_t) L)
        left_scale_tls.resize((size_t) L);
    double* left_scale = left_scale_tls.data();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < L; ++i) {
        const double inv_i = (double) inv[(size_t) i];
        left_scale[(size_t) i] = (std::isfinite(inv_i) && inv_i > 0.0) ? (inv_i * inv_i * inv_denom2) : 0.0;
    }

    int QPANEL = 32768;
    if (const char* s = std::getenv("SUMMIT_P2_QPANEL")) QPANEL = std::atoi(s);
    if (QPANEL <= 0) QPANEL = Q;
    QPANEL = std::min(QPANEL, Q);
    QPANEL = round_down(QPANEL, 64);
    if (QPANEL < 64) QPANEL = std::min(Q, 64);

    static thread_local AlignedBuffer<T> Work_panel_tls;
    const size_t needW = (size_t) L * (size_t) QPANEL;
    if (Work_panel_tls.n < needW) Work_panel_tls.allocate(needW, 64);
    T* Work = Work_panel_tls.ptr;

    int IBLK = 2048;
    if (const char* s = std::getenv("SUMMIT_P2_IBLK")) IBLK = std::atoi(s);
    IBLK = clampi(IBLK, 512, 16384);
    IBLK = round_up(IBLK, 512);

    int REDUCE_THREADS = 1;
    if (const char* s = std::getenv("SUMMIT_P2_REDUCE_THREADS")) REDUCE_THREADS = std::atoi(s);
#ifdef _OPENMP
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = omp_get_max_threads();
#else
    if (REDUCE_THREADS <= 0) REDUCE_THREADS = 1;
#endif
    const int nslabs = ceil_div_i(L, IBLK);
    REDUCE_THREADS = clampi(REDUCE_THREADS, 1, std::max(1, nslabs));

    const bool do_timing = verbose_enabled();
    using Clock = std::chrono::high_resolution_clock;
    BlockTimers t;
    const double upd_rel_factor = phase2_upd_rel_factor();
    const double upd_mean_floor_ld = phase2_upd_mean_floor_ld();
    std::atomic<unsigned long long> dropped_updates{0};
    nb::gil_scoped_release nogil;

    for (int q0 = 0; q0 < Q; q0 += QPANEL) {
        check_for_interrupt();
        const int q = std::min(QPANEL, Q - q0);
        const T* rhs = Xptr + (size_t) q0 * (size_t) N;
        const auto t2 = do_timing ? Clock::now() : Clock::time_point{};
        gemm_col_major_tn<T>(L, q, N, Geno, N, rhs, N, Work, L, T(1), T(0));
        if (do_timing) {
            const auto t3 = Clock::now();
            t.add_gemm(std::chrono::duration<double, std::milli>(t3 - t2).count());
        }

        if (B > 512)
            throw std::runtime_error("B too large for fixed seg buffers in phase2_accum_XtXz_bed");
        int seg_k[512], seg_tcol0[512], seg_len[512], segments = 0;
        int g = q0, g_end = q0 + q;
        while (g < g_end) {
            const int k = g / tile_nvecs;
            const int v_in = g - k * tile_nvecs;
            const int len = std::min(g_end - g, tile_nvecs - v_in);
            seg_k[segments] = k;
            seg_tcol0[segments] = g - q0;
            seg_len[segments] = len;
            ++segments;
            g += len;
        }

        const auto t4 = do_timing ? Clock::now() : Clock::time_point{};
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(REDUCE_THREADS)
#endif
        for (int i0 = 0; i0 < L; i0 += IBLK) {
            const int ib = std::min(IBLK, L - i0);
            static thread_local AlignedBuffer<double> acc_tls;
            if (acc_tls.n < (size_t) IBLK) acc_tls.allocate((size_t) IBLK, 64);
            double* acc = acc_tls.ptr;
            unsigned long long local_dropped = 0;
            for (int sidx = 0; sidx < segments; ++sidx) {
                const int k = seg_k[sidx], tcol = seg_tcol0[sidx], len = seg_len[sidx];
                const double mean_floor_update = upd_mean_floor_ld * (double) len;
                double accepted_sum = 0.0;
                unsigned long long accepted_count = 0;
                std::fill(acc, acc + ib, 0.0);
                for (int c = 0; c < len; ++c) {
                    const T* wcol = Work + (size_t) (tcol + c) * (size_t) L + (size_t) i0;
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (int ii = 0; ii < ib; ++ii) {
                        const double w = (double) wcol[ii];
                        acc[ii] += w * w;
                    }
                }
                T* out = Mptr + ((size_t) (blk_start + i0) * (size_t) B + (size_t) k);
                for (int ii = 0; ii < ib; ++ii) {
                    double upd = acc[ii] * left_scale[(size_t) (i0 + ii)];
                    const double running_limit = phase2_upd_running_limit(
                        accepted_sum, accepted_count, mean_floor_update, upd_rel_factor);
                    if (phase2_reject_update<T>(upd, running_limit)) {
                        ++local_dropped;
                        continue;
                    }
                    if (upd < 0.0) upd = 0.0;
                    out[(size_t) ii * (size_t) B] += (T) upd;
                    accepted_sum += upd;
                    ++accepted_count;
                }
            }
            if (local_dropped)
                dropped_updates.fetch_add(local_dropped, std::memory_order_relaxed);
        }
        if (do_timing) {
            const auto t5 = Clock::now();
            t.add_reduce(std::chrono::duration<double, std::milli>(t5 - t4).count());
        }
    }
    const unsigned long long ndrop = dropped_updates.load(std::memory_order_relaxed);
    if (ndrop > 0) {
        std::fprintf(stderr,
            "[phase2-warn] dropped %llu oversized/non-finite updates in block [%d:%d) "
            "(rel_factor=%.6g, mean_floor_ld=%.6g)\n",
            ndrop, blk_start, blk_end, upd_rel_factor, upd_mean_floor_ld);
    }
    t.dump(blk_start, blk_end, B, tile_nvecs);
}

template <typename T>
void phase2_accum_XtXz_bed_mailman_impl(const std::string &bed_prefix,
                                        const std::string &fam_path,
                                        int blk_start, int blk_end,
                                        nb::object row_sel_obj,
                                        int ddof,
                                        nb_vec1_ro<T> inv_left,
                                        int tile_nvecs,
                                        nb_mat2f_ro<T> Xz2d,
                                        nb_mat2c_rw<T> meansq_accum,
                                        nb::object C_opt,
                                        nb::object R_opt,
                                        int N_denom,
                                        nb::object impute_seed_obj)
{
    using Tacc = std::conditional_t<std::is_same_v<T, double>, double, float>;

    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total)
        throw std::runtime_error("blk_end exceeds #SNPs in BIM");
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    MailmanPackedBlock pack;
    read_block_mailman_hwe(bed_path, fam_path, blk_start, blk_end, rows, ddof, impute_seed, pack);
    const int N = pack.N;
    const int L = pack.L;
    if (L == 0)
        return;

    if ((int) inv_left.shape(0) != L)
        throw std::runtime_error("inv_left shape mismatch in phase2_accum_XtXz_bed_mailman");
    const T* inv = inv_left.data();

    if ((int) Xz2d.shape(0) != N)
        throw std::runtime_error("Xz2d shape mismatch in phase2_accum_XtXz_bed_mailman");
    const T* Xptr = Xz2d.data();
    const int Q = (int) Xz2d.shape(1);
    if (tile_nvecs <= 0 || (Q % tile_nvecs) != 0)
        throw std::runtime_error("Xz2d col count must be a multiple of tile_nvecs");
    const int B = Q / tile_nvecs;

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;
    if (have_proj) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in phase2_accum_XtXz_bed_mailman");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    }

    if ((int) meansq_accum.shape(1) != B)
        throw std::runtime_error("meansq_accum shape mismatch in phase2_accum_XtXz_bed_mailman");
    if (blk_end > (int) meansq_accum.shape(0))
        throw std::runtime_error("meansq_accum rows smaller than SNP count");
    T* Mptr = meansq_accum.data();

    double denom = (double) N_denom - 1.0;
    if (denom <= 0.0)
        denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    std::vector<double> left_scale((size_t) L);
    for (int i = 0; i < L; ++i) {
        const double inv_i = (double) inv[(size_t) i];
        if (std::isfinite(inv_i) && inv_i > 0.0)
            left_scale[(size_t) i] = inv_i * inv_i * inv_denom2 * pack.inv_std[(size_t) i] * pack.inv_std[(size_t) i];
        else
            left_scale[(size_t) i] = 0.0;
    }

    const int qpanel = compute_mailman_qpanel_balanced<Tacc, T>(pack.table_size, Q, pack.segment_size, N, have_proj, p);
    AlignedBuffer<T> rhs_panel_col;
    AlignedBuffer<T> rhs_panel_row;
    AlignedBuffer<T> proj_tmp(have_proj ? (size_t) p * (size_t) std::max(1, qpanel) : 0, 64);
    std::vector<double> sum_rhs_panel;
    const double upd_rel_factor = phase2_upd_rel_factor();
    const double upd_mean_floor_ld = phase2_upd_mean_floor_ld();
    std::atomic<unsigned long long> dropped_updates{0};

    nb::gil_scoped_release nogil;

    for (int q0 = 0; q0 < Q; q0 += qpanel) {
        check_for_interrupt();
        const int q = std::min(qpanel, Q - q0);
        T* rhs = prepare_mailman_rhs_panel_from_f<T>(Xptr, N, Q, q0, q, Cptr, Rptr, p,
                                                     rhs_panel_col, rhs_panel_row, proj_tmp,
                                                     sum_rhs_panel);
        const double* sum_rhs = sum_rhs_panel.data();

        std::vector<int> seg_k;
        std::vector<int> seg_tcol0;
        std::vector<int> seg_len;
        std::vector<double> seg_sumsq;
        seg_k.reserve((size_t) B + 1);
        seg_tcol0.reserve((size_t) B + 1);
        seg_len.reserve((size_t) B + 1);
        seg_sumsq.reserve((size_t) B + 1);

        int g = q0;
        const int g_end = q0 + q;
        while (g < g_end) {
            const int k = g / tile_nvecs;
            const int v_in = g - k * tile_nvecs;
            const int len = std::min(g_end - g, tile_nvecs - v_in);
            seg_k.push_back(k);
            seg_tcol0.push_back(g - q0);
            seg_len.push_back(len);
            double ss = 0.0;
            for (int c = 0; c < len; ++c) {
                const double v = sum_rhs[(size_t) (g - q0 + c)];
                ss += v * v;
            }
            seg_sumsq.push_back(ss);
            g += len;
        }

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            static thread_local AlignedBuffer<Tacc> work_table_tls;
            static thread_local AlignedBuffer<Tacc> raw_seg_tls;

            const size_t need_table = (size_t) pack.table_size * (size_t) q;
            const size_t need_raw = (size_t) pack.segment_size * (size_t) q;
            if (work_table_tls.n < need_table) {
                work_table_tls.allocate(need_table, 64);
                std::memset(work_table_tls.ptr, 0, need_table * sizeof(Tacc));
            }
            if (raw_seg_tls.n < need_raw) raw_seg_tls.allocate(need_raw, 64);
            Tacc* work_table = work_table_tls.ptr;
            Tacc* raw_seg = raw_seg_tls.ptr;

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int64_t seg = 0; seg < pack.n_segments; ++seg) {
                unsigned long long local_dropped = 0;
                const int base = (int) (seg * (int64_t) pack.segment_size);
                const int actual = std::min(pack.segment_size, L - base);
                if (pack.use_u16) {
                    mailman_pre_multiply_rowmajor<uint16_t, T, Tacc>(pack.packed16.data() + (size_t) seg * (size_t) N,
                                                                     actual, N, q, rhs, q,
                                                                     raw_seg, work_table);
                } else {
                    mailman_pre_multiply_rowmajor<uint32_t, T, Tacc>(pack.packed32.data() + (size_t) seg * (size_t) N,
                                                                     actual, N, q, rhs, q,
                                                                     raw_seg, work_table);
                }
                for (int r = 0; r < actual; ++r) {
                    const int j = base + r;
                    const double row_scale = left_scale[(size_t) j];
                    if (row_scale <= 0.0 || !std::isfinite(row_scale)) continue;
                    const double mean = pack.mean[(size_t) j];
                    const double mean2 = mean * mean;
                    T* out = Mptr + ((size_t) (blk_start + j) * (size_t) B);
                    const Tacc* src = raw_seg + (size_t) r * (size_t) q;
                    double accepted_sum = 0.0;
                    unsigned long long accepted_count = 0;
                    for (std::size_t sidx = 0; sidx < seg_k.size(); ++sidx) {
                        const int k = seg_k[sidx];
                        const int tcol = seg_tcol0[sidx];
                        const int len = seg_len[sidx];
                        const double mean_floor_update = upd_mean_floor_ld * (double) len;
                        const double* sum_seg = sum_rhs + (size_t) tcol;
                        double sq = 0.0;
                        double dot = 0.0;
#ifdef _OPENMP
                        #pragma omp simd reduction(+:sq,dot)
#endif
                        for (int c = 0; c < len; ++c) {
                            const double w = (double) src[(size_t) (tcol + c)];
                            sq += w * w;
                            dot += w * sum_seg[(size_t) c];
                        }
                        double upd = (sq - 2.0 * mean * dot + mean2 * seg_sumsq[sidx]) * row_scale;
                        const double running_limit = phase2_upd_running_limit(
                            accepted_sum, accepted_count, mean_floor_update, upd_rel_factor);
                        if (phase2_reject_update<T>(upd, running_limit)) {
                            ++local_dropped;
                            continue;
                        }
                        if (upd < 0.0) upd = 0.0;
                        out[(size_t) k] += (T) upd;
                        accepted_sum += upd;
                        ++accepted_count;
                    }
                }
                if (local_dropped)
                    dropped_updates.fetch_add(local_dropped, std::memory_order_relaxed);
            }
        }
    }
    const unsigned long long ndrop = dropped_updates.load(std::memory_order_relaxed);
    if (ndrop > 0) {
        std::fprintf(stderr,
            "[phase2-warn] dropped %llu oversized/non-finite mailman updates in block [%d:%d) "
            "(rel_factor=%.6g, mean_floor_ld=%.6g)\n",
            ndrop, blk_start, blk_end, upd_rel_factor, upd_mean_floor_ld);
    }
}

template <typename T>
void phase2_accum_XtXz_bed_mailman_rm_impl(const std::string &bed_prefix,
                                           const std::string &fam_path,
                                           int blk_start, int blk_end,
                                           nb::object row_sel_obj,
                                           int ddof,
                                           nb_vec1_ro<T> inv_left,
                                           int tile_nvecs,
                                           nb_mat2c_ro<T> Xz2d,
                                           nb_mat2c_rw<T> meansq_accum,
                                           nb_vec1_ro<double> sum_Xz,
                                           int N_denom,
                                           nb::object impute_seed_obj)
{
    using Tacc = std::conditional_t<std::is_same_v<T, double>, double, float>;

    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total)
        throw std::runtime_error("blk_end exceeds #SNPs in BIM");
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);

    MailmanPackedBlock pack;
    read_block_mailman_hwe(bed_path, fam_path, blk_start, blk_end, rows, ddof, impute_seed, pack);
    const int N = pack.N;
    const int L = pack.L;
    if (L == 0)
        return;

    if ((int) inv_left.shape(0) != L)
        throw std::runtime_error("inv_left shape mismatch in phase2_accum_XtXz_bed_mailman_rm");
    const T* inv = inv_left.data();

    if ((int) Xz2d.shape(0) != N)
        throw std::runtime_error("Xz2d shape mismatch in phase2_accum_XtXz_bed_mailman_rm");
    const T* Xptr = Xz2d.data();
    const int Q = (int) Xz2d.shape(1);
    if (tile_nvecs <= 0 || (Q % tile_nvecs) != 0)
        throw std::runtime_error("Xz2d col count must be a multiple of tile_nvecs");
    const int B = Q / tile_nvecs;

    if ((int) sum_Xz.shape(0) != Q)
        throw std::runtime_error("sum_Xz shape mismatch in phase2_accum_XtXz_bed_mailman_rm");
    const double* sump = sum_Xz.data();

    if ((int) meansq_accum.shape(1) != B)
        throw std::runtime_error("meansq_accum shape mismatch in phase2_accum_XtXz_bed_mailman_rm");
    if (blk_end > (int) meansq_accum.shape(0))
        throw std::runtime_error("meansq_accum rows smaller than SNP count");
    T* Mptr = meansq_accum.data();

    double denom = (double) N_denom - 1.0;
    if (denom <= 0.0)
        denom = 1.0;
    const double inv_denom2 = 1.0 / (denom * denom);

    std::vector<double> left_scale((size_t) L);
    for (int i = 0; i < L; ++i) {
        const double inv_i = (double) inv[(size_t) i];
        if (std::isfinite(inv_i) && inv_i > 0.0)
            left_scale[(size_t) i] = inv_i * inv_i * inv_denom2 * pack.inv_std[(size_t) i] * pack.inv_std[(size_t) i];
        else
            left_scale[(size_t) i] = 0.0;
    }

    const int qpanel = compute_mailman_qpanel_table<Tacc>(pack.table_size, Q, pack.segment_size);
    const double upd_rel_factor = phase2_upd_rel_factor();
    const double upd_mean_floor_ld = phase2_upd_mean_floor_ld();
    std::atomic<unsigned long long> dropped_updates{0};

    nb::gil_scoped_release nogil;

    for (int q0 = 0; q0 < Q; q0 += qpanel) {
        check_for_interrupt();
        const int q = std::min(qpanel, Q - q0);
        const T* rhs = Xptr + (size_t) q0;
        const double* sum_rhs = sump + (size_t) q0;

        std::vector<int> grp_k;
        std::vector<int> grp_tcol0;
        std::vector<int> grp_len;
        std::vector<double> grp_sumsq;
        grp_k.reserve((size_t) B + 1);
        grp_tcol0.reserve((size_t) B + 1);
        grp_len.reserve((size_t) B + 1);
        grp_sumsq.reserve((size_t) B + 1);

        int g = q0;
        const int g_end = q0 + q;
        while (g < g_end) {
            const int k = g / tile_nvecs;
            const int v_in = g - k * tile_nvecs;
            const int len = std::min(g_end - g, tile_nvecs - v_in);
            grp_k.push_back(k);
            grp_tcol0.push_back(g - q0);
            grp_len.push_back(len);
            double ss = 0.0;
            for (int c = 0; c < len; ++c) {
                const double v = sum_rhs[(size_t) (g - q0 + c)];
                ss += v * v;
            }
            grp_sumsq.push_back(ss);
            g += len;
        }

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            static thread_local AlignedBuffer<Tacc> work_table_tls;
            static thread_local AlignedBuffer<Tacc> row_buf_tls;
            const size_t need_table = (size_t) pack.table_size * (size_t) q;
            const size_t need_row = (size_t) q;
            if (work_table_tls.n < need_table) {
                work_table_tls.allocate(need_table, 64);
                std::memset(work_table_tls.ptr, 0, need_table * sizeof(Tacc));
            }
            if (row_buf_tls.n < need_row) row_buf_tls.allocate(need_row, 64);
            Tacc* work_table = work_table_tls.ptr;
            Tacc* row_buf = row_buf_tls.ptr;

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int64_t seg = 0; seg < pack.n_segments; ++seg) {
                const int base = (int) (seg * (int64_t) pack.segment_size);
                const int actual = std::min(pack.segment_size, L - base);
                if (pack.use_u16) {
                    mailman_pre_accum_groups_rowmajor<uint16_t, T, Tacc>(pack.packed16.data() + (size_t) seg * (size_t) N,
                                                                         actual, N, q, rhs, Q,
                                                                         sum_rhs,
                                                                         grp_k.data(), grp_tcol0.data(), grp_len.data(), grp_sumsq.data(), (int) grp_k.size(),
                                                                         left_scale.data(), pack.mean.data(),
                                                                         base, blk_start, B, Mptr,
                                                                         work_table, row_buf,
                                                                         upd_rel_factor, upd_mean_floor_ld, &dropped_updates);
                } else {
                    mailman_pre_accum_groups_rowmajor<uint32_t, T, Tacc>(pack.packed32.data() + (size_t) seg * (size_t) N,
                                                                         actual, N, q, rhs, Q,
                                                                         sum_rhs,
                                                                         grp_k.data(), grp_tcol0.data(), grp_len.data(), grp_sumsq.data(), (int) grp_k.size(),
                                                                         left_scale.data(), pack.mean.data(),
                                                                         base, blk_start, B, Mptr,
                                                                         work_table, row_buf,
                                                                         upd_rel_factor, upd_mean_floor_ld, &dropped_updates);
                }
            }
        }
    }
    const unsigned long long ndrop = dropped_updates.load(std::memory_order_relaxed);
    if (ndrop > 0) {
        std::fprintf(stderr,
            "[phase2-warn] dropped %llu oversized/non-finite mailman rowmajor updates in block [%d:%d) "
            "(rel_factor=%.6g, mean_floor_ld=%.6g)\n",
            ndrop, blk_start, blk_end, upd_rel_factor, upd_mean_floor_ld);
    }
}

template <typename T>
nb_numpy_vec1<double> compute_col_sums_rowmajor_impl(nb_mat2c_ro<T> Xz2d)
{
    const int N = (int) Xz2d.shape(0);
    const int Q = (int) Xz2d.shape(1);
    const T* Xptr = Xz2d.data();
    double* optr = nullptr;
    auto out = make_owned_numpy_vec1<double>((size_t) Q, &optr);
    nb::gil_scoped_release nogil;
    compute_col_sums_rowmajor_core<T>(Xptr, N, Q, optr);
    return out;
}

template <typename T>
nb_numpy_vec1<double> project_rowmajor_inplace_and_col_sums_impl(nb_mat2c_rw<T> Xz2d,
                                                                  nb::object C_opt,
                                                                  nb::object R_opt)
{
    const int N = (int) Xz2d.shape(0);
    const int Q = (int) Xz2d.shape(1);
    T* Xptr = Xz2d.data();

    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;
    if (!C_opt.is_none() && !R_opt.is_none()) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in project_rowmajor_inplace_and_col_sums");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    }

    std::vector<double> sums;
    AlignedBuffer<T> panel_col;
    AlignedBuffer<T> proj_tmp;
    {
        nb::gil_scoped_release nogil;
        project_rowmajor_inplace_and_sums_core<T>(Xptr, N, Q, Cptr, Rptr, p, sums, panel_col, proj_tmp);
    }

    double* optr = nullptr;
    auto out = make_owned_numpy_vec1<double>((size_t) Q, &optr);
    std::memcpy(optr, sums.data(), (size_t) Q * sizeof(double));
    return out;
}

template <typename T>
void project_colmajor_inplace_impl(nb_mat2f_rw<T> Xz2d,
                                   nb::object C_opt,
                                   nb::object R_opt)
{
    const int N = (int) Xz2d.shape(0);
    const int Q = (int) Xz2d.shape(1);
    if (N <= 0 || Q <= 0)
        return;

    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;
    if (!C_opt.is_none() && !R_opt.is_none()) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in project_colmajor_inplace");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    } else {
        return;
    }

    T* Xptr = Xz2d.data();
    static thread_local AlignedBuffer<T> proj_tmp_tls;
    const size_t need = (size_t) p * (size_t) Q;
    if (proj_tmp_tls.n < need)
        proj_tmp_tls.allocate(need, 64);

    nb::gil_scoped_release nogil;
    project_panel_inplace<T>(Xptr, N, Q, Cptr, Rptr, p, proj_tmp_tls);
}

template <typename T>
static void project_target_block_inplace(T* G, int N, int L,
                                         const T* Cptr, const T* Rptr, int p,
                                         AlignedBuffer<T>& tmp_buf) {
    if (!G || p <= 0) return;
    if (tmp_buf.n < (size_t)p * (size_t)L) tmp_buf.allocate((size_t)p * (size_t)L, 64);
    T* tmp = tmp_buf.ptr;
    gemm_col_major_nn<T>(p, L, N, Rptr, p, G, N, tmp, p, T(1), T(0));
    gemm_col_major_nn<T>(N, L, p, Cptr, N, tmp, p, G, N, T(-1), T(1));
}

template <typename T>
static void compute_inv_resvar_from_geno_block(
    const T* Geno,
    int N,
    int L,
    double eps,
    nb::object C_opt,
    nb::object R_opt,
    nb::object gram_opt,
    T* inv_out)
{
    if (!Geno || !inv_out)
        throw std::runtime_error("Null buffer in compute_inv_resvar_from_geno_block");
    if (N <= 1 || L < 0)
        throw std::runtime_error("Invalid block dimensions in compute_inv_resvar_from_geno_block");

    const bool c_missing = C_opt.is_none();
    const bool r_missing = R_opt.is_none();
    if (c_missing != r_missing)
        throw std::runtime_error("C and R must be provided together for fused residual variances");

    const bool have_proj = !c_missing;
    int p = 0;
    const T* Rptr = nullptr;
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;
    nb_mat2c_ro<T> Garr;
    const T* Gptr = nullptr;
    if (have_proj) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N)
            throw std::runtime_error("C/R shape mismatch in fused residual variances");
        if (N - p <= 1)
            throw std::runtime_error("N_eff must be > 1 in fused residual variances");
        Rptr = Rarr.data();
        if (!gram_opt.is_none()) {
            Garr = nb::cast<nb_mat2c_ro<T>>(gram_opt);
            if ((int) Garr.shape(0) != p || (int) Garr.shape(1) != p)
                throw std::runtime_error("covariate Gram shape mismatch in fused residual variances");
            Gptr = Garr.data();
        }
    }

    const double proj_df = have_proj ? (double) (N - p - 1) : (double) (N - 1);
    if (proj_df <= 0.0)
        throw std::runtime_error("Projected residual df must be > 0 in fused residual variances");

    nb::gil_scoped_release nogil;

    static thread_local std::vector<double> ss_tls;
    if (ss_tls.size() < (size_t) L)
        ss_tls.resize((size_t) L);
    double* ss = ss_tls.data();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < L; ++j) {
        const T* col = Geno + (size_t) j * (size_t) N;
        double acc = 0.0;
        for (int i = 0; i < N; ++i) {
            const double x = (double) col[(size_t) i];
            acc += x * x;
        }
        ss[(size_t) j] = acc;
    }

    if (have_proj && p > 0) {
        static thread_local AlignedBuffer<T> proj_tls;
        const size_t need = (size_t) p * (size_t) L;
        if (proj_tls.n < need)
            proj_tls.allocate(need, 64);
        T* proj = proj_tls.ptr;
        gemm_col_major_nn<T>(p, L, N, Rptr, p, Geno, N, proj, p, T(1), T(0));

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < L; ++j) {
            const T* tcol = proj + (size_t) j * (size_t) p;
            double h = 0.0;
            for (int k = 0; k < p; ++k) {
                const double v = (double) tcol[(size_t) k];
                h += v * v;
            }
            double q = h;
            if (Gptr) {
                q = 0.0;
                for (int a = 0; a < p; ++a) {
                    const double ya = (double) tcol[(size_t) a];
                    double gy = 0.0;
                    const T* grow = Gptr + (size_t) a * (size_t) p;
                    for (int b = 0; b < p; ++b)
                        gy += (double) grow[(size_t) b] * (double) tcol[(size_t) b];
                    q += ya * gy;
                }
            }
            ss[(size_t) j] += q - 2.0 * h;
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < L; ++j) {
        const double var = ss[(size_t) j] / proj_df;
        double safe = 0.0;
        if (std::isnan(var)) safe = std::numeric_limits<double>::quiet_NaN();
        else safe = (var > eps ? var : eps);
        inv_out[(size_t) j] = (T) (1.0 / std::sqrt(safe));
    }
}

template <typename T>
nb::tuple precompute_residual_variances_bed_impl(
    const std::string& bed_prefix,
    const std::string& fam_path,
    int nsnps,
    int step_size,
    nb::object row_sel_obj,
    int ddof,
    double eps,
    bool compute_mu22,
    nb::object annot_all_obj,
    nb::object C_opt,
    nb::object R_opt,
    const std::string& impute_mode_str,
    nb::object impute_seed_obj)
{
    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (nsnps < 0 || nsnps > (int) M_total)
        throw std::runtime_error("nsnps is out of range in precompute_residual_variances_bed");
    if (step_size <= 0)
        throw std::runtime_error("step_size must be > 0 in precompute_residual_variances_bed");

    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    const int N_rows = (int) rows.size();
    if (N_rows <= 1)
        throw std::runtime_error("Too few rows in precompute_residual_variances_bed");

    bool have_proj = (!C_opt.is_none() && !R_opt.is_none());
    nb_mat2f_ro<T> Carr;
    nb_mat2f_ro<T> Rarr;
    const T* Cptr = nullptr;
    const T* Rptr = nullptr;
    int p = 0;
    if (have_proj) {
        Carr = nb::cast<nb_mat2f_ro<T>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<T>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N_rows || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N_rows)
            throw std::runtime_error("C/R shape mismatch in precompute_residual_variances_bed");
        if (N_rows - p <= 1)
            throw std::runtime_error("N_eff must be > 1 in precompute_residual_variances_bed");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    }

    const double proj_df = have_proj ? (double) (N_rows - p - 1) : (double) (N_rows - 1);
    if (proj_df <= 0.0)
        throw std::runtime_error("Projected residual df must be > 0 in precompute_residual_variances_bed");

    T* invp = nullptr;
    auto inv_out = make_owned_numpy_vec1<T>((size_t) nsnps, &invp);

    int B = 0;
    nb_mat2c_ro<T> Aarr;
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
        if (annot_all_obj.is_none())
            throw std::runtime_error("annot_all must be provided when compute_mu22=True");
        Aarr = nb::cast<nb_mat2c_ro<T>>(annot_all_obj);
        if ((int) Aarr.shape(0) != nsnps)
            throw std::runtime_error("annot_all shape mismatch in precompute_residual_variances_bed");
        annp = Aarr.data();
        B = (int) Aarr.shape(1);
        if (B <= 0)
            throw std::runtime_error("annot_all must have at least one column");
        S_mat.allocate((size_t) N_rows * (size_t) B, 64);
        std::fill(S_mat.ptr, S_mat.ptr + S_mat.n, 0.0);
        JPANEL = std::max(1, std::min(JPANEL, std::max(1, step_size)));
        X2_panel.allocate((size_t) N_rows * (size_t) JPANEL, 64);
        W_panel.allocate((size_t) JPANEL * (size_t) B, 64);
        S_sum.allocate((size_t) B * (size_t) B, 64);
        Msum.assign((size_t) B, 0.0);
    }

    AlignedBuffer<T> proj_tmp(have_proj ? (size_t) p * (size_t) std::min(step_size, nsnps) : 0, 64);

    {
        nb::gil_scoped_release nogil;
        for (int s = 0; s < nsnps; s += step_size) {
            check_for_interrupt();
            const int e = std::min(nsnps, s + step_size);
            const int L = e - s;
            if (L <= 0)
                continue;

            int N_blk = 0, L_blk = 0;
            static thread_local AlignedBuffer<T> Geno_tls;
            T* Geno = read_block_standardized_aligned<T>(bed_path, fam_path, s, e,
                                                         rows, ddof,
                                                         impute_mode, impute_seed,
                                                         Geno_tls, N_blk, L_blk);
            if (N_blk != N_rows || L_blk != L)
                throw std::runtime_error("Unexpected block dimensions in precompute_residual_variances_bed");

            if (have_proj)
                project_target_block_inplace<T>(Geno, N_blk, L_blk, Cptr, Rptr, p, proj_tmp);

            double* inv2_local = nullptr;
            static thread_local std::vector<double> inv2_local_tls;
            if (compute_mu22) {
                if (inv2_local_tls.size() < (size_t) L)
                    inv2_local_tls.resize((size_t) L);
                inv2_local = inv2_local_tls.data();
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < L; ++j) {
                const T* col = Geno + (size_t) j * (size_t) N_blk;
                double ss = 0.0;
                for (int i = 0; i < N_blk; ++i) {
                    const double x = (double) col[(size_t) i];
                    ss += x * x;
                }
                const double var = ss / proj_df;
                double safe = 0.0;
                if (std::isnan(var)) safe = std::numeric_limits<double>::quiet_NaN();
                else safe = (var > eps ? var : eps);
                const double invj = 1.0 / std::sqrt(safe);
                invp[(size_t) (s + j)] = (T) invj;
                if (inv2_local)
                    inv2_local[(size_t) j] = invj * invj;
            }

            if (compute_mu22) {
                const T* ann_blk = annp + (size_t) s * (size_t) B;
                for (int j = 0; j < L; ++j) {
                    const T* arow = ann_blk + (size_t) j * (size_t) B;
                    for (int k = 0; k < B; ++k)
                        Msum[(size_t) k] += (double) arow[(size_t) k];
                }
                for (int j0 = 0; j0 < L; j0 += JPANEL) {
                    const int jb = std::min(JPANEL, L - j0);
                    for (int jj = 0; jj < jb; ++jj) {
                        const int j = j0 + jj;
                        const T* gcol = Geno + (size_t) j * (size_t) N_blk;
                        double* xcol = X2_panel.ptr + (size_t) jj * (size_t) N_blk;
                        for (int i = 0; i < N_blk; ++i) {
                            const double y = (double) gcol[(size_t) i];
                            xcol[(size_t) i] = y * y;
                        }
                    }
                    for (int k = 0; k < B; ++k) {
                        double* wcol = W_panel.ptr + (size_t) k * (size_t) jb;
                        for (int jj = 0; jj < jb; ++jj) {
                            const int j = j0 + jj;
                            const double a = (double) ann_blk[(size_t) j * (size_t) B + (size_t) k];
                            wcol[(size_t) jj] = a * inv2_local[(size_t) j];
                        }
                    }
                    gemm_col_major_nn<double>(N_blk, B, jb, X2_panel.ptr, N_blk, W_panel.ptr, jb, S_mat.ptr, N_blk, 1.0, 1.0);
                }
            }
        }

        if (compute_mu22)
            gemm_col_major_tn<double>(B, B, N_rows, S_mat.ptr, N_rows, S_mat.ptr, N_rows, S_sum.ptr, B, 1.0, 0.0);
    }

    if (!compute_mu22)
        return nb::make_tuple(inv_out, nb::none());

    double* mu22_ptr = nullptr;
    auto mu22_out = make_owned_numpy_mat2c<double>((size_t) B, (size_t) B, &mu22_ptr);
    for (int a = 0; a < B; ++a) {
        for (int b = 0; b < B; ++b) {
            const double denom = (double) N_rows * Msum[(size_t) a] * Msum[(size_t) b];
            double v = 0.0;
            if (denom != 0.0) {
                v = S_sum.ptr[(size_t) a + (size_t) b * (size_t) B] / denom;
                if (!std::isfinite(v)) v = 0.0;
            }
            mu22_ptr[(size_t) a * (size_t) B + (size_t) b] = v;
        }
    }
    return nb::make_tuple(inv_out, mu22_out);
}

template <typename Tann>
nb::tuple compute_block_corrections_binary_impl(
    nb_mat2c_ro<double> meansq_raw,
    nb_mat2c_ro<double> mu22,
    nb_mat2c_ro<Tann> annot_all,
    int N,
    double d)
{
    const int M = (int) meansq_raw.shape(0);
    const int B = (int) meansq_raw.shape(1);
    if ((int) mu22.shape(0) != B || (int) mu22.shape(1) != B)
        throw std::runtime_error("mu22 shape mismatch");
    if ((int) annot_all.shape(0) != M || (int) annot_all.shape(1) != B)
        throw std::runtime_error("annot_all shape mismatch");
    if (N <= 1)
        throw std::runtime_error("N must be > 1 in compute_block_corrections_binary");

    const double* mptr = meansq_raw.data();
    const double* uptr = mu22.data();
    const Tann* aptr = annot_all.data();

    std::vector<double> M_a((size_t) B, 0.0);
    std::vector<double> S((size_t) B * (size_t) B, 0.0);
    for (int i = 0; i < M; ++i) {
        const Tann* arow = aptr + (size_t) i * (size_t) B;
        const double* mrow = mptr + (size_t) i * (size_t) B;
        for (int a = 0; a < B; ++a) {
            if (arow[(size_t) a] == (Tann) 0) continue;
            M_a[(size_t) a] += 1.0;
            double* Srow = S.data() + (size_t) a * (size_t) B;
            for (int b = 0; b < B; ++b)
                Srow[(size_t) b] += mrow[(size_t) b];
        }
    }

    double* R2p = nullptr;
    double* Rp = nullptr;
    double* Bp = nullptr;
    double* Dp = nullptr;
    auto R2_out = make_owned_numpy_mat2c<double>((size_t) B, (size_t) B, &R2p);
    auto rho2_out = make_owned_numpy_mat2c<double>((size_t) B, (size_t) B, &Rp);
    auto bias_out = make_owned_numpy_mat2c<double>((size_t) B, (size_t) B, &Bp);
    auto delta_out = make_owned_numpy_mat2c<double>((size_t) B, (size_t) B, &Dp);
    std::fill(R2p, R2p + (size_t) B * (size_t) B, 0.0);
    std::fill(Rp, Rp + (size_t) B * (size_t) B, 0.0);
    std::fill(Bp, Bp + (size_t) B * (size_t) B, 0.0);
    std::fill(Dp, Dp + (size_t) B * (size_t) B, 0.0);

    const double Nf = (double) N;
    const double denom_rho = Nf * (Nf - 1.0);
    for (int a = 0; a < B; ++a) {
        const double Ma = M_a[(size_t) a];
        if (Ma <= 0.0) continue;
        for (int b = 0; b < B; ++b) {
            const double Mb = M_a[(size_t) b];
            if (Mb <= 0.0) continue;
            const double S_ab = S[(size_t) a * (size_t) B + (size_t) b];
            if (S_ab == 0.0) continue;
            const double R2_ab = S_ab / (Ma * Mb);
            const double mu_ab = uptr[(size_t) a * (size_t) B + (size_t) b];
            const double rho2_ab = (d * d * R2_ab - Nf * mu_ab) / denom_rho;
            const double bias_ab = S_ab - (Ma * Mb * rho2_ab);
            const double delta_ab = mu_ab - (1.0 + 2.0 * rho2_ab);
            R2p[(size_t) a * (size_t) B + (size_t) b] = R2_ab;
            Rp[(size_t) a * (size_t) B + (size_t) b] = rho2_ab;
            Bp[(size_t) a * (size_t) B + (size_t) b] = bias_ab;
            Dp[(size_t) a * (size_t) B + (size_t) b] = delta_ab;
        }
    }
    return nb::make_tuple(R2_out, rho2_out, bias_out, delta_out);
}

void set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void) n;
#endif
}

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

NB_MODULE(gwldcore, m) {
    m.doc() = "C++ core for SUMMIT GW LD score (bed parser + BLAS-safe GEMMs + Mailman)";

    m.def("set_verbose", &set_verbose, nb::arg("enabled"));
    m.def("set_num_threads", &set_num_threads, nb::arg("n"));
    m.def("get_max_threads", &get_max_threads);
    m.def("prefetch_bed_block", &prefetch_bed_block_py,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("ahead_blocks") = 1);

    m.def("phase1_compute_Xz_bed_chunk", &phase1_compute_Xz_bed_chunk_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"),
          nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1, nb::arg("annot_blk"), nb::arg("inv_right"),
          nb::arg("v_start"), nb::arg("v_count"), nb::arg("kmax_hint"), nb::arg("rand_dist") = "rademacher",
          nb::arg("seed") = nb::none(), nb::arg("Xz2d_chunk"), nb::arg("bin_init_mask") = nb::none(), nb::arg("project_right") = false,
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none(),
          nb::arg("resvar_gram") = nb::none(),
          nb::arg("inv_out") = nb::none(), nb::arg("resvar_eps") = 1e-10);
    m.def("phase1_compute_Xz_bed_chunk", &phase1_compute_Xz_bed_chunk_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"),
          nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1, nb::arg("annot_blk"), nb::arg("inv_right"),
          nb::arg("v_start"), nb::arg("v_count"), nb::arg("kmax_hint"), nb::arg("rand_dist") = "rademacher",
          nb::arg("seed") = nb::none(), nb::arg("Xz2d_chunk"), nb::arg("bin_init_mask") = nb::none(), nb::arg("project_right") = false,
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none(),
          nb::arg("resvar_gram") = nb::none(),
          nb::arg("inv_out") = nb::none(), nb::arg("resvar_eps") = 1e-10);

    m.def("phase1_compute_Xz_bed_chunk_rowmajor", &phase1_compute_Xz_bed_chunk_rowmajor_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"),
          nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1, nb::arg("annot_blk"), nb::arg("inv_right"),
          nb::arg("v_start"), nb::arg("v_count"), nb::arg("kmax_hint"), nb::arg("rand_dist") = "rademacher",
          nb::arg("seed") = nb::none(), nb::arg("Xz2d_chunk"), nb::arg("project_right") = false,
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none(),
          nb::arg("resvar_gram") = nb::none(),
          nb::arg("inv_out") = nb::none(), nb::arg("resvar_eps") = 1e-10);
    m.def("phase1_compute_Xz_bed_chunk_rowmajor", &phase1_compute_Xz_bed_chunk_rowmajor_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"),
          nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1, nb::arg("annot_blk"), nb::arg("inv_right"),
          nb::arg("v_start"), nb::arg("v_count"), nb::arg("kmax_hint"), nb::arg("rand_dist") = "rademacher",
          nb::arg("seed") = nb::none(), nb::arg("Xz2d_chunk"), nb::arg("project_right") = false,
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none(),
          nb::arg("resvar_gram") = nb::none(),
          nb::arg("inv_out") = nb::none(), nb::arg("resvar_eps") = 1e-10);

    m.def("compute_col_sums_rowmajor", &compute_col_sums_rowmajor_impl<float>, nb::arg("Xz2d"));
    m.def("compute_col_sums_rowmajor", &compute_col_sums_rowmajor_impl<double>, nb::arg("Xz2d"));
    m.def("project_rowmajor_inplace_and_col_sums", &project_rowmajor_inplace_and_col_sums_impl<float>,
          nb::arg("Xz2d"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none());
    m.def("project_rowmajor_inplace_and_col_sums", &project_rowmajor_inplace_and_col_sums_impl<double>,
          nb::arg("Xz2d"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none());
    m.def("project_colmajor_inplace", &project_colmajor_inplace_impl<float>,
          nb::arg("Xz2d"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none());
    m.def("project_colmajor_inplace", &project_colmajor_inplace_impl<double>,
          nb::arg("Xz2d"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none());

    m.def("phase2_compute_XtXz_bed", &phase2_compute_XtXz_bed_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("nvecs"), nb::arg("vchunk"), nb::arg("Xz2d"), nb::arg("meansq"),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("N_denom") = 0,
          nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());
    m.def("phase2_compute_XtXz_bed", &phase2_compute_XtXz_bed_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("nvecs"), nb::arg("vchunk"), nb::arg("Xz2d"), nb::arg("meansq"),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("N_denom") = 0,
          nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());

    m.def("clear_phase1_csr_cache", &clear_phase1_csr_cache);

    m.def("precompute_residual_variances_bed", &precompute_residual_variances_bed_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("nsnps"), nb::arg("step_size"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("eps") = 1e-10, nb::arg("compute_mu22") = false, nb::arg("annot_all") = nb::none(),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());
    m.def("precompute_residual_variances_bed", &precompute_residual_variances_bed_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("nsnps"), nb::arg("step_size"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("eps") = 1e-10, nb::arg("compute_mu22") = false, nb::arg("annot_all") = nb::none(),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());

    m.def("phase2_accum_XtXz_bed", &phase2_accum_XtXz_bed_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("tile_nvecs"), nb::arg("Xz2d"), nb::arg("meansq_accum"),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("N_denom") = 0,
          nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());
    m.def("phase2_accum_XtXz_bed", &phase2_accum_XtXz_bed_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("tile_nvecs"), nb::arg("Xz2d"), nb::arg("meansq_accum"),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("N_denom") = 0,
          nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());

    m.def("phase2_accum_XtXz_bed_mailman", &phase2_accum_XtXz_bed_mailman_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("tile_nvecs"), nb::arg("Xz2d"), nb::arg("meansq_accum"),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("N_denom") = 0, nb::arg("impute_seed") = nb::none());
    m.def("phase2_accum_XtXz_bed_mailman", &phase2_accum_XtXz_bed_mailman_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("tile_nvecs"), nb::arg("Xz2d"), nb::arg("meansq_accum"),
          nb::arg("C") = nb::none(), nb::arg("R") = nb::none(), nb::arg("N_denom") = 0, nb::arg("impute_seed") = nb::none());

    m.def("phase2_accum_XtXz_bed_mailman_rowmajor", &phase2_accum_XtXz_bed_mailman_rm_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("tile_nvecs"), nb::arg("Xz2d"), nb::arg("meansq_accum"),
          nb::arg("sum_Xz"), nb::arg("N_denom") = 0, nb::arg("impute_seed") = nb::none());
    m.def("phase2_accum_XtXz_bed_mailman_rowmajor", &phase2_accum_XtXz_bed_mailman_rm_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_left"), nb::arg("tile_nvecs"), nb::arg("Xz2d"), nb::arg("meansq_accum"),
          nb::arg("sum_Xz"), nb::arg("N_denom") = 0, nb::arg("impute_seed") = nb::none());

    m.def("compute_block_corrections_binary", &compute_block_corrections_binary_impl<float>,
          nb::arg("meansq_raw"), nb::arg("mu22"), nb::arg("annot_all"), nb::arg("N"), nb::arg("d"));
    m.def("compute_block_corrections_binary", &compute_block_corrections_binary_impl<double>,
          nb::arg("meansq_raw"), nb::arg("mu22"), nb::arg("annot_all"), nb::arg("N"), nb::arg("d"));

    m.def("apply_grm_bed_panel", &apply_grm_bed_panel_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("nsnps"), nb::arg("step_size"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_all"), nb::arg("panel_in"), nb::arg("panel_out"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none(),
          nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());
    m.def("apply_grm_bed_panel", &apply_grm_bed_panel_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("nsnps"), nb::arg("step_size"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_all"), nb::arg("panel_in"), nb::arg("panel_out"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none(),
          nb::arg("impute_mode") = "hwe", nb::arg("impute_seed") = nb::none());

    m.def("apply_grm_bed_panel_mailman", &apply_grm_bed_panel_mailman_impl<float>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("nsnps"), nb::arg("step_size"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_all"), nb::arg("panel_in"), nb::arg("panel_out"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none(),
          nb::arg("impute_seed") = nb::none());
    m.def("apply_grm_bed_panel_mailman", &apply_grm_bed_panel_mailman_impl<double>,
          nb::arg("bed_prefix"), nb::arg("fam_path"), nb::arg("nsnps"), nb::arg("step_size"), nb::arg("row_sel") = nb::none(),
          nb::arg("ddof") = 1, nb::arg("inv_all"), nb::arg("panel_in"), nb::arg("panel_out"), nb::arg("C") = nb::none(), nb::arg("R") = nb::none(),
          nb::arg("impute_seed") = nb::none());
}
