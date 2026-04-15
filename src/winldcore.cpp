#include "nb_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <list>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

#if defined(__linux__)
  #include <unistd.h>
#endif

#include "blas_compat.hpp"
#include "genotype.hpp"

static inline void check_for_interrupt() { nb_check_for_interrupt(); }

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
        if (ptr) {
            std::free(ptr);
            ptr = nullptr;
            n = 0;
        }
    }
    ~AlignedBuffer() { free(); }
};

static std::atomic<bool> g_verbose{false};
static inline bool verbose_enabled() { return g_verbose.load(std::memory_order_relaxed); }
static inline void set_verbose(bool v) { g_verbose.store(v, std::memory_order_relaxed); }

static std::atomic<int> g_progress_total{0};
static std::atomic<int> g_progress_done{0};
static std::atomic<bool> g_progress_active{false};

static inline void progress_begin(int total) {
    g_progress_total.store(std::max(0, total), std::memory_order_relaxed);
    g_progress_done.store(0, std::memory_order_relaxed);
    g_progress_active.store(true, std::memory_order_relaxed);
}

static inline void progress_step() {
    g_progress_done.fetch_add(1, std::memory_order_relaxed);
}

static inline void progress_end(bool completed) {
    if (completed) {
        g_progress_done.store(g_progress_total.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    }
    g_progress_active.store(false, std::memory_order_relaxed);
}

static inline int get_progress_total() {
    return g_progress_total.load(std::memory_order_relaxed);
}

static inline int get_progress_done() {
    return g_progress_done.load(std::memory_order_relaxed);
}

static inline bool progress_active() {
    return g_progress_active.load(std::memory_order_relaxed);
}

struct ProgressScope {
    bool completed = false;
    explicit ProgressScope(int total) { progress_begin(total); }
    ~ProgressScope() { progress_end(completed); }
};

static inline int clampi(int x, int lo, int hi) {
    return std::max(lo, std::min(hi, x));
}

static inline int ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}

static int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static size_t available_memory_bytes() {
#if defined(__linux__)
    const long pages = ::sysconf(_SC_AVPHYS_PAGES);
    const long page_sz = ::sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_sz > 0) {
        const long double v = (long double) pages * (long double) page_sz;
        if (v > 0.0L) {
            const long double cap = (long double) std::numeric_limits<size_t>::max();
            return (size_t) std::min(v, cap);
        }
    }
#endif
    return 0;
}

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

static const std::vector<int>& parse_row_sel(nb::object row_sel_obj, int64_t N_total) {
    return parse_row_sel_nb(std::move(row_sel_obj), N_total);
}

static inline ImputeMode parse_impute_mode(const std::string& s) {
    if (s == "hwe") return ImputeMode::Hwe;
    if (s == "mean") return ImputeMode::Mean;
    throw std::runtime_error("impute_mode must be 'mean' or 'hwe'");
}

static void set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

static inline size_t cache_key(int start, int end) {
    return ((uint64_t)(uint32_t)start << 32) ^ (uint64_t)(uint32_t)end;
}

static void project_block_inplace(std::vector<double>& G,
                                  int N,
                                  int L,
                                  const double* Cptr,
                                  const double* Rptr,
                                  int p,
                                  AlignedBuffer<double>& tmp)
{
    if (p <= 0 || N <= 0 || L <= 0) return;
    const size_t need = (size_t)p * (size_t)L;
    if (tmp.n < need) tmp.allocate(need, 64);
    gemm_col_major_nn<double>(p, L, N, Rptr, p, G.data(), N, tmp.ptr, p, 1.0, 0.0);
    gemm_col_major_nn<double>(N, L, p, Cptr, N, tmp.ptr, p, G.data(), N, -1.0, 1.0);
}

static void restandardize_cols_inplace(std::vector<double>& G, int N, int L)
{
    if (N <= 0 || L <= 0) return;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < L; ++j) {
        double* col = G.data() + (size_t)j * (size_t)N;
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += col[(size_t)i];
        const double mean = sum / (double)N;

        double ss = 0.0;
        for (int i = 0; i < N; ++i) {
            const double d = col[(size_t)i] - mean;
            ss += d * d;
        }
        double inv_sd = 1.0;
        if (ss > 0.0 && std::isfinite(ss)) {
            const double var = ss / (double)N;
            if (var > 0.0 && std::isfinite(var)) inv_sd = 1.0 / std::sqrt(var);
        }
        for (int i = 0; i < N; ++i) col[(size_t)i] = (col[(size_t)i] - mean) * inv_sd;
    }
}

struct PreparedPanel {
    int start = 0;   // global SNP index (genome-wide, BIM order)
    int end = 0;     // exclusive
    int N = 0;
    std::vector<double> X; // column-major N x L
    size_t bytes() const { return X.size() * sizeof(double); }
    int L() const { return end - start; }
};

class PreparedPanelCache {
public:
    PreparedPanelCache(std::string bed_path,
                       std::string fam_path,
                       const std::vector<int>& rows,
                       const double* Cptr,
                       const double* Rptr,
                       int p,
                       ImputeMode impute_mode,
                       uint64_t impute_seed,
                       size_t cap_bytes)
        : bed_path_(std::move(bed_path)),
          fam_path_(std::move(fam_path)),
          rows_(rows),
          Cptr_(Cptr),
          Rptr_(Rptr),
          p_(p),
          have_proj_(Cptr != nullptr && Rptr != nullptr && p > 0),
          impute_mode_(impute_mode),
          impute_seed_(impute_seed),
          cap_bytes_(cap_bytes) {}

    std::shared_ptr<PreparedPanel> get(int start, int end) {
        const size_t key = cache_key(start, end);
        auto it = map_.find(key);
        if (it != map_.end()) {
            touch(it);
            return it->second.panel;
        }

        auto panel = load_panel(start, end);
        const size_t need = panel->bytes();
        if (cap_bytes_ == 0 || need > cap_bytes_) return panel;

        while (bytes_ + need > cap_bytes_ && !lru_.empty()) {
            const size_t old = lru_.back();
            auto jt = map_.find(old);
            if (jt != map_.end()) {
                bytes_ -= jt->second.panel->bytes();
                lru_.pop_back();
                map_.erase(jt);
            } else {
                lru_.pop_back();
            }
        }
        lru_.push_front(key);
        map_[key] = Entry{panel, lru_.begin()};
        bytes_ += need;
        return panel;
    }

    void clear() {
        map_.clear();
        lru_.clear();
        bytes_ = 0;
    }

private:
    struct Entry {
        std::shared_ptr<PreparedPanel> panel;
        std::list<size_t>::iterator it;
    };

    std::shared_ptr<PreparedPanel> load_panel(int start, int end) {
        if (start >= end) throw std::runtime_error("Invalid panel interval");
        auto panel = std::make_shared<PreparedPanel>();
        panel->start = start;
        panel->end = end;
        prefetch_bed_block(bed_path_, fam_path_, start, end, 1);
        int N = 0, L = 0;
        read_block_standardized<double>(bed_path_, fam_path_, start, end,
                                        rows_, 0,
                                        impute_mode_, impute_seed_,
                                        panel->X, N, L);
        if (L != (end - start)) throw std::runtime_error("Prepared panel length mismatch");
        panel->N = N;
        if (have_proj_) {
            project_block_inplace(panel->X, N, L, Cptr_, Rptr_, p_, proj_tmp_);
            restandardize_cols_inplace(panel->X, N, L);
        } else if (impute_mode_ == ImputeMode::Mean) {
            // Mean-imputation path must match the legacy windowed estimator,
            // which standardizes after imputation across all selected rows.
            restandardize_cols_inplace(panel->X, N, L);
        }
        return panel;
    }

    void touch(std::unordered_map<size_t, Entry>::iterator it) {
        lru_.erase(it->second.it);
        lru_.push_front(it->first);
        it->second.it = lru_.begin();
    }

    std::string bed_path_;
    std::string fam_path_;
    const std::vector<int>& rows_;
    const double* Cptr_ = nullptr;
    const double* Rptr_ = nullptr;
    int p_ = 0;
    bool have_proj_ = false;
    ImputeMode impute_mode_ = ImputeMode::Mean;
    uint64_t impute_seed_ = 0;
    size_t cap_bytes_ = 0;
    size_t bytes_ = 0;
    std::list<size_t> lru_;
    std::unordered_map<size_t, Entry> map_;
    AlignedBuffer<double> proj_tmp_;
};

static inline void unbiased_r2_inplace(double* buf, size_t n, int N_rows)
{
    const double denom = (N_rows > 2) ? (double)(N_rows - 2) : (double)std::max(1, N_rows);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i) {
        const double r = buf[(size_t)i];
        double r2 = r * r;
        r2 -= (1.0 - r2) / denom;
        buf[(size_t)i] = r2;
    }
}

static void accum_cross_panels(int chr_global_start,
                               const PreparedPanel& left,
                               const PreparedPanel& right,
                               const double* annot_ptr,
                               int annot_ld,
                               double* ld_ptr,
                               int ld_ld,
                               int B,
                               int N_rows,
                               AlignedBuffer<double>& cross)
{
    const int Ll = left.L();
    const int Lr = right.L();
    if (Ll <= 0 || Lr <= 0) return;

    const int left_local = left.start - chr_global_start;
    const int right_local = right.start - chr_global_start;
    const size_t need = (size_t)Ll * (size_t)Lr;
    if (cross.n < need) cross.allocate(need, 64);

    gemm_col_major_tn<double>(Ll, Lr, N_rows,
                              left.X.data(), N_rows,
                              right.X.data(), N_rows,
                              cross.ptr, Ll,
                              1.0 / (double)N_rows,
                              0.0);
    unbiased_r2_inplace(cross.ptr, need, N_rows);

    gemm_col_major_nn<double>(Ll, B, Lr,
                              cross.ptr, Ll,
                              annot_ptr + (size_t)right_local, annot_ld,
                              ld_ptr + (size_t)left_local, ld_ld,
                              1.0, 1.0);

    gemm_col_major_tn<double>(Lr, B, Ll,
                              cross.ptr, Ll,
                              annot_ptr + (size_t)left_local, annot_ld,
                              ld_ptr + (size_t)right_local, ld_ld,
                              1.0, 1.0);
}

static void accum_left_from_right_panel(int chr_global_start,
                                        const PreparedPanel& left,
                                        const PreparedPanel& right,
                                        const double* annot_ptr,
                                        int annot_ld,
                                        double* ld_ptr,
                                        int ld_ld,
                                        int B,
                                        int N_rows,
                                        AlignedBuffer<double>& cross)
{
    const int Ll = left.L();
    const int Lr = right.L();
    if (Ll <= 0 || Lr <= 0) return;

    const int left_local = left.start - chr_global_start;
    const int right_local = right.start - chr_global_start;
    const size_t need = (size_t)Ll * (size_t)Lr;
    if (cross.n < need) cross.allocate(need, 64);

    gemm_col_major_tn<double>(Ll, Lr, N_rows,
                              left.X.data(), N_rows,
                              right.X.data(), N_rows,
                              cross.ptr, Ll,
                              1.0 / (double)N_rows,
                              0.0);
    unbiased_r2_inplace(cross.ptr, need, N_rows);

    gemm_col_major_nn<double>(Ll, B, Lr,
                              cross.ptr, Ll,
                              annot_ptr + (size_t)right_local, annot_ld,
                              ld_ptr + (size_t)left_local, ld_ld,
                              1.0, 1.0);
}

static std::vector<int> compute_block_left(const int64_t* bp, int m, double ld_wind_kb)
{
    std::vector<int> left((size_t)m, 0);
    int j = 0;
    for (int i = 0; i < m; ++i) {
        const double ci = (double)bp[(size_t)i] / 1000.0;
        while (j < i && (ci - (double)bp[(size_t)j] / 1000.0) > ld_wind_kb) ++j;
        left[(size_t)i] = j;
    }
    return left;
}

static int auto_panel_cols(int N_rows, int chunk_size, int panel_cols, size_t cache_bytes)
{
    if (panel_cols > 0) return std::max(1, std::min(panel_cols, chunk_size));

    size_t target_bytes = 0;
    if (const char* s = std::getenv("SUMMIT_WIN_PANEL_MB")) {
        const long long mb = std::atoll(s);
        if (mb > 0) target_bytes = (size_t)mb * 1024ULL * 1024ULL;
    }
    if (target_bytes == 0) {
        const size_t min_target = 256ULL * 1024ULL * 1024ULL;
        const size_t max_target = 1024ULL * 1024ULL * 1024ULL;
        if (cache_bytes > 0) target_bytes = cache_bytes / 16ULL;
        else target_bytes = 512ULL * 1024ULL * 1024ULL;
        if (target_bytes < min_target) target_bytes = min_target;
        if (target_bytes > max_target) target_bytes = max_target;
    }

    const size_t bytes_per_col = (size_t)std::max(1, N_rows) * sizeof(double);
    int cols = (int)(target_bytes / bytes_per_col);
    cols = std::min(cols, chunk_size);
    cols = std::min(cols, 2048);
    if (cols >= 32) {
        cols = (cols / 32) * 32;
        if (cols < 32) cols = 32;
    }
    if (cols < 1) cols = 1;
    if (cols > chunk_size) cols = chunk_size;
    return cols;
}

static size_t auto_cache_bytes(int cache_mb)
{
    if (cache_mb < 0) {
        if (const char* s = std::getenv("SUMMIT_WIN_CACHE_MB")) {
            const long long v = std::atoll(s);
            if (v <= 0) return 0;
            return (size_t)v * 1024ULL * 1024ULL;
        }

        const size_t avail = available_memory_bytes();
        if (avail == 0) {
            return 4ULL * 1024ULL * 1024ULL * 1024ULL;
        }

        const size_t one_gib = 1024ULL * 1024ULL * 1024ULL;
        const size_t thirty_two_gib = 32ULL * one_gib;
        const size_t half_avail = avail / 2ULL;
        size_t target = avail / 4ULL;
        if (target < one_gib) target = one_gib;
        if (target > thirty_two_gib) target = thirty_two_gib;
        if (target > half_avail && half_avail > 0) target = half_avail;
        return target;
    }
    if (cache_mb == 0) return 0;
    return (size_t)cache_mb * 1024ULL * 1024ULL;
}

static void accum_logic_tile_self(PreparedPanelCache& cache,
                                  int chr_global_start,
                                  int local_start,
                                  int local_end,
                                  int panel_cols,
                                  const double* annot_ptr,
                                  int annot_ld,
                                  double* ld_ptr,
                                  int ld_ld,
                                  int B,
                                  int N_rows,
                                  AlignedBuffer<double>& cross)
{
    int rp_idx = 0;
    for (int rp0 = local_start; rp0 < local_end; rp0 += panel_cols, ++rp_idx) {
        if ((rp_idx & 7) == 0) check_for_interrupt();
        const int rp1 = std::min(local_end, rp0 + panel_cols);
        auto right = cache.get(chr_global_start + rp0, chr_global_start + rp1);

        // Diagonal block: one pass updates this panel from itself.
        accum_left_from_right_panel(chr_global_start, *right, *right,
                                    annot_ptr, annot_ld, ld_ptr, ld_ld,
                                    B, N_rows, cross);

        // Strictly lower-triangular off-diagonal blocks: compute once and update both sides.
        for (int lp0 = local_start; lp0 < rp0; lp0 += panel_cols) {
            const int lp1 = std::min(local_end, lp0 + panel_cols);
            auto left = cache.get(chr_global_start + lp0, chr_global_start + lp1);
            accum_cross_panels(chr_global_start, *left, *right,
                               annot_ptr, annot_ld, ld_ptr, ld_ld,
                               B, N_rows, cross);
        }
    }
}

static void accum_logic_tile_cross(PreparedPanelCache& cache,
                                   int chr_global_start,
                                   int left_local_start,
                                   int left_local_end,
                                   int right_local_start,
                                   int right_local_end,
                                   int panel_cols,
                                   const double* annot_ptr,
                                   int annot_ld,
                                   double* ld_ptr,
                                   int ld_ld,
                                   int B,
                                   int N_rows,
                                   AlignedBuffer<double>& cross)
{
    if (left_local_start >= left_local_end || right_local_start >= right_local_end) return;
    int rp_idx = 0;
    for (int rp0 = right_local_start; rp0 < right_local_end; rp0 += panel_cols, ++rp_idx) {
        if ((rp_idx & 7) == 0) check_for_interrupt();
        const int rp1 = std::min(right_local_end, rp0 + panel_cols);
        auto right = cache.get(chr_global_start + rp0, chr_global_start + rp1);
        for (int lp0 = left_local_start; lp0 < left_local_end; lp0 += panel_cols) {
            const int lp1 = std::min(left_local_end, lp0 + panel_cols);
            auto left = cache.get(chr_global_start + lp0, chr_global_start + lp1);
            accum_cross_panels(chr_global_start, *left, *right,
                               annot_ptr, annot_ld, ld_ptr, ld_ld,
                               B, N_rows, cross);
        }
    }
}

static void accum_all_pairs_interval(PreparedPanelCache& cache,
                                     int chr_global_start,
                                     int interval_len,
                                     int logic_chunk_size,
                                     int panel_cols,
                                     const double* annot_ptr,
                                     int annot_ld,
                                     double* ld_ptr,
                                     int ld_ld,
                                     int B,
                                     int N_rows)
{
    const int ntiles = ceil_div_i(interval_len, logic_chunk_size);
    AlignedBuffer<double> cross;
    for (int t = 0; t < ntiles; ++t) {
        check_for_interrupt();
        const int t0 = t * logic_chunk_size;
        const int t1 = std::min(interval_len, t0 + logic_chunk_size);
        accum_logic_tile_self(cache, chr_global_start, t0, t1, panel_cols,
                              annot_ptr, annot_ld, ld_ptr, ld_ld, B, N_rows, cross);
        for (int a = 0; a < t; ++a) {
            const int a0 = a * logic_chunk_size;
            const int a1 = std::min(interval_len, a0 + logic_chunk_size);
            accum_logic_tile_cross(cache, chr_global_start,
                                   a0, a1,
                                   t0, t1,
                                   panel_cols,
                                   annot_ptr, annot_ld, ld_ptr, ld_ld, B, N_rows, cross);
        }
        progress_step();
    }
}

static nb_numpy_mat2f<double> compute_windowed_ld_chr_impl(
    const std::string& bed_prefix,
    const std::string& fam_path,
    int chr_start,
    int chr_end,
    nb_vec1_ro<int64_t> bp,
    nb_mat2f_ro<double> annot_chr,
    double ld_wind_kb,
    int chunk_size,
    nb::object row_sel_obj,
    nb::object C_opt,
    nb::object R_opt,
    const std::string& impute_mode_str,
    nb::object impute_seed_obj,
    int panel_cols,
    int cache_mb)
{
    if (chr_end < chr_start)
        throw std::runtime_error("chr_end must be >= chr_start");
    const int m = chr_end - chr_start;
    if (m <= 0)
        throw std::runtime_error("Chromosome slice is empty");
    if (chunk_size <= 0)
        throw std::runtime_error("chunk_size must be > 0");

    const int ntiles = ceil_div_i(m, chunk_size);
    ProgressScope progress(ntiles);

    if ((int) bp.shape(0) != m)
        throw std::runtime_error("bp length mismatch in compute_windowed_ld_chr");
    const int64_t* bp_ptr = bp.data();
    for (int i = 1; i < m; ++i) {
        if (bp_ptr[(size_t) i] < bp_ptr[(size_t) (i - 1)])
            throw std::runtime_error("BP not sorted within chromosome block. Sort your .bim by CHR+BP.");
    }

    if ((int) annot_chr.shape(0) != m)
        throw std::runtime_error("annot_chr shape mismatch in compute_windowed_ld_chr");
    const int B = (int) annot_chr.shape(1);
    if (B <= 0)
        throw std::runtime_error("annot_chr must have at least one column");
    const double* annot_ptr = annot_chr.data();

    const std::string bed_path = bed_prefix + ".bed";
    const int64_t N_total = count_lines_cached(fam_path);
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    const int N_rows = (int) rows.size();
    if (N_rows <= 2)
        throw std::runtime_error("Too few rows for windowed LD score computation");

    const ImputeMode impute_mode = parse_impute_mode(impute_mode_str);
    const uint64_t impute_seed = impute_seed_obj.is_none() ? 0ULL : nb::cast<uint64_t>(impute_seed_obj);

    const double* Cptr = nullptr;
    const double* Rptr = nullptr;
    int p = 0;
    nb_mat2f_ro<double> Carr;
    nb_mat2f_ro<double> Rarr;
    if (!C_opt.is_none() && !R_opt.is_none()) {
        Carr = nb::cast<nb_mat2f_ro<double>>(C_opt);
        Rarr = nb::cast<nb_mat2f_ro<double>>(R_opt);
        p = (int) Carr.shape(1);
        if ((int) Carr.shape(0) != N_rows || (int) Rarr.shape(0) != p || (int) Rarr.shape(1) != N_rows)
            throw std::runtime_error("C/R shape mismatch in compute_windowed_ld_chr");
        Cptr = Carr.data();
        Rptr = Rarr.data();
    }

    const size_t cache_bytes = auto_cache_bytes(cache_mb);
    panel_cols = auto_panel_cols(N_rows, chunk_size, panel_cols, cache_bytes);

    double* ld_ptr = nullptr;
    auto ld_out = make_owned_numpy_mat2f<double>((size_t) m, (size_t) B, &ld_ptr);
    std::fill(ld_ptr, ld_ptr + (size_t) m * (size_t) B, 0.0);

    const std::vector<int> left = compute_block_left(bp_ptr, m, ld_wind_kb);
    int first_pos = -1;
    for (int i = 0; i < m; ++i) {
        if (left[(size_t) i] > 0) {
            first_pos = i;
            break;
        }
    }
    int b0 = (((first_pos >= 0) ? first_pos : m) + chunk_size - 1) / chunk_size;
    b0 *= chunk_size;

    std::vector<int> block_sizes((size_t) m, 0);
    for (int i = 0; i < m; ++i) {
        const int span = i - left[(size_t) i];
        block_sizes[(size_t) i] = ((span + chunk_size - 1) / chunk_size) * chunk_size;
    }

    PreparedPanelCache cache(bed_path, fam_path, rows, Cptr, Rptr, p,
                             impute_mode, impute_seed, cache_bytes);

    nb::gil_scoped_release nogil;

    if (verbose_enabled()) {
        const double cache_gib = (double)cache_bytes / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
                     "[winldcore] chr [%d:%d) m=%d chunk=%d panel=%d cache=%.2f GiB prefix=%d\n",
                     chr_start, chr_end, m, chunk_size, panel_cols, cache_gib, b0);
    }

    if (b0 >= m) {
        accum_all_pairs_interval(cache, chr_start, m, chunk_size, panel_cols,
                                 annot_ptr, m, ld_ptr, m, B, N_rows);
        progress.completed = true;
        return ld_out;
    }

    accum_all_pairs_interval(cache, chr_start, b0, chunk_size, panel_cols,
                             annot_ptr, m, ld_ptr, m, B, N_rows);

    const int prefix_tiles = b0 / chunk_size;
    AlignedBuffer<double> cross;
    for (int t = prefix_tiles; t < ntiles; ++t) {
        check_for_interrupt();
        const int t0 = t * chunk_size;
        const int t1 = std::min(m, t0 + chunk_size);
        const int left_tiles = block_sizes[(size_t) t0] / chunk_size;
        const int a_start_tile = std::max(0, t - left_tiles);
        for (int a = a_start_tile; a < t; ++a) {
            const int a0 = a * chunk_size;
            const int a1 = std::min(m, a0 + chunk_size);
            accum_logic_tile_cross(cache, chr_start,
                                   a0, a1,
                                   t0, t1,
                                   panel_cols,
                                   annot_ptr, m, ld_ptr, m, B, N_rows, cross);
        }
        accum_logic_tile_self(cache, chr_start,
                              t0, t1,
                              panel_cols,
                              annot_ptr, m, ld_ptr, m, B, N_rows, cross);
        progress_step();
    }

    progress.completed = true;
    return ld_out;
}

static nb_numpy_vec1<double> compute_maf_bed_impl(const std::string& bed_prefix,
                                                  const std::string& fam_path,
                                                  int nsnps,
                                                  int step_size,
                                                  nb::object row_sel_obj)
{
    if (nsnps < 0)
        throw std::runtime_error("nsnps must be >= 0");
    if (step_size <= 0)
        throw std::runtime_error("step_size must be > 0");

    const std::string bed_path = bed_prefix + ".bed";
    const int64_t N_total = count_lines_cached(fam_path);
    const std::vector<int>& rows = parse_row_sel(row_sel_obj, N_total);
    if (rows.empty())
        throw std::runtime_error("No rows selected in compute_maf_bed");

    double* mp = nullptr;
    auto maf_out = make_owned_numpy_vec1<double>((size_t) nsnps, &mp);

    nb::gil_scoped_release nogil;
    std::vector<double> maf_blk;
    for (int s = 0; s < nsnps; s += step_size) {
        check_for_interrupt();
        const int e = std::min(nsnps, s + step_size);
        compute_maf_block(bed_path, fam_path, s, e, rows, maf_blk);
        std::memcpy(mp + (size_t) s, maf_blk.data(), (size_t) (e - s) * sizeof(double));
    }
    return maf_out;
}

NB_MODULE(winldcore, m) {
    m.doc() = "C++ core for deterministic windowed LD scores";
    m.def("set_verbose", &set_verbose, nb::arg("enabled"));
    m.def("set_num_threads", &set_num_threads, nb::arg("n"));
    m.def("get_max_threads", &get_max_threads);
    m.def("get_progress_total", &get_progress_total);
    m.def("get_progress_done", &get_progress_done);
    m.def("progress_active", &progress_active);
    m.def("compute_windowed_ld_chr", &compute_windowed_ld_chr_impl,
          nb::arg("bed_prefix"),
          nb::arg("fam_path"),
          nb::arg("chr_start"),
          nb::arg("chr_end"),
          nb::arg("bp"),
          nb::arg("annot_chr"),
          nb::arg("ld_wind_kb"),
          nb::arg("chunk_size"),
          nb::arg("row_sel") = nb::none(),
          nb::arg("C") = nb::none(),
          nb::arg("R") = nb::none(),
          nb::arg("impute_mode") = "mean",
          nb::arg("impute_seed") = nb::none(),
          nb::arg("panel_cols") = 0,
          nb::arg("cache_mb") = -1);
    m.def("compute_maf_bed", &compute_maf_bed_impl,
          nb::arg("bed_prefix"),
          nb::arg("fam_path"),
          nb::arg("nsnps"),
          nb::arg("step_size"),
          nb::arg("row_sel") = nb::none());
}
