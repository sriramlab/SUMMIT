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

namespace py = pybind11;

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
static void read_block_standardized(const std::string &bed_path,
                                    const std::string &fam_path,
                                    int blk_start, int blk_end,
                                    const std::vector<int> &rows, // selected individuals
                                    int ddof,
                                    std::vector<T> &Geno, // (N x L), col-major
                                    int &N, int &L) {
    const int64_t N_total = count_lines(fam_path);
    if (N_total <= 0) throw std::runtime_error("FAM has zero rows: " + fam_path);
    const int64_t s = blk_start, e = blk_end;
    if (e <= s) { N = (int)rows.size(); L = 0; Geno.clear(); return; }
    L = (int)(e - s);
    N = (int)rows.size();

    std::ifstream bed(bed_path, std::ios::binary);
    if (!bed) throw std::runtime_error("Failed to open bed: " + bed_path);

    // header 3 bytes
    bed.seekg(0, std::ios::beg);
    unsigned char magic[3];
    bed.read(reinterpret_cast<char*>(magic), 3);
    // magic[2] == 1 => SNP-major; we assume SNP-major

    const int nbytes_per_snp = (int)ceil_div((std::size_t)N_total, (std::size_t)4);
    const std::streamoff offset = 3 + static_cast<std::streamoff>(s) * nbytes_per_snp;
    bed.seekg(offset, std::ios::beg);

    Geno.assign((size_t)N * (size_t)L, T(0));
    std::vector<unsigned char> line(nbytes_per_snp);
    std::vector<T> full_col(N_total);

    for (int col = 0; col < L; ++col) {
        bed.read(reinterpret_cast<char*>(line.data()), nbytes_per_snp);
        if (!bed) throw std::runtime_error("BED read failed at SNP " + std::to_string(s + col));
        decode_bed_snp_to_vector(line.data(), (int)N_total, full_col);

        auto [mu, sd] = nan_mean_std(full_col, rows, ddof);

        // fill Geno[:, col] in column-major with standardized values; NaN→0 after standardization
        T *dst = Geno.data() + (size_t)col * (size_t)N;
        for (int i = 0; i < N; ++i) {
            T x = full_col[rows[i]];
            if (std::isnan(x)) {
                dst[i] = T(0);
            } else {
                dst[i] = (x - mu) / sd;
            }
        }
    }
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

// -------------------------- Phase 1: compute_Xz ------------------------------
template <typename T>
void phase1_compute_Xz_bed_chunk_impl(const std::string &bed_prefix,
                                      const std::string &fam_path,
                                      int blk_start, int blk_end,
                                      py::object row_sel_obj,
                                      int ddof,
                                      py::array_t<T, py::array::c_style | py::array::forcecast> annot_blk, // (L x B)
                                      py::array_t<T, py::array::c_style | py::array::forcecast> inv_right,  // (L,)
                                      int v_start,            // global V offset
                                      int v_count,            // V cols in this chunk
                                      int kmax_hint,          // ALWAYS trusted
                                      const std::string &rand_dist,
                                      py::object seed_obj,    // None or int
                                      py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d_chunk, // (N x (B*v_count))
                                      bool /*project_right*/ = false,
                                      py::object /*C_opt*/ = py::none(),
                                      py::object /*R_opt*/ = py::none()) {
    // If Kmax==0 for this block, nothing contributes; leave Xz2d_chunk unchanged.
    if (kmax_hint == 0) return;

    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines(fam_path);
    const int64_t M_total = count_lines(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    if (kmax_hint < 0) throw std::runtime_error("kmax_hint must be >= 0");

    // Selected rows (individuals)
    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);

    // Read & standardize block -> Geno (N x L), col-major
    int N = 0, L = 0;
    std::vector<T> Geno; // (N x L)
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // annot_blk: (L x B), row-major; inv_right: (L,)
    auto Ainfo = annot_blk.request();
    auto Iinfo = inv_right.request();
    if (Ainfo.ndim != 2) throw std::runtime_error("annot_blk must be 2D (L x B)");
    if (Iinfo.ndim != 1) throw std::runtime_error("inv_right must be 1D (L,)");
    const int B = (int)Ainfo.shape[1];
    if ((int)Ainfo.shape[0] != L) throw std::runtime_error("annot_blk.shape[0] != L");
    if ((int)Iinfo.shape[0] != L) throw std::runtime_error("inv_right.shape[0] != L");

    const T *ann = static_cast<const T*>(Ainfo.ptr);
    const T *inv = static_cast<const T*>(Iinfo.ptr);

    // Xz2d_chunk: (N x (B*v_count)), Fortran (col-major), accumulated across SNP blocks by caller
    auto Xinfo = Xz2d_chunk.request();
    if (Xinfo.ndim != 2) throw std::runtime_error("Xz2d_chunk must be 2D");
    if ((int)Xinfo.shape[0] != N || (int)Xinfo.shape[1] != B * v_count)
        throw std::runtime_error("Xz2d_chunk shape must be (N, B*v_count)");
    T *Xptr = static_cast<T*>(Xinfo.ptr);
    const int ldc = N;  // leading dimension of C is full N

    // RNG seeded by (root, block_start, v_start) → chunk-size invariance
    const bool have_root = !seed_obj.is_none();
    uint64_t root_seed = have_root ? seed_obj.cast<uint64_t>() : std::random_device{}();
    std::mt19937_64 rng(make_seed(root_seed, /*block=*/blk_start, /*v0=*/v_start));
    std::normal_distribution<T> gN(0, (T)1);

    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_normal     = (rand_dist == "normal");
    const bool is_spherical  = (rand_dist == "spherical");

    // --------- Reusable buffers ----------
    // Per-bin working buffers (size L; we use the first K entries)
    std::vector<int> idx_buf((size_t)L);
    std::vector<T>   scale_buf((size_t)L);

    // Z: (L x v_count), col-major (same for all bins)
    std::vector<T> Z((size_t)L * (size_t)v_count, T(0));
    for (int c = 0; c < v_count; ++c) {
        if (is_rademacher) {
            for (int r = 0; r < L; ++r) {
                int s = (rng() & 1) ? +1 : -1;
                Z[(size_t)r + (size_t)c * (size_t)L] = (T)s;
            }
        } else {
            long double ss = 0.0L;
            for (int r = 0; r < L; ++r) {
                T z = gN(rng);
                Z[(size_t)r + (size_t)c * (size_t)L] = z;
                if (is_spherical) ss += (long double)z * (long double)z;
            }
            if (is_spherical) {
                T scale = (ss > 0.0L) ? static_cast<T>(std::sqrt((long double)L / ss)) : T(1);
                for (int r = 0; r < L; ++r) {
                    Z[(size_t)r + (size_t)c * (size_t)L] *= scale;
                }
            }
        }
    }

    // N-tiling size by precision
    const int N_TILE = std::is_same_v<T,float> ? 128 : 64;

    // Pre-allocate buffers based on trusted Kmax (no per-bin realloc)
    const int Kmax = kmax_hint;
    std::vector<T> Bcol((size_t)Kmax * (size_t)v_count); // we'll pack tight K×v_count into the front

    for (int k = 0; k < B; ++k) {
        // Build idx_buf & scale_buf for this bin
        int K = 0;
        for (int i = 0; i < L; ++i) {
            T ak = ann[(std::size_t)i * (std::size_t)B + (std::size_t)k];
            if (ak != T(0)) {
                idx_buf[(size_t)K]   = i;
                scale_buf[(size_t)K] = inv[i] * std::sqrt(ak);
                ++K;
            }
        }
        if (K == 0) continue; // nothing to do for this bin

        // Pack Bcol tightly as (K x v_count) into the FRONT of the reusable buffer
        for (int c = 0; c < v_count; ++c) {
            const T *zc = Z.data() + (std::size_t)c * (std::size_t)L;
            T *dst_col  = Bcol.data() + (std::size_t)c * (std::size_t)K;  // tight stride K (not Kmax)
            for (int r = 0; r < K; ++r) {
                dst_col[r] = zc[idx_buf[(size_t)r]];
            }
        }

        // Base pointer for C columns of this bin
        T *C_base = Xptr + (std::size_t)(k * v_count) * (std::size_t)N;

        // ---------------- N-tiling with OpenMP over row tiles ----------------
        #ifdef _OPENMP
        #pragma omp parallel
        {
            // Per-thread scratch for A_tile; capacity Nt*Kmax (Nt<=N_TILE)
            std::vector<T> A_tile((size_t)N_TILE * (size_t)Kmax);

            #pragma omp for schedule(static)
            for (int n0 = 0; n0 < N; n0 += N_TILE) {
                const int Nt = std::min(N - n0, N_TILE);

                // Pack A_tile: (Nt x K), col-major, tight
                for (int c = 0; c < K; ++c) {
                    const int snp = idx_buf[(size_t)c];
                    const T   s   = scale_buf[(size_t)c];
                    const T *src  = Geno.data() + (std::size_t)snp * (std::size_t)N + (std::size_t)n0;
                    T *dst       = A_tile.data() + (std::size_t)c   * (std::size_t)Nt;
                    for (int r = 0; r < Nt; ++r) dst[r] = src[r] * s;
                }

                // GEMM on this row tile
                T *C_tile = C_base + (std::size_t)n0;
                const int lda = Nt;   // A_tile leading dimension
                const int ldb = K;    // Bcol packed tightly
                const int ldc_tile = N;
                const T alpha = T(1), beta = T(1);
                gemm_col_major_nn<T>(/*m=*/Nt, /*n=*/v_count, /*k=*/K,
                                     /*A=*/A_tile.data(), /*lda=*/lda,
                                     /*B=*/Bcol.data(),   /*ldb=*/ldb,
                                     /*C=*/C_tile,        /*ldc=*/ldc_tile,
                                     alpha, beta);
            } // n0
        } // parallel
        #else
        // Fallback: single-threaded N-tiling
        {
            std::vector<T> A_tile((size_t)N_TILE * (size_t)Kmax);
            for (int n0 = 0; n0 < N; n0 += N_TILE) {
                const int Nt = std::min(N - n0, N_TILE);
                for (int c = 0; c < K; ++c) {
                    const int snp = idx_buf[(size_t)c];
                    const T   s   = scale_buf[(size_t)c];
                    const T *src  = Geno.data() + (std::size_t)snp * (std::size_t)N + (std::size_t)n0;
                    T *dst       = A_tile.data() + (std::size_t)c   * (std::size_t)Nt;
                    for (int r = 0; r < Nt; ++r) dst[r] = src[r] * s;
                }
                T *C_tile = C_base + (std::size_t)n0;
                const int lda = Nt, ldb = K, ldc_tile = N;
                const T alpha = T(1), beta = T(1);
                gemm_col_major_nn<T>(Nt, v_count, K,
                                     A_tile.data(), lda,
                                     Bcol.data(),   ldb,
                                     C_tile,        ldc_tile,
                                     alpha, beta);
            }
        }
        #endif
    } // bins
}

// -------------------------- Phase 2: compute_XtXz ----------------------------

template <typename T>
void phase2_compute_XtXz_bed_impl(const std::string &bed_prefix,
                                  const std::string &fam_path,
                                  int blk_start, int blk_end,
                                  py::object row_sel_obj,
                                  int ddof,
                                  py::array_t<T, py::array::c_style | py::array::forcecast> inv_left, // (L,)
                                  int nvecs,
                                  int vchunk,
                                  py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d,      // (N x (B*V))
                                  py::array_t<T, py::array::c_style | py::array::forcecast> meansq,    // (M x B)
                                  py::object C_opt, py::object R_opt,
                                  int N_denom) {
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines(fam_path);
    const int64_t M_total = count_lines(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    // Parse rows to keep
    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);

    // Read & standardize [blk_start:blk_end) → Geno (N x L), col-major
    int N = 0, L = 0;
    std::vector<T> Geno; // (N x L)
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // Optional projection: Y = Geno - C @ (R @ Geno)
    std::vector<T> Y = Geno; // start from Geno
    if (!C_opt.is_none() && !R_opt.is_none()) {
        py::array_t<T, py::array::f_style | py::array::forcecast> C = C_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        py::array_t<T, py::array::f_style | py::array::forcecast> R = R_opt.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Ci = C.request();
        auto Ri = R.request();
        if (Ci.ndim != 2 || Ri.ndim != 2) throw std::runtime_error("C and R must be 2D");
        int p = (int)Ci.shape[1];
        if ((int)Ci.shape[0] != N || (int)Ri.shape[0] != p || (int)Ri.shape[1] != N)
            throw std::runtime_error("C (N x p) / R (p x N) shape mismatch");

        const T *Cptr = static_cast<const T*>(Ci.ptr);
        const T *Rptr = static_cast<const T*>(Ri.ptr);

        // tmpG = R @ Geno   (p x L)
        std::vector<T> tmpG((size_t)p * (size_t)L, T(0));
        gemm_col_major_nn<T>(/*m=*/p, /*n=*/L, /*k=*/N,
                             /*A=*/Rptr, /*lda=*/p,
                             /*B=*/Geno.data(), /*ldb=*/N,
                             /*C=*/tmpG.data(), /*ldc=*/p,
                             /*alpha=*/T(1), /*beta=*/T(0));

        // Y ← Geno - C @ tmpG
        gemm_col_major_nn<T>(/*m=*/N, /*n=*/L, /*k=*/p,
                             /*A=*/Cptr, /*lda=*/N,
                             /*B=*/tmpG.data(), /*ldb=*/p,
                             /*C=*/Y.data(), /*ldc=*/N,
                             /*alpha=*/T(-1), /*beta=*/T(1));
    }

    // inv_left: (L,)
    auto Ii = inv_left.request();
    if (Ii.ndim != 1 || (int)Ii.shape[0] != L) throw std::runtime_error("inv_left shape mismatch");
    const T *inv = static_cast<const T*>(Ii.ptr);

    // Xz2d: (N x (B*V)), F-contiguous
    auto Xi = Xz2d.request();
    if (Xi.ndim != 2) throw std::runtime_error("Xz2d must be 2D");
    if ((int)Xi.shape[0] != N) throw std::runtime_error("Xz2d row count must equal N");
    T *Xptr = static_cast<T*>(Xi.ptr);
    const int BV = (int)Xi.shape[1];
    if ((BV % nvecs) != 0) throw std::runtime_error("Xz2d column count must be a multiple of nvecs");
    const int B = BV / nvecs;

    // meansq: (M x B), C-contiguous, we will write rows [blk_start:blk_end)
    auto Mi = meansq.request();
    if (Mi.ndim != 2) throw std::runtime_error("meansq must be 2D");
    if ((int)Mi.shape[1] != B) throw std::runtime_error("meansq.shape[1] != nbins");
    T *Mptr = static_cast<T*>(Mi.ptr);
    const int M = (int)Mi.shape[0];
    if (blk_end > M) throw std::runtime_error("meansq rows smaller than SNP count");

    // -------- Pre-scale Y columns once: s[i] = inv_left[i] / (N_denom - 1) --------
    T denom = T(N_denom) - T(1);
    if (denom <= T(0)) denom = T(1); // safeguard
    for (int i = 0; i < L; ++i) {
        const T s = inv[i] / denom;
        T *col = Y.data() + (size_t)i * (size_t)N;
        for (int r = 0; r < N; ++r) col[r] *= s;
    }

    // -------- N-tiling & per-thread reductions setup --------
    const int N_TILE = std::is_same_v<T,float> ? 128 : 64;    // row tile
    const int V_COL_TILE = std::is_same_v<T,float> ? 64 : 32; // per-thread column tile to bound scratch

    // For each bin, accumulate across all V in chunks of vchunk
    for (int k = 0; k < B; ++k) {
        // Per-bin accumulator over i=0..L-1
        std::vector<T> acc((size_t)L, T(0));

        for (int c0 = 0; c0 < nvecs; c0 += vchunk) {
            const int v = std::min(vchunk, nvecs - c0);
            if (v <= 0) break;

            // Parallelize over column tiles; each thread keeps its own local accumulators
            #ifdef _OPENMP
            #pragma omp parallel
            {
                std::vector<T> acc_thr((size_t)L, T(0));                        // per-thread row accumulator
                std::vector<T> Work_local((size_t)L * (size_t)V_COL_TILE, T(0)); // per-thread (L x vt) scratch

                #pragma omp for schedule(static)
                for (int jc = 0; jc < v; jc += V_COL_TILE) {
                    const int vt = std::min(V_COL_TILE, v - jc);

                    // zero the active portion of Work_local
                    std::fill(Work_local.begin(), Work_local.begin() + (size_t)L * (size_t)vt, T(0));

                    // Sum across N tiles: Work_local = Y^T_tile_sum @ MB_tile
                    for (int n0 = 0; n0 < N; n0 += N_TILE) {
                        const int Nt = std::min(N - n0, N_TILE);

                        // A = Y_tile (Nt x L), used with Transpose → (L x Nt); lda = N
                        const T *A_ptr = Y.data() + (size_t)n0;

                        // B = MB_tile (Nt x vt), NoTrans; ldb = N
                        const T *B_ptr = Xptr
                                       + (size_t)((k * nvecs) + (c0 + jc)) * (size_t)N
                                       + (size_t)n0;

                        // C = Work_local (L x vt), ldc = L
                        gemm_col_major_tn<T>(/*m=*/L, /*n=*/vt, /*k=*/Nt,
                                             /*A=*/A_ptr, /*lda=*/N,
                                             /*B=*/B_ptr, /*ldb=*/N,
                                             /*C=*/Work_local.data(), /*ldc=*/L,
                                             /*alpha=*/T(1), /*beta=*/T(1));
                    }

                    // Accumulate squares into acc_thr
                    for (int col = 0; col < vt; ++col) {
                        const T *wcol = Work_local.data() + (size_t)col * (size_t)L;
                        for (int i = 0; i < L; ++i) {
                            const T z = wcol[i];
                            acc_thr[(size_t)i] += z * z;
                        }
                    }
                } // jc

                // Reduce per-thread rows into bin accumulator
                #pragma omp critical
                {
                    for (int i = 0; i < L; ++i) acc[(size_t)i] += acc_thr[(size_t)i];
                }
            } // parallel
            #else
            // Single-thread fallback
            {
                std::vector<T> acc_thr((size_t)L, T(0));
                std::vector<T> Work_local((size_t)L * (size_t)V_COL_TILE, T(0));
                for (int jc = 0; jc < v; jc += V_COL_TILE) {
                    const int vt = std::min(V_COL_TILE, v - jc);
                    std::fill(Work_local.begin(), Work_local.begin() + (size_t)L * (size_t)vt, T(0));
                    for (int n0 = 0; n0 < N; n0 += N_TILE) {
                        const int Nt = std::min(N - n0, N_TILE);
                        const T *A_ptr = Y.data() + (size_t)n0;
                        const T *B_ptr = Xptr
                                       + (size_t)((k * nvecs) + (c0 + jc)) * (size_t)N
                                       + (size_t)n0;
                        gemm_col_major_tn<T>(L, vt, Nt,
                                             A_ptr, N,
                                             B_ptr, N,
                                             Work_local.data(), L,
                                             T(1), T(1));
                    }
                    for (int col = 0; col < vt; ++col) {
                        const T *wcol = Work_local.data() + (size_t)col * (size_t)L;
                        for (int i = 0; i < L; ++i) {
                            const T z = wcol[i];
                            acc_thr[(size_t)i] += z * z;
                        }
                    }
                }
                for (int i = 0; i < L; ++i) acc[(size_t)i] += acc_thr[(size_t)i];
            }
            #endif
        } // c0 over vchunk

        // Write means over nvecs into meansq rows [blk_start:blk_end), column k
        const T invV = T(1) / T(nvecs);
        for (int i = 0; i < L; ++i) {
            const std::size_t row = (std::size_t)blk_start + (std::size_t)i;
            Mptr[row * (std::size_t)B + (std::size_t)k] = acc[(size_t)i] * invV;
        }
    } // bins
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
