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
void phase1_compute_Xz_bed_impl(const std::string &bed_prefix,
                                const std::string &fam_path,
                                int blk_start, int blk_end,
                                py::object row_sel_obj,
                                int ddof,
                                py::array_t<T, py::array::c_style | py::array::forcecast> annot_blk, // (L x B)
                                py::array_t<T, py::array::c_style | py::array::forcecast> inv_right, // (L,)
                                int nvecs,
                                int vchunk,
                                const std::string &rand_dist,
                                py::object seed_obj, // None or int
                                py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d, // (N x (B*V))
                                bool /*project_right*/,
                                py::object /*C_opt*/, py::object /*R_opt*/) {
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";
    const int64_t N_total = count_lines(fam_path);
    const int64_t M_total = count_lines(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);
    int N = 0, L = 0;
    std::vector<T> Geno; // N x L, col-major
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // annot_blk: (L x B)
    auto Ainfo = annot_blk.request();
    auto Iinfo = inv_right.request();
    if (Ainfo.ndim != 2) throw std::runtime_error("annot_blk must be 2D (L x B)");
    if (Iinfo.ndim != 1) throw std::runtime_error("inv_right must be 1D (L,)");

    const int B = (int)Ainfo.shape[1];
    if ((int)Ainfo.shape[0] != L) throw std::runtime_error("annot_blk.shape[0] != L");
    if ((int)Iinfo.shape[0] != L) throw std::runtime_error("inv_right.shape[0] != L");

    const T *ann = static_cast<const T*>(Ainfo.ptr);
    const T *inv = static_cast<const T*>(Iinfo.ptr);

    // Build per-bin indices and scales
    std::vector<std::vector<int>> idxs(B);
    std::vector<std::vector<T>>   scales(B);
    for (int k = 0; k < B; ++k) {
        // collect indices with nonzero annotation
        for (int i = 0; i < L; ++i) {
            T ak = ann[(size_t)i * (size_t)B + (size_t)k];
            if (ak != T(0)) {
                idxs[k].push_back(i);
            }
        }
        auto &bi = idxs[k];
        auto &sk = scales[k];
        sk.resize(bi.size());
        for (size_t c = 0; c < bi.size(); ++c) {
            int i = bi[c];
            T ak = ann[(size_t)i * (size_t)B + (size_t)k];
            sk[c] = inv[i] * std::sqrt(ak);
        }
    }

    // Xz2d: (N x (B*V)), Fortran (col-major)
    auto Xinfo = Xz2d.request();
    if (Xinfo.ndim != 2) throw std::runtime_error("Xz2d must be 2D");
    if ((int)Xinfo.shape[0] != N || (int)Xinfo.shape[1] != B * nvecs)
        throw std::runtime_error("Xz2d shape mismatch");
    T *Xptr = static_cast<T*>(Xinfo.ptr);
    const int ldc = N;

    // RNG seeding per (block, v0)
    const bool have_root = !seed_obj.is_none();
    uint64_t root_seed = have_root ? seed_obj.cast<uint64_t>() : std::random_device{}();

    // Generate Z for each V-chunk (L x Vt), then slice rows for each bin
    std::vector<T> Z; Z.reserve((size_t)L * std::min(vchunk, nvecs)); // col-major
    std::mt19937_64 rng;
    std::normal_distribution<T> gN(0, (T)1);

    const bool is_rademacher = (rand_dist == "rademacher");
    const bool is_normal     = (rand_dist == "normal");
    const bool is_spherical  = (rand_dist == "spherical");

    for (int v0 = 0; v0 < nvecs; v0 += vchunk) {
        int Vt = std::min(vchunk, nvecs - v0);

        // seed
        rng.seed(make_seed(root_seed, /*block=*/blk_start, v0));

        // Z: (L x Vt), col-major
        Z.assign((size_t)L * Vt, T(0));
        for (int c = 0; c < Vt; ++c) {
            if (is_rademacher) {
                for (int r = 0; r < L; ++r) {
                    int s = (rng() & 1) ? +1 : -1;
                    Z[(size_t)r + (size_t)c * (size_t)L] = (T)s;
                }
            } else {
                // normal for both 'normal' and 'spherical'
                long double ss = 0.0L;
                for (int r = 0; r < L; ++r) {
                    T z = gN(rng);
                    Z[(size_t)r + (size_t)c * (size_t)L] = z;
                    if (is_spherical) { ss += (long double)z * (long double)z; }
                }
                if (is_spherical) {
                    T scale = (ss > 0.0L) ? static_cast<T>(std::sqrt((long double)L / ss)) : T(1);
                    for (int r = 0; r < L; ++r) {
                        Z[(size_t)r + (size_t)c * (size_t)L] *= scale;
                    }
                }
            }
        }

        // Work buffers per bin:
        for (int k = 0; k < B; ++k) {
            const auto &bi = idxs[k];
            int K = (int)bi.size();
            if (K == 0) continue;

            // Acol: (N x K), col-major; copy Geno[:, bi] with scaling
            std::vector<T> Acol((size_t)N * K);
            for (int c = 0; c < K; ++c) {
                int snp = bi[c];
                T s = scales[k][(size_t)c];
                const T *colG = Geno.data() + (size_t)snp * (size_t)N;
                T *dst = Acol.data() + (size_t)c * (size_t)N;
                for (int r = 0; r < N; ++r) dst[r] = colG[r] * s;
            }

            // Bcol: (K x Vt), col-major; copy rows bi from Z
            std::vector<T> Bcol((size_t)K * Vt);
            for (int c = 0; c < Vt; ++c) {
                const T *zc = Z.data() + (size_t)c * (size_t)L;
                T *dst = Bcol.data() + (size_t)c * (size_t)K;
                for (int r = 0; r < K; ++r) {
                    dst[r] = zc[bi[r]];
                }
            }

            // C points into Xz2d at columns [k*nvecs + v0 : ... + Vt)
            T *C = Xptr + (size_t)(k * nvecs + v0) * (size_t)N;

            const int lda = N, ldb = K;
            const T alpha = T(1), beta = T(1);
            gemm_col_major_nn<T>(/*m=*/N, /*n=*/Vt, /*k=*/K,
                                 Acol.data(), lda,
                                 Bcol.data(), ldb,
                                 C, /*ldc=*/ldc,
                                 alpha, beta);
        }
    }
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

    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);

    int N = 0, L = 0;
    std::vector<T> Geno; // (N x L) col-major, standardized raw
    read_block_standardized<T>(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
    if (L == 0) return;

    // Possibly project: Y = Geno - C @ (R @ Geno), with C (N x p), R (p x N)
    std::vector<T> Y = Geno; // start as Geno
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

        // tmpG = R @ Geno   → (p x L)
        std::vector<T> tmpG((size_t)p * L, T(0));
        gemm_col_major_nn<T>(/*m=*/p, /*n=*/L, /*k=*/N,
                             /*A=R*/ Rptr, /*lda=*/p,
                             /*B=*/Geno.data(), /*ldb=*/N,
                             /*C=*/tmpG.data(), /*ldc=*/p,
                             /*alpha=*/T(1), /*beta=*/T(0));
        // Y = Geno - C @ tmpG
        // We do: Y ← (-1)*C@tmpG + (1)*Y
        gemm_col_major_nn<T>(/*m=*/N, /*n=*/L, /*k=*/p,
                             /*A=C*/ Cptr, /*lda=*/N,
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
    T *Xptr = static_cast<T*>(Xi.ptr);
    const int BV = (int)Xi.shape[1];
    if ((int)Xi.shape[0] != N || (BV % nvecs) != 0)
        throw std::runtime_error("Xz2d shape mismatch for N and V");
    const int B = BV / nvecs;

    // meansq: (M x B), C-contiguous
    auto Mi = meansq.request();
    if (Mi.ndim != 2) throw std::runtime_error("meansq must be 2D");
    if ((int)Mi.shape[1] != B) throw std::runtime_error("meansq.shape[1] != nbins");
    T *Mptr = static_cast<T*>(Mi.ptr);
    const int M = (int)Mi.shape[0];
    if (blk_end > M) throw std::runtime_error("meansq rows smaller than SNP count");
    const T scale_cols = T(1) / T(N_denom - 1);

    // For each bin, accumulate over V in chunks: Work = Y^T @ MB_k  (L x v)
    for (int k = 0; k < B; ++k) {
        std::vector<T> acc((size_t)L, T(0));

        for (int c0 = 0; c0 < nvecs; c0 += vchunk) {
            const int v = std::min(vchunk, nvecs - c0);

            // MB_k points to Xz2d columns [k*nvecs + c0 : ... + v)
            const T *MB = Xptr + (size_t)(k * nvecs + c0) * (size_t)N;
            const int ldb = N;

            // Work (L x v), col-major
            std::vector<T> Work((size_t)L * v);

            // Work = Y^T @ MB_k
            gemm_col_major_tn<T>(/*m=*/L, /*n=*/v, /*k=*/N,
                                 /*A=Y*/ Y.data(), /*lda=*/N,
                                 /*B=*/MB, /*ldb=*/ldb,
                                 /*C=*/Work.data(), /*ldc=*/L,
                                 /*alpha=*/T(1), /*beta=*/T(0));

            // Left normalization & divide by (N_denom - 1), then accumulate squares
            for (int col = 0; col < v; ++col) {
                T *wcol = Work.data() + (size_t)col * (size_t)L;
                for (int i = 0; i < L; ++i) {
                    T z = wcol[i] * inv[i] * scale_cols;
                    wcol[i] = z;
                }
                for (int i = 0; i < L; ++i) {
                    T z = wcol[i];
                    acc[i] += z * z;
                }
            }
        }

        // Write mean over V into meansq rows [blk_start:blk_end), column k
        for (int i = 0; i < L; ++i) {
            const std::size_t row = (std::size_t)blk_start + (std::size_t)i;
            Mptr[row * (std::size_t)B + (std::size_t)k] = acc[i] / T(nvecs);
        }
    }
}

// ------------------------------- PyBind module -------------------------------

PYBIND11_MODULE(gwldcore, m) {
    m.doc() = "C++ core for SUMMIT GW LD score (bed parser + BLAS-safe GEMMs)";

    // float32
    m.def("phase1_compute_Xz_bed",
          &phase1_compute_Xz_bed_impl<float>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("blk_start"), py::arg("blk_end"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("annot_blk"),
          py::arg("inv_right"),
          py::arg("nvecs"),
          py::arg("vchunk"),
          py::arg("rand_dist") = "rademacher",
          py::arg("seed") = py::none(),
          py::arg("Xz2d"),
          py::arg("project_right") = false,
          py::arg("C") = py::none(),
          py::arg("R") = py::none());

    // float64
    m.def("phase1_compute_Xz_bed",
          &phase1_compute_Xz_bed_impl<double>,
          py::arg("bed_prefix"),
          py::arg("fam_path"),
          py::arg("blk_start"), py::arg("blk_end"),
          py::arg("row_sel") = py::none(),
          py::arg("ddof") = 1,
          py::arg("annot_blk"),
          py::arg("inv_right"),
          py::arg("nvecs"),
          py::arg("vchunk"),
          py::arg("rand_dist") = "rademacher",
          py::arg("seed") = py::none(),
          py::arg("Xz2d"),
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
