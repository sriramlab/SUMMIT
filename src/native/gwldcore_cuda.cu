// src/gwldcore_cuda.cu
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
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <chrono>

#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "blas_compat.hpp"
#include "arch_compat.hpp"
#include "genotype.hpp"   // declares count_lines_cached and read_block_standardized_*()

namespace py = pybind11;

// ---- CUDA / cuBLAS error helpers -------------------------------------------
#ifndef GWLDCORE_CUDA_CHECKS_H
#define GWLDCORE_CUDA_CHECKS_H
#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _err = (call);                                                    \
    if (_err != cudaSuccess) {                                                    \
      throw std::runtime_error(                                                   \
        std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) \
        + " " + cudaGetErrorName(_err) + " - " + cudaGetErrorString(_err));      \
    }                                                                             \
  } while (0)

#define CUBLAS_CHECK(call)                                                        \
  do {                                                                            \
    cublasStatus_t _st = (call);                                                  \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                           \
      throw std::runtime_error(                                                   \
        std::string("cuBLAS error at ") + __FILE__ + ":" + std::to_string(__LINE__) \
        + " status=" + std::to_string(int(_st)));                                 \
    }                                                                             \
  } while (0)

#if CUDART_VERSION >= 11000
static inline void set_tf32(cublasHandle_t h, bool enable) {
  CUBLAS_CHECK(cublasSetMathMode(h, enable ? CUBLAS_TF32_TENSOR_OP_MATH
                                           : CUBLAS_DEFAULT_MATH));
}
#else
static inline void set_tf32(cublasHandle_t, bool){ /* no-op pre-Ampere */ }
#endif
#endif // GWLDCORE_CUDA_CHECKS_H

// ---- Device helpers ---------------------------------------------------------
static int device_count() {
    int n = 0; cudaGetDeviceCount(&n); return n;
}
static void set_device_or_throw(int idx) {
    int n = device_count();
    if (n <= 0) throw std::runtime_error("No CUDA devices visible.");
    if (idx < 0 || idx >= n) throw std::runtime_error("Invalid CUDA device index: " + std::to_string(idx));
    CUDA_CHECK(cudaSetDevice(idx));
}
static std::pair<size_t,size_t> mem_info() {
    size_t free_b=0,total_b=0; CUDA_CHECK(cudaMemGetInfo(&free_b,&total_b)); return {free_b,total_b};
}

// ---- Local-only helper: parse_row_sel ---------------------------------------
static std::vector<int> parse_row_sel(py::object row_sel_obj, int64_t N_total) {
    if (row_sel_obj.is_none()) {
        std::vector<int> rows((size_t)N_total);
        for (int64_t i = 0; i < N_total; ++i) rows[(size_t)i] = (int)i;
        return rows;
    }
    py::array idx = row_sel_obj.cast<py::array>();
    py::buffer_info bi = idx.request();
    std::vector<int> rows((size_t)bi.shape[0]);
    if (bi.format == py::format_descriptor<int32_t>::format()) {
        auto p = static_cast<const int32_t*>(bi.ptr);
        for (ssize_t i = 0; i < bi.shape[0]; ++i) rows[(size_t)i] = (int)p[i];
    } else {
        auto p = static_cast<const int64_t*>(bi.ptr);
        for (ssize_t i = 0; i < bi.shape[0]; ++i) rows[(size_t)i] = (int)p[i];
    }
    return rows;
}

// ---- Host BLAS helpers (for optional (I-CR) projection on host) --------------
template <typename T>
inline void host_gemm_col_major_nn(int m, int n, int k,
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

// Copy a row slice [n0:n0+Nt) out of a Fortran (col-major) N×q panel into Nt×q.
template <typename T>
static void pack_rows_colmajor(const T* src_colmajor, int N, int q,
                               int n0, int Nt, T* dst_rowpack /*Nt x q*/) {
    for (int c = 0; c < q; ++c) {
        const T* src = src_colmajor + (size_t)c * (size_t)N + (size_t)n0;
        T*       dst = dst_rowpack   + (size_t)c * (size_t)Nt;
        std::memcpy(dst, src, (size_t)Nt * sizeof(T));
    }
}

// Pack A_t (Nt×L) with per-column scale applied: A_t(:,i) = Geno(n0:…)*scale_i
template <typename T>
static void pack_A_scaled(const T* Geno_colmajor, int N, int L,
                          int n0, int Nt,
                          const T* inv, T denom,
                          T* A_pack /*Nt x L, ld = Nt*/) {
    const T sden = (denom > T(0)) ? denom : T(1);
    for (int i = 0; i < L; ++i) {
        const T scale = inv[i] / sden;
        const T* src  = Geno_colmajor + (size_t)i * (size_t)N + (size_t)n0;
        T*       dst  = A_pack        + (size_t)i * (size_t)Nt;
        #pragma omp simd
        for (int r = 0; r < Nt; ++r) dst[r] = src[r] * scale;
    }
}

// ---- cuBLAS GEMM wrappers ---------------------------------------------------
template <typename T>
struct GemmKernels {};
template <>
struct GemmKernels<float> {
    static void gemm(cublasHandle_t h,
                     cublasOperation_t opA, cublasOperation_t opB,
                     int m, int n, int k,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float* C, int ldc,
                     bool use_tf32, float alpha, float beta) {
        cudaDataType_t dt = CUDA_R_32F;
        cublasComputeType_t comp =
            use_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        cublasGemmAlgo_t algo =
            use_tf32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
        CUBLAS_CHECK(cublasGemmEx(h, opA, opB, m, n, k,
                                  &alpha, A, dt, lda,
                                          B, dt, ldb,
                                  &beta,  C, dt, ldc,
                                  comp, algo));
    }
};
template <>
struct GemmKernels<double> {
    static void gemm(cublasHandle_t h,
                     cublasOperation_t opA, cublasOperation_t opB,
                     int m, int n, int k,
                     const double* A, int lda,
                     const double* B, int ldb,
                     double* C, int ldc,
                     bool /*use_tf32*/, double alpha, double beta) {
        CUBLAS_CHECK(cublasDgemm(h, opA, opB, m, n, k,
                                 &alpha, A, lda, B, ldb, &beta, C, ldc));
    }
};

// ---- env helpers ------------------------------------------------------------
static inline double getenv_double(const char* k, double defv) {
    const char* v = std::getenv(k);
    if (!v) return defv;
    try { return std::stod(v); } catch (...) { return defv; }
}
static inline int getenv_int(const char* k, int defv) {
    const char* v = std::getenv(k);
    if (!v) return defv;
    try { return std::stoi(v); } catch (...) { return defv; }
}

// ===================== Main CUDA Phase-2 =====================================
template <typename T>
void phase2_compute_XtXz_bed_cuda_impl(const std::string &bed_prefix,
                                       const std::string &fam_path,
                                       int blk_start, int blk_end,
                                       py::object row_sel_obj,
                                       int ddof,
                                       py::array_t<T, py::array::c_style | py::array::forcecast> inv_left, // (L,)
                                       int nvecs,
                                       int /*vchunk_unused*/,
                                       py::array_t<T, py::array::f_style | py::array::forcecast> Xz2d,      // (N x (B*V)), K-major
                                       py::array_t<T, py::array::c_style | py::array::forcecast> meansq,    // (M x B)
                                       py::object C_opt, py::object R_opt,
                                       int N_denom,
                                       bool use_tf32 = false, int device_index = -1)
{
    if (device_index >= 0) set_device_or_throw(device_index);
    const std::string bed_path = bed_prefix + ".bed";
    const std::string bim_path = bed_prefix + ".bim";

    const int64_t N_total = count_lines_cached(fam_path);
    const int64_t M_total = count_lines_cached(bim_path);
    if (blk_end > M_total) throw std::runtime_error("blk_end exceeds #SNPs in BIM");

    std::vector<int> rows = parse_row_sel(row_sel_obj, N_total);

    // ------------------- Read + standardize Geno block (host) -------------------
    int N = 0, L = 0;
    std::vector<T> Geno; // (N x L), column-major
    {
        if constexpr (std::is_same_v<T,float>) {
            read_block_standardized_float(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
        } else {
            read_block_standardized_double(bed_path, fam_path, blk_start, blk_end, rows, ddof, Geno, N, L);
        }
    }
    if (L == 0) return;

    // Optional projection: Geno = (I - C R) * Geno (host side, reusing BLAS)
    if (!C_opt.is_none() && !R_opt.is_none()) {
        py::array_t<T, py::array::f_style | py::array::forcecast> C = C_opt.template cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        py::array_t<T, py::array::f_style | py::array::forcecast> R = R_opt.template cast<py::array_t<T, py::array::f_style | py::array::forcecast>>();
        auto Ci = C.request(); auto Ri = R.request();
        int p = (int)Ci.shape[1];
        if ((int)Ci.shape[0] != N || (int)Ri.shape[0] != p || (int)Ri.shape[1] != N)
            throw std::runtime_error("C/R shape mismatch");
        const T* Cptr = static_cast<const T*>(Ci.ptr);
        const T* Rptr = static_cast<const T*>(Ri.ptr);

        std::vector<T> tmpG((size_t)p * (size_t)L);
        host_gemm_col_major_nn<T>(/*m=*/p, /*n=*/L, /*k=*/N,
                                  /*A=*/Rptr, /*lda=*/p,
                                  /*B=*/Geno.data(), /*ldb=*/N,
                                  /*C=*/tmpG.data(), /*ldc=*/p,
                                  /*alpha=*/T(1), /*beta=*/T(0));
        host_gemm_col_major_nn<T>(/*m=*/N, /*n=*/L, /*k=*/p,
                                  /*A=*/Cptr, /*lda=*/N,
                                  /*B=*/tmpG.data(), /*ldb=*/p,
                                  /*C=*/Geno.data(), /*ldc=*/N,
                                  /*alpha=*/T(-1), /*beta=*/T(1));
    }

    // inv_left
    auto Ii = inv_left.request();
    if (Ii.ndim != 1) throw std::runtime_error("inv_left must be 1D");
    if ((int)Ii.shape[0] != L) throw std::runtime_error("inv_left length != L");
    const T* inv = static_cast<const T*>(Ii.ptr);

    // Xz2d (N x (B*V)) K-major
    auto Xi = Xz2d.request();
    if (Xi.ndim != 2 || (int)Xi.shape[0] != N) throw std::runtime_error("Xz2d shape mismatch");
    const int BV = (int)Xi.shape[1];
    if (nvecs <= 0 || (BV % nvecs) != 0) throw std::runtime_error("Xz2d col count must be multiple of nvecs");
    const int B = BV / nvecs;
    T* Xptr = static_cast<T*>(Xi.ptr); // column-major

    // meansq (M x B)
    auto Mi = meansq.request();
    if (Mi.ndim != 2 || (int)Mi.shape[1] != B) throw std::runtime_error("meansq shape mismatch");
    const int M = (int)Mi.shape[0];
    if (blk_end > M) throw std::runtime_error("meansq rows smaller than SNP count");
    T* Mptr = static_cast<T*>(Mi.ptr);

    // -------------- Q-tiling over columns (B*V) to limit device workbuf --------------
    const int Q = BV;
    int QPANEL = getenv_int("SUMMIT_P2_QP", 16384);
    QPANEL = std::max(64, std::min(QPANEL, Q));
    QPANEL = ((QPANEL + 63) / 64) * 64;
    if (QPANEL > Q) QPANEL = Q;

    const T invV = T(1) / T(nvecs);
    const T denom = T(N_denom) - T(1);

    // ------------------------- CUDA setup -------------------------
    py::gil_scoped_release nogil;

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaStream_t stream_copy[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_copy[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_copy[1], cudaStreamNonBlocking));
    cudaStream_t stream_compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream_compute));

    // Optional TF32 (float only)
    if constexpr (std::is_same_v<T,float>) {
        set_tf32(handle, use_tf32);
    }

    // Double buffers for H2D staging
    T* dA[2] = {nullptr, nullptr}; // (Nt x L), ld = Nt, used transposed in GEMM
    T* dB[2] = {nullptr, nullptr}; // (Nt x q), ld = Nt
    // Work panel lives once per q-panel: (L x q), ld = L
    T* dC = nullptr;

    // Pinned host staging (ping-pong)
    T* hA[2] = {nullptr, nullptr};
    T* hB[2] = {nullptr, nullptr};

    // Events to orchestrate safe reuse
    cudaEvent_t ev_copy_done[2], ev_gemm_done[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_copy_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_copy_done[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_gemm_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_gemm_done[1], cudaEventDisableTiming));

    // Loop over Q-panels
    for (int q0 = 0; q0 < Q; q0 += QPANEL) {
        const int q = std::min(QPANEL, Q - q0);

        // Decide Nt from free VRAM and q, L
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const double frac = std::min(0.9, std::max(0.3, getenv_double("SUMMIT_GPU_MEM_FRACTION", 0.7)));
        const size_t elem = sizeof(T);
        // Reserve room for: dC(L*q) + 2 * [dA(Nt*L) + dB(Nt*q)]
        size_t budget = (size_t)(frac * (double)free_b);
        size_t c_bytes = (size_t)L * (size_t)q * elem;
        size_t denom_bytes = 2 * elem * ((size_t)L + (size_t)q);
        int Nt = 0;
        if (budget > c_bytes + denom_bytes) {
            Nt = int((budget - c_bytes) / denom_bytes);
        }
        // clamp and align
        if (Nt <= 0) Nt = std::min(N, 2048);
        Nt = std::min(Nt, N);
        Nt = ((Nt + 127) / 128) * 128;
        Nt = std::min(Nt, N);

        // (Re)allocate device buffers for this q-panel
        if (dC) CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)L * (size_t)q * elem));
        CUDA_CHECK(cudaMemsetAsync(dC, 0, (size_t)L * (size_t)q * elem, stream_compute));

        for (int b = 0; b < 2; ++b) {
            if (dA[b]) CUDA_CHECK(cudaFree(dA[b]));
            if (dB[b]) CUDA_CHECK(cudaFree(dB[b]));
            CUDA_CHECK(cudaMalloc(&dA[b], (size_t)Nt * (size_t)L * elem));
            CUDA_CHECK(cudaMalloc(&dB[b], (size_t)Nt * (size_t)q * elem));
            if (hA[b]) CUDA_CHECK(cudaFreeHost(hA[b]));
            if (hB[b]) CUDA_CHECK(cudaFreeHost(hB[b]));
            CUDA_CHECK(cudaHostAlloc(&hA[b], (size_t)Nt * (size_t)L * elem, cudaHostAllocPortable));
            CUDA_CHECK(cudaHostAlloc(&hB[b], (size_t)Nt * (size_t)q * elem, cudaHostAllocPortable));
        }

        bool first_tile = true;
        int tiles = (N + Nt - 1) / Nt;

        for (int t = 0; t < tiles; ++t) {
            const int buf = t & 1;
            const int prev = buf ^ 1;
            const int n0 = t * Nt;
            const int thisNt = std::min(N - n0, Nt);

            // Make sure previous use of this buffer is done before we overwrite it
            if (t >= 2) {
                CUDA_CHECK(cudaEventSynchronize(ev_gemm_done[buf]));
            }

            // Pack host A and B for this tile
            {
                pack_A_scaled<T>(Geno.data(), N, L, n0, thisNt, inv, denom, hA[buf]);
                const T* rhs_panel = Xptr + (size_t)q0 * (size_t)N; // base to first column
                pack_rows_colmajor<T>(rhs_panel, N, q, n0, thisNt, hB[buf]);
            }

            // Async H2D copies on this buffer's copy stream
            CUDA_CHECK(cudaMemcpyAsync(dA[buf], hA[buf], (size_t)thisNt * (size_t)L * elem,
                                       cudaMemcpyHostToDevice, stream_copy[buf]));
            CUDA_CHECK(cudaMemcpyAsync(dB[buf], hB[buf], (size_t)thisNt * (size_t)q * elem,
                                       cudaMemcpyHostToDevice, stream_copy[buf]));
            CUDA_CHECK(cudaEventRecord(ev_copy_done[buf], stream_copy[buf]));

            // Ensure compute stream waits for copies of this buffer
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_copy_done[buf], 0));
            // Also ensure serialization of GEMMs that accumulate into dC
            if (t > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_gemm_done[prev], 0));
            }

            // GEMM: dC(L x q) += (dA^T)(L x Nt) * dB(Nt x q)
            const T alpha = T(1), beta = first_tile ? T(0) : T(1);
            GemmKernels<T>::gemm(
                /*h*/      handle,
                /*opA*/    CUBLAS_OP_T,
                /*opB*/    CUBLAS_OP_N,
                /*m*/      L,
                /*n*/      q,
                /*k*/      thisNt,
                /*A*/      dA[buf], /*lda=*/thisNt,
                /*B*/      dB[buf], /*ldb=*/thisNt,
                /*C*/      dC,      /*ldc=*/L,
                /*tf32*/   std::is_same_v<T,float> ? use_tf32 : false,
                /*alpha*/  alpha, /*beta*/ beta
            );

            CUDA_CHECK(cudaEventRecord(ev_gemm_done[buf], stream_compute));
            first_tile = false;
        }

        // Wait for the last GEMM in this q-panel
        CUDA_CHECK(cudaEventSynchronize(ev_gemm_done[(tiles-1)&1]));

        // D2H the (L x q) work panel and do the same reduction as CPU
        std::vector<T> Work_panel((size_t)L * (size_t)q);
        CUDA_CHECK(cudaMemcpyAsync(Work_panel.data(), dC,
                                   (size_t)L * (size_t)q * elem,
                                   cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        // Reduce on host: meansq[blk_start+i, k] += (wcol[i]^2) / nvecs
        #pragma omp parallel for schedule(static)
        for (int tcol = 0; tcol < q; ++tcol) {
            const int g = q0 + tcol;      // global column in [0, B*V)
            const int k = g / nvecs;      // K-major bin index
            const T* __restrict wcol = Work_panel.data() + (size_t)tcol * (size_t)L;
            T* __restrict out = Mptr + ((size_t)blk_start * (size_t)B + (size_t)k);
            #pragma omp simd
            for (int i = 0; i < L; ++i) {
                const T z2 = wcol[i] * wcol[i] * (T(1) / T(nvecs));
                out[(size_t)i * (size_t)B] += z2; // stride by B across SNP rows
            }
        }

        // Free buffers for this q-panel
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaFree(dA[b])); dA[b] = nullptr;
            CUDA_CHECK(cudaFree(dB[b])); dB[b] = nullptr;
            CUDA_CHECK(cudaFreeHost(hA[b])); hA[b] = nullptr;
            CUDA_CHECK(cudaFreeHost(hB[b])); hB[b] = nullptr;
        }
        CUDA_CHECK(cudaFree(dC)); dC = nullptr;
    }

    // Cleanup streams/handles/events
    CUDA_CHECK(cudaEventDestroy(ev_copy_done[0]));
    CUDA_CHECK(cudaEventDestroy(ev_copy_done[1]));
    CUDA_CHECK(cudaEventDestroy(ev_gemm_done[0]));
    CUDA_CHECK(cudaEventDestroy(ev_gemm_done[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_copy[0]));
    CUDA_CHECK(cudaStreamDestroy(stream_copy[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUBLAS_CHECK(cublasDestroy(handle));
}

// ===================== PyBind module =========================================
PYBIND11_MODULE(gwldcore_cuda, m) {
    m.doc() = "CUDA/cuBLAS backend for SUMMIT Phase-2 (XtXz) with optional TF32";

    // float32
    m.def("phase2_compute_XtXz_bed",
          &phase2_compute_XtXz_bed_cuda_impl<float>,
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
          py::arg("N_denom") = 0,
          py::arg("use_tf32") = false,
          py::arg("device_index") = -1);

    // float64
    m.def("phase2_compute_XtXz_bed",
          &phase2_compute_XtXz_bed_cuda_impl<double>,
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
          py::arg("N_denom") = 0,
          py::arg("use_tf32") = false,
          py::arg("device_index") = -1);

    m.def("list_gpus", [](){
        py::list out;
        int n = device_count();
        for (int i=0;i<n;++i){
            cudaDeviceProp p{}; cudaGetDeviceProperties(&p,i);
            int cur=0; cudaGetDevice(&cur);
            if (cur!=i) cudaSetDevice(i);
            auto [f,t] = mem_info();
            if (cur!=i) cudaSetDevice(cur);
            py::dict d;
            d["id"] = i;
            d["name"] = p.name;
            d["total_bytes"] = py::int_(t);
            d["free_bytes"]  = py::int_(f);
            out.append(d);
        }
        return out;
    }, "Return list of visible CUDA GPUs with free/total bytes.");
}
