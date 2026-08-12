#include "nb_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sched.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "blas_compat.hpp"

namespace {

#ifndef GWLDCORE_SOURCE_COMMIT
#define GWLDCORE_SOURCE_COMMIT "unknown"
#endif
#ifndef GWLDCORE_COMPILER_ID
#define GWLDCORE_COMPILER_ID "unknown"
#endif
#ifndef GWLDCORE_COMPILER_VERSION
#define GWLDCORE_COMPILER_VERSION "unknown"
#endif
#ifndef GWLDCORE_BUILD_TYPE
#define GWLDCORE_BUILD_TYPE "unknown"
#endif
#ifndef GWLDCORE_BLAS_VENDOR
#define GWLDCORE_BLAS_VENDOR "unknown"
#endif
#ifndef GWLDCORE_NATIVE_OPT
#define GWLDCORE_NATIVE_OPT 0
#endif
#ifndef GWLDCORE_OPENMP_ENABLED
#define GWLDCORE_OPENMP_ENABLED 0
#endif

constexpr double kOrthonormalTolerance = 1.0e-10;
constexpr double kStandardizedEnvTolerance = 1.0e-8;

size_t checked_add(size_t a, size_t b, const char* label) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error(std::string("GxE native size overflow in ") + label);
    }
    return a + b;
}

size_t checked_mul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("GxE native size overflow in ") + label);
    }
    return a * b;
}

int checked_blas_dim(size_t value, const char* label) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string("GxE native BLAS dimension overflow in ") + label);
    }
    return static_cast<int>(value);
}

void dgemm_nn(int m, int n, int k,
              const double* a, int lda,
              const double* b, int ldb,
              double* c, int ldc,
              double alpha = 1.0, double beta = 0.0) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm_tn(int m, int n, int k,
              const double* a, int lda,
              const double* b, int ldb,
              double* c, int ldc,
              double alpha = 1.0, double beta = 0.0) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#if defined(__linux__)

struct FileState {
    dev_t device{};
    ino_t inode{};
    off_t size{};
    nlink_t links{};
    timespec mtime{};
    timespec ctime{};
};

bool same_timespec(const timespec& a, const timespec& b) {
    return a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec;
}

FileState state_from_stat(const struct stat& observed) {
    return FileState{
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_nlink,
        observed.st_mtim,
        observed.st_ctim,
    };
}

bool same_state(const FileState& a, const FileState& b) {
    return a.device == b.device && a.inode == b.inode && a.size == b.size &&
           a.links == b.links &&
           same_timespec(a.mtime, b.mtime) && same_timespec(a.ctime, b.ctime);
}

int duplicate_cloexec(int descriptor, const char* label) {
    if (descriptor < 0) {
        throw std::runtime_error(std::string("Invalid GxE native ") + label + " descriptor");
    }
#ifdef F_DUPFD_CLOEXEC
    int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
#else
    int duplicate = ::dup(descriptor);
    if (duplicate >= 0) {
        const int flags = ::fcntl(duplicate, F_GETFD);
        if (flags < 0 || ::fcntl(duplicate, F_SETFD, flags | FD_CLOEXEC) != 0) {
            const int saved = errno;
            ::close(duplicate);
            errno = saved;
            duplicate = -1;
        }
    }
#endif
    if (duplicate < 0) {
        throw std::runtime_error(
            std::string("Failed to duplicate GxE native ") + label +
            " descriptor with close-on-exec: " + std::strerror(errno)
        );
    }
    return duplicate;
}

FileState validate_regular_fd(int descriptor, const char* label) {
    struct stat observed{};
    if (::fstat(descriptor, &observed) != 0) {
        throw std::runtime_error(
            std::string("Failed to stat GxE native ") + label + " descriptor: " +
            std::strerror(errno)
        );
    }
    if (!S_ISREG(observed.st_mode) || observed.st_size < 0) {
        throw std::runtime_error(
            std::string("GxE native ") + label + " descriptor must reference a regular file"
        );
    }
    return state_from_stat(observed);
}

int count_validated_rows_fd(int descriptor, const FileState& state, const char* label) {
    constexpr size_t chunk_size = 1U << 20;
    std::vector<unsigned char> buffer(chunk_size);
    off_t offset = 0;
    int64_t rows = 0;
    bool line_has_content = false;
    bool line_started = false;
    bool saw_any_byte = false;
    while (offset < state.size) {
        const size_t want = static_cast<size_t>(
            std::min<off_t>(static_cast<off_t>(chunk_size), state.size - offset)
        );
        const ssize_t got = ::pread(descriptor, buffer.data(), want, offset);
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("Failed to read GxE native ") + label + ": " +
                std::strerror(errno)
            );
        }
        if (got == 0) {
            throw std::runtime_error(std::string("Unexpected EOF in GxE native ") + label);
        }
        saw_any_byte = true;
        for (ssize_t i = 0; i < got; ++i) {
            const unsigned char value = buffer[static_cast<size_t>(i)];
            if (value == '\n') {
                if (!line_has_content) {
                    throw std::runtime_error(std::string("GxE native ") + label + " contains a blank row");
                }
                ++rows;
                line_has_content = false;
                line_started = false;
            } else if (value != ' ' && value != '\t' && value != '\r' && value != '\f' && value != '\v') {
                line_has_content = true;
                line_started = true;
            } else {
                line_started = true;
            }
        }
        offset += got;
    }
    if (line_started) {
        if (!line_has_content) {
            throw std::runtime_error(std::string("GxE native ") + label + " contains a blank final row");
        }
        ++rows;
    }
    if (!saw_any_byte || rows <= 0 || rows > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("GxE native ") + label + " has invalid row count");
    }
    return static_cast<int>(rows);
}

#endif

static uint64_t next_context_id() {
    static std::atomic<uint64_t> next{1};
    const uint64_t value = next.fetch_add(1, std::memory_order_relaxed);
    if (value == 0 || value == std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("GxE native context identity space is exhausted");
    }
    return value;
}

class ProjectedPanel {
public:
    ProjectedPanel(ProjectedPanel&&) noexcept = default;
    ProjectedPanel& operator=(ProjectedPanel&&) noexcept = default;
    ProjectedPanel(const ProjectedPanel&) = delete;
    ProjectedPanel& operator=(const ProjectedPanel&) = delete;

    int columns() const noexcept { return columns_; }
    double leakage() const noexcept { return leakage_; }

private:
    friend class DirectContext;

    ProjectedPanel(uint64_t context_id,
                   int rows,
                   int columns,
                   double leakage,
                   std::vector<double>&& data,
                   std::vector<double>&& environment_data)
        : context_id_(context_id), rows_(rows), columns_(columns),
          leakage_(leakage), data_(std::move(data)),
          environment_data_(std::move(environment_data)) {}

    uint64_t context_id_ = 0;
    int rows_ = 0;
    int columns_ = 0;
    double leakage_ = 0.0;
    std::vector<double> data_;
    std::vector<double> environment_data_;
};

class DirectContext {
public:
    DirectContext(int bed_descriptor,
                  int bim_descriptor,
                  int fam_descriptor,
                  nb::object row_sel_obj,
                  int ddof,
                  nb_vec1_ro<double> env,
                  nb_mat2f_ro<double> q_basis,
                  int decode_threads,
                  uint64_t max_workspace_bytes,
                  int target_panel_columns)
        : context_id_(next_context_id()),
          ddof_(ddof),
          decode_threads_(decode_threads),
          max_workspace_bytes_(max_workspace_bytes),
          target_panel_columns_(target_panel_columns) {
#if !defined(__linux__)
        (void)bed_descriptor; (void)bim_descriptor; (void)fam_descriptor;
        (void)row_sel_obj; (void)env; (void)q_basis;
        throw std::runtime_error("The bounded GxE native context requires Linux");
#else
        if (ddof_ != 0 && ddof_ != 1) {
            throw std::runtime_error("GxE native context supports ddof 0 or 1");
        }
        if (decode_threads_ <= 0) {
            throw std::runtime_error("GxE native decode_threads must be positive");
        }
#ifdef _OPENMP
        int thread_limit = omp_get_thread_limit();
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        if (::sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
            const int affinity_count = CPU_COUNT(&affinity);
            if (affinity_count > 0) thread_limit = std::min(thread_limit, affinity_count);
        }
        if (decode_threads_ > thread_limit) {
            throw std::runtime_error(
                "GxE native decode_threads exceeds the OpenMP/CPU-affinity limit"
            );
        }
#else
        if (decode_threads_ != 1) {
            throw std::runtime_error(
                "GxE native decode_threads must be one when OpenMP is unavailable"
            );
        }
#endif
        if (max_workspace_bytes_ == 0 ||
            max_workspace_bytes_ > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("GxE native max_workspace_bytes is invalid");
        }
        if (target_panel_columns_ <= 0) {
            throw std::runtime_error("GxE native target_panel_columns must be positive");
        }
        try {
            bed_fd_ = duplicate_cloexec(bed_descriptor, "BED");
            bim_fd_ = duplicate_cloexec(bim_descriptor, "BIM");
            fam_fd_ = duplicate_cloexec(fam_descriptor, "FAM");
            bed_state_ = validate_regular_fd(bed_fd_, "BED");
            bim_state_ = validate_regular_fd(bim_fd_, "BIM");
            fam_state_ = validate_regular_fd(fam_fd_, "FAM");
            n_total_ = count_validated_rows_fd(fam_fd_, fam_state_, "FAM");
            m_total_ = count_validated_rows_fd(bim_fd_, bim_state_, "BIM");
            const size_t bytes_per_snp = checked_add(static_cast<size_t>(n_total_), 3, "BED stride") / 4;
            const size_t expected = checked_add(
                3, checked_mul(bytes_per_snp, static_cast<size_t>(m_total_), "BED byte size"),
                "BED byte size"
            );
            if (static_cast<uint64_t>(bed_state_.size) != static_cast<uint64_t>(expected)) {
                throw std::runtime_error("GxE native BED byte size disagrees with FAM/BIM dimensions");
            }
            bed_size_ = expected;
            bytes_per_snp_ = bytes_per_snp;
            bed_base_ = static_cast<unsigned char*>(
                ::mmap(nullptr, bed_size_, PROT_READ, MAP_PRIVATE, bed_fd_, 0)
            );
            if (bed_base_ == MAP_FAILED) {
                bed_base_ = nullptr;
                throw std::runtime_error(
                    std::string("Failed to mmap GxE native BED: ") + std::strerror(errno)
                );
            }
            if (bed_base_[0] != 0x6c || bed_base_[1] != 0x1b || bed_base_[2] != 0x01) {
                throw std::runtime_error("GxE native input is not a SNP-major PLINK BED");
            }
            parse_rows(std::move(row_sel_obj));
            copy_and_validate_design(env, q_basis);
            check_files_unchanged();
        } catch (...) {
            close_internal();
            throw;
        }
#endif
    }

    ~DirectContext() { close_internal(); }
    DirectContext(const DirectContext&) = delete;
    DirectContext& operator=(const DirectContext&) = delete;

    void close() {
        auto guard = acquire_call_lock();
        close_internal();
    }

    nb::dict info() const {
        auto guard = acquire_call_lock();
        ensure_open();
        nb::dict result;
        result["n_total"] = n_total_;
        result["m_total"] = m_total_;
        result["n_selected"] = n_;
        result["q_rank"] = q_;
        result["ddof"] = ddof_;
        result["decode_threads"] = decode_threads_;
        result["max_workspace_bytes"] = max_workspace_bytes_;
        result["target_panel_columns"] = target_panel_columns_;
        result["environment_mean"] = environment_mean_;
        result["environment_variance"] = environment_variance_;
        result["max_q_gram_error"] = max_q_gram_error_;
        return result;
    }

    nb::dict feature_block(int blk_start,
                           int blk_end,
                           double eps_var,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        if (!(eps_var > 0.0) || !std::isfinite(eps_var)) {
            throw std::runtime_error("GxE native eps_var must be positive and finite");
        }
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int rank = n_ - q_;
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "feature genotype");
        elements = checked_add(elements, checked_mul(4U, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(l), "feature moments"), "feature moments"), "feature workspace");
        elements = checked_add(elements, checked_mul(11U, static_cast<size_t>(l), "feature vectors"), "feature workspace");
        ensure_workspace(elements, "feature block");

        double* scale_x = nullptr;
        double* scale_w = nullptr;
        double* norm_x = nullptr;
        double* norm_w = nullptr;
        double* diag_x = nullptr;
        double* diag_w = nullptr;
        double* corr_xw = nullptr;
        auto scale_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &scale_x);
        auto scale_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &scale_w);
        auto norm_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &norm_x);
        auto norm_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &norm_w);
        auto diag_x_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &diag_x);
        auto diag_w_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &diag_w);
        auto corr_xw_out = make_owned_numpy_vec1<double>(static_cast<size_t>(l), &corr_xw);

        int64_t missing = 0;
        double max_leak_x = 0.0;
        double max_leak_w = 0.0;
        {
            nb::gil_scoped_release release;
            std::vector<double> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            std::vector<double> moments(
                checked_mul(4U, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(l), "feature moments"), "feature moments"),
                0.0
            );
            std::vector<double> scalar(4U * static_cast<size_t>(l), 0.0);
            double* s0 = scalar.data();
            double* s1 = s0 + l;
            double* s2 = s1 + l;
            double* s4 = s2 + l;
            for (int j = 0; j < l; ++j) {
                const double* column = geno.data() + static_cast<size_t>(j) * static_cast<size_t>(n_);
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a4 = 0.0;
                for (int i = 0; i < n_; ++i) {
                    const double e = env_[static_cast<size_t>(i)];
                    const double g2 = column[i] * column[i];
                    const double e2 = e * e;
                    a0 += g2;
                    a1 += e * g2;
                    a2 += e2 * g2;
                    a4 += e2 * e2 * g2;
                }
                s0[j] = a0; s1[j] = a1; s2[j] = a2; s4[j] = a4;
            }
            for (int power = 0; power < 4; ++power) {
                double* out = moments.data() + static_cast<size_t>(power) * static_cast<size_t>(q_) * static_cast<size_t>(l);
                dgemm_tn(q_, l, n_, q_basis_.data(), n_, geno.data(), n_, out, q_);
                if (power != 3) {
                    for (int j = 0; j < l; ++j) {
                        double* column = geno.data() + static_cast<size_t>(j) * static_cast<size_t>(n_);
                        for (int i = 0; i < n_; ++i) column[i] *= env_[static_cast<size_t>(i)];
                    }
                }
            }
            const double* u0_all = moments.data();
            const double* u1_all = u0_all + static_cast<size_t>(q_) * static_cast<size_t>(l);
            const double* u2_all = u1_all + static_cast<size_t>(q_) * static_cast<size_t>(l);
            const double* u3_all = u2_all + static_cast<size_t>(q_) * static_cast<size_t>(l);
            for (int j = 0; j < l; ++j) {
                const double* u0 = u0_all + static_cast<size_t>(j) * static_cast<size_t>(q_);
                const double* u1 = u1_all + static_cast<size_t>(j) * static_cast<size_t>(q_);
                const double* u2 = u2_all + static_cast<size_t>(j) * static_cast<size_t>(q_);
                const double* u3 = u3_all + static_cast<size_t>(j) * static_cast<size_t>(q_);
                double u00 = 0.0, u11 = 0.0, u01 = 0.0;
                double u0u2 = 0.0, u1u3 = 0.0;
                double u0e2u0 = 0.0, u1e2u1 = 0.0;
                double leak_x_sq = 0.0, leak_w_sq = 0.0;
                for (int a = 0; a < q_; ++a) {
                    u00 += u0[a] * u0[a];
                    u11 += u1[a] * u1[a];
                    u01 += u0[a] * u1[a];
                    u0u2 += u0[a] * u2[a];
                    u1u3 += u1[a] * u3[a];
                    double e2u0 = 0.0, e2u1 = 0.0;
                    double gram_u0 = 0.0, gram_u1 = 0.0;
                    for (int b = 0; b < q_; ++b) {
                        const size_t index = static_cast<size_t>(b) * static_cast<size_t>(q_) + static_cast<size_t>(a);
                        e2u0 += q_e2_q_[index] * u0[b];
                        e2u1 += q_e2_q_[index] * u1[b];
                        gram_u0 += q_gram_[index] * u0[b];
                        gram_u1 += q_gram_[index] * u1[b];
                    }
                    u0e2u0 += u0[a] * e2u0;
                    u1e2u1 += u1[a] * e2u1;
                    const double lx = u0[a] - gram_u0;
                    const double lw = u1[a] - gram_u1;
                    leak_x_sq += lx * lx;
                    leak_w_sq += lw * lw;
                }
                const double ssx = s0[j] - u00;
                const double ssw = s2[j] - u11;
                const double varx = ssx / static_cast<double>(rank);
                const double varw = ssw / static_cast<double>(rank);
                if (!std::isfinite(varx) || !std::isfinite(varw) ||
                    varx <= eps_var || varw <= eps_var) {
                    throw std::runtime_error(
                        "GxE native projected feature has zero or invalid variance at block offset " +
                        std::to_string(j)
                    );
                }
                scale_x[j] = 1.0 / std::sqrt(varx);
                scale_w[j] = 1.0 / std::sqrt(varw);
                norm_x[j] = scale_x[j] * scale_x[j] * ssx / static_cast<double>(rank);
                norm_w[j] = scale_w[j] * scale_w[j] * ssw / static_cast<double>(rank);
                diag_x[j] = scale_x[j] * scale_x[j] *
                    (s2[j] - 2.0 * u0u2 + u0e2u0) / static_cast<double>(rank);
                diag_w[j] = scale_w[j] * scale_w[j] *
                    (s4[j] - 2.0 * u1u3 + u1e2u1) / static_cast<double>(rank);
                corr_xw[j] = scale_x[j] * scale_w[j] *
                    (s1[j] - u01) / static_cast<double>(rank);
                max_leak_x = std::max(max_leak_x, std::sqrt(std::max(0.0, leak_x_sq) / std::max(ssx, std::numeric_limits<double>::min())));
                max_leak_w = std::max(max_leak_w, std::sqrt(std::max(0.0, leak_w_sq) / std::max(ssw, std::numeric_limits<double>::min())));
                if (!std::isfinite(scale_x[j]) || !std::isfinite(scale_w[j]) ||
                    !std::isfinite(norm_x[j]) || !std::isfinite(norm_w[j]) ||
                    !std::isfinite(diag_x[j]) || !std::isfinite(diag_w[j]) ||
                    !std::isfinite(corr_xw[j])) {
                    throw std::runtime_error("GxE native feature output contains a non-finite value");
                }
            }
            check_files_unchanged();
        }
        nb::dict result;
        result["scale_x"] = scale_x_out;
        result["scale_w"] = scale_w_out;
        result["norm_x"] = norm_x_out;
        result["norm_w"] = norm_w_out;
        result["diag_nxe_x"] = diag_x_out;
        result["diag_nxe_w"] = diag_w_out;
        result["corr_xw"] = corr_xw_out;
        result["max_projection_leakage_additive"] = max_leak_x;
        result["max_projection_leakage_interaction"] = max_leak_w;
        result["missing_genotype_calls"] = missing;
        return result;
    }

    nb::tuple source_block(int blk_start,
                           int blk_end,
                           nb_vec1_ro<double> scale_x,
                           nb_vec1_ro<double> scale_w,
                           nb_vec1_ro<double> sqrt_annotation,
                           nb_mat2f_ro<double> probes,
                           nb_vec1_ro<int32_t> group_ids,
                           int num_groups,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int v = checked_blas_dim(probes.shape(1), "source probes");
        if (v <= 0 || checked_blas_dim(probes.shape(0), "source variants") != l ||
            checked_blas_dim(scale_x.shape(0), "source scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "source scale_w") != l ||
            checked_blas_dim(sqrt_annotation.shape(0), "source annotation") != l ||
            checked_blas_dim(group_ids.shape(0), "source groups") != l) {
            throw std::runtime_error("GxE native source input shape mismatch");
        }
        if (num_groups < 1 || num_groups > 4) {
            throw std::runtime_error("GxE native source supports one to four local groups");
        }
        const size_t columns_size = checked_mul(static_cast<size_t>(num_groups), static_cast<size_t>(v), "source columns");
        const int columns = checked_blas_dim(columns_size, "source columns");
        const size_t probe_elements = checked_mul(
            static_cast<size_t>(l), static_cast<size_t>(v), "source probe snapshot"
        );
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "source genotype");
        elements = checked_add(elements, checked_mul(static_cast<size_t>(l), columns_size, "source weights"), "source workspace");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(n_), columns_size, "source outputs"), "source outputs"), "source workspace");
        elements = checked_add(elements, checked_mul(static_cast<size_t>(q_), columns_size, "source projection"), "source workspace");
        elements = checked_add(elements, probe_elements, "source input snapshots");
        elements = checked_add(elements, checked_mul(4U, static_cast<size_t>(l), "source vector snapshots"), "source input snapshots");
        ensure_workspace(elements, "source block");

        // Nanobind array views borrow caller memory.  Snapshot every input while
        // the GIL is held so no borrowed buffer is read during native compute.
        const std::vector<double> scale_x_snapshot(
            scale_x.data(), scale_x.data() + static_cast<size_t>(l)
        );
        const std::vector<double> scale_w_snapshot(
            scale_w.data(), scale_w.data() + static_cast<size_t>(l)
        );
        const std::vector<double> annotation_snapshot(
            sqrt_annotation.data(), sqrt_annotation.data() + static_cast<size_t>(l)
        );
        const std::vector<double> probe_snapshot(
            probes.data(), probes.data() + probe_elements
        );
        const std::vector<int32_t> group_snapshot(
            group_ids.data(), group_ids.data() + static_cast<size_t>(l)
        );

        double* source_x = nullptr;
        double* source_w = nullptr;
        auto source_x_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(n_), columns_size, &source_x);
        auto source_w_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(n_), columns_size, &source_w);
        int64_t missing = 0;
        {
            nb::gil_scoped_release release;
            std::vector<double> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            std::vector<double> weighted(
                checked_mul(static_cast<size_t>(l), columns_size, "source weights"), 0.0
            );
            const double* zp = probe_snapshot.data();
            const double* sxp = scale_x_snapshot.data();
            const double* swp = scale_w_snapshot.data();
            const double* ap = annotation_snapshot.data();
            const int32_t* gp = group_snapshot.data();
            for (int j = 0; j < l; ++j) {
                const int group = static_cast<int>(gp[j]);
                if (group < 0 || group >= num_groups) {
                    throw std::runtime_error("GxE native source group ID is out of range");
                }
                if (!std::isfinite(sxp[j]) || !std::isfinite(swp[j]) ||
                    sxp[j] <= 0.0 || swp[j] <= 0.0 ||
                    !std::isfinite(ap[j]) || ap[j] < 0.0) {
                    throw std::runtime_error("GxE native source scale/annotation is invalid");
                }
                for (int c = 0; c < v; ++c) {
                    const double probe = zp[static_cast<size_t>(c) * static_cast<size_t>(l) + static_cast<size_t>(j)];
                    if (!std::isfinite(probe)) {
                        throw std::runtime_error("GxE native source probe contains a non-finite value");
                    }
                    const size_t index = static_cast<size_t>(group * v + c) * static_cast<size_t>(l) + static_cast<size_t>(j);
                    weighted[index] = probe * ap[j] * sxp[j];
                }
            }
            dgemm_nn(n_, columns, l, geno.data(), n_, weighted.data(), l, source_x, n_);
            for (int j = 0; j < l; ++j) {
                const int group = static_cast<int>(gp[j]);
                for (int c = 0; c < v; ++c) {
                    const double probe = zp[
                        static_cast<size_t>(c) * static_cast<size_t>(l) +
                        static_cast<size_t>(j)
                    ];
                    const size_t index = static_cast<size_t>(group * v + c) * static_cast<size_t>(l) + static_cast<size_t>(j);
                    // Form the interaction weight directly.  Reusing the X
                    // weight through ``*(scale_w / scale_x)`` is
                    // algebraically unnecessary and can overflow at the
                    // intermediate ratio even when this final product is
                    // finite.
                    weighted[index] = probe * ap[j] * swp[j];
                }
            }
            dgemm_nn(n_, columns, l, geno.data(), n_, weighted.data(), l, source_w, n_);
            for (int c = 0; c < columns; ++c) {
                double* column = source_w + static_cast<size_t>(c) * static_cast<size_t>(n_);
                for (int i = 0; i < n_; ++i) column[i] *= env_[static_cast<size_t>(i)];
            }
            project_panel_inplace(source_x, columns);
            project_panel_inplace(source_w, columns);
            validate_finite_output(source_x, checked_mul(static_cast<size_t>(n_), columns_size, "source output"), "source_x");
            validate_finite_output(source_w, checked_mul(static_cast<size_t>(n_), columns_size, "source output"), "source_w");
            check_files_unchanged();
        }
        return nb::make_tuple(source_x_out, source_w_out, missing);
    }

    ProjectedPanel prepare_projected_sources(nb_mat2f_ro<double> sources,
                                             double tolerance) const {
        auto guard = acquire_call_lock();
        ensure_open();
        check_files_unchanged();
        const int columns = checked_blas_dim(sources.shape(1), "projected source columns");
        if (columns <= 0 || checked_blas_dim(sources.shape(0), "projected source rows") != n_) {
            throw std::runtime_error("GxE native projected source shape mismatch");
        }
        if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
            throw std::runtime_error("GxE native projected source tolerance must be finite and nonnegative");
        }
        const size_t source_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(columns), "projected sources"
        );
        const size_t coefficient_elements = checked_mul(
            static_cast<size_t>(q_), static_cast<size_t>(columns),
            "projected source coefficients"
        );
        ensure_workspace(
            checked_add(
                checked_mul(2U, source_elements, "projected source snapshots"),
                coefficient_elements, "projected source preparation"
            ),
            "projected source preparation"
        );
        std::vector<double> snapshot(sources.data(), sources.data() + source_elements);
        std::vector<double> environment_snapshot(source_elements, 0.0);
        double leakage = 0.0;
        {
            nb::gil_scoped_release release;
            leakage = projected_source_leakage(snapshot.data(), columns);
            if (leakage > tolerance) {
                throw std::runtime_error(
                    "GxE native source is not in the projected fixed-effect complement: leakage=" +
                    std::to_string(leakage) + ", tolerance=" + std::to_string(tolerance)
                );
            }
            // Float32 sketch storage can reintroduce small fixed-effect
            // components after an exact source projection.  Seal the opaque
            // panel only after a float64 reprojection; target W-left products
            // use e*S and therefore require S itself to be projected.
            project_panel_inplace(snapshot.data(), columns);
            for (int column = 0; column < columns; ++column) {
                const double* source = snapshot.data() +
                    static_cast<size_t>(column) * static_cast<size_t>(n_);
                double* weighted = environment_snapshot.data() +
                    static_cast<size_t>(column) * static_cast<size_t>(n_);
                for (int row = 0; row < n_; ++row) {
                    weighted[row] = env_[static_cast<size_t>(row)] * source[row];
                }
            }
            validate_finite_output(
                environment_snapshot.data(), source_elements,
                "environment-weighted projected sources"
            );
            check_files_unchanged();
        }
        return ProjectedPanel(
            context_id_, n_, columns, leakage,
            std::move(snapshot), std::move(environment_snapshot)
        );
    }

    double validate_projected_sources(nb_mat2f_ro<double> sources,
                                      double tolerance) const {
        return prepare_projected_sources(sources, tolerance).leakage();
    }

    nb::tuple target_block(int blk_start,
                           int blk_end,
                           nb_vec1_ro<double> scale_x,
                           nb_vec1_ro<double> scale_w,
                           nb_mat2f_ro<double> sources,
                           bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        const int l = blk_end - blk_start;
        const int columns = checked_blas_dim(sources.shape(1), "target source columns");
        if (columns <= 0 || checked_blas_dim(sources.shape(0), "target source rows") != n_ ||
            checked_blas_dim(scale_x.shape(0), "target scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "target scale_w") != l) {
            throw std::runtime_error("GxE native target input shape mismatch");
        }
        const int width = std::min(target_panel_columns_, columns);
        const size_t source_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(columns), "target source snapshot"
        );
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "target genotype");
        elements = checked_add(elements, source_elements, "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, static_cast<size_t>(l), "target scale snapshots"), "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(l), static_cast<size_t>(columns), "target outputs"), "target outputs"), "target workspace");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(n_), static_cast<size_t>(width), "target panels"), "target panels"), "target workspace");
        elements = checked_add(elements, checked_mul(static_cast<size_t>(q_), static_cast<size_t>(width), "target QTS"), "target workspace");
        ensure_workspace(elements, "target block");

        const std::vector<double> scale_x_snapshot(
            scale_x.data(), scale_x.data() + static_cast<size_t>(l)
        );
        const std::vector<double> scale_w_snapshot(
            scale_w.data(), scale_w.data() + static_cast<size_t>(l)
        );
        const std::vector<double> source_snapshot(
            sources.data(), sources.data() + source_elements
        );

        double* work_x = nullptr;
        double* work_w = nullptr;
        auto work_x_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), static_cast<size_t>(columns), &work_x);
        auto work_w_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), static_cast<size_t>(columns), &work_w);
        int64_t missing = 0;
        double max_source_leakage = 0.0;
        {
            nb::gil_scoped_release release;
            validate_scales(scale_x_snapshot, scale_w_snapshot);
            validate_finite_output(source_snapshot.data(), source_elements, "target sources");
            std::vector<double> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            std::vector<double> projected(
                checked_mul(static_cast<size_t>(n_), static_cast<size_t>(width), "target projected panel"), 0.0
            );
            std::vector<double> env_projected(projected.size(), 0.0);
            std::vector<double> qts(
                checked_mul(static_cast<size_t>(q_), static_cast<size_t>(width), "target QTS"), 0.0
            );
            for (int c0 = 0; c0 < columns; c0 += width) {
                const int count = std::min(width, columns - c0);
                const double* source_panel = source_snapshot.data() + static_cast<size_t>(c0) * static_cast<size_t>(n_);
                const size_t panel_elements = checked_mul(
                    static_cast<size_t>(n_), static_cast<size_t>(count), "target panel"
                );
                dgemm_tn(q_, count, n_, q_basis_.data(), n_, source_panel, n_, qts.data(), q_);
                max_source_leakage = std::max(
                    max_source_leakage,
                    projected_source_leakage_from_coefficients(
                        source_panel, panel_elements, qts.data(),
                        checked_mul(static_cast<size_t>(q_), static_cast<size_t>(count), "target coefficients")
                    )
                );
                std::memcpy(
                    projected.data(), source_panel,
                    checked_mul(panel_elements, sizeof(double), "target panel copy")
                );
                dgemm_nn(n_, count, q_, q_basis_.data(), n_, qts.data(), q_, projected.data(), n_, -1.0, 1.0);
                for (int c = 0; c < count; ++c) {
                    const double* source = projected.data() + static_cast<size_t>(c) * static_cast<size_t>(n_);
                    double* weighted = env_projected.data() + static_cast<size_t>(c) * static_cast<size_t>(n_);
                    for (int i = 0; i < n_; ++i) weighted[i] = env_[static_cast<size_t>(i)] * source[i];
                }
                dgemm_tn(l, count, n_, geno.data(), n_, projected.data(), n_, work_x + static_cast<size_t>(c0) * static_cast<size_t>(l), l);
                dgemm_tn(l, count, n_, geno.data(), n_, env_projected.data(), n_, work_w + static_cast<size_t>(c0) * static_cast<size_t>(l), l);
            }
            scale_target_outputs(work_x, work_w, l, columns, scale_x_snapshot, scale_w_snapshot);
            check_files_unchanged();
        }
        return nb::make_tuple(work_x_out, work_w_out, missing, max_source_leakage);
    }

    nb::tuple target_projected_block(int blk_start,
                                     int blk_end,
                                     nb_vec1_ro<double> scale_x,
                                     nb_vec1_ro<double> scale_w,
                                     const ProjectedPanel& sources,
                                     bool require_missing_free) const {
        return target_projected_panels(
            blk_start, blk_end, scale_x, scale_w, sources, nullptr,
            require_missing_free
        );
    }

    nb::tuple target_projected_pair_block(int blk_start,
                                          int blk_end,
                                          nb_vec1_ro<double> scale_x,
                                          nb_vec1_ro<double> scale_w,
                                          const ProjectedPanel& first,
                                          const ProjectedPanel& second,
                                          bool require_missing_free) const {
        return target_projected_panels(
            blk_start, blk_end, scale_x, scale_w, first, &second,
            require_missing_free
        );
    }

private:
    double projected_source_leakage_from_coefficients(
        const double* sources,
        size_t source_elements,
        const double* coefficients,
        size_t coefficient_elements) const {
        long double source_ss = 0.0L;
        long double coefficient_ss = 0.0L;
        for (size_t index = 0; index < source_elements; ++index) {
            const long double value = static_cast<long double>(sources[index]);
            source_ss += value * value;
        }
        for (size_t index = 0; index < coefficient_elements; ++index) {
            const long double value = static_cast<long double>(coefficients[index]);
            coefficient_ss += value * value;
        }
        const double leakage = std::sqrt(static_cast<double>(
            coefficient_ss /
            std::max(source_ss, static_cast<long double>(std::numeric_limits<double>::min()))
        ));
        if (!std::isfinite(leakage)) {
            throw std::runtime_error("GxE native projected source diagnostic is non-finite");
        }
        return leakage;
    }

    double projected_source_leakage(const double* sources, int columns) const {
        const size_t source_elements = checked_mul(
            static_cast<size_t>(n_), static_cast<size_t>(columns), "projected sources"
        );
        const size_t coefficient_elements = checked_mul(
            static_cast<size_t>(q_), static_cast<size_t>(columns),
            "projected source coefficients"
        );
        validate_finite_output(sources, source_elements, "projected sources");
        std::vector<double> coefficients(coefficient_elements, 0.0);
        dgemm_tn(
            q_, columns, n_, q_basis_.data(), n_, sources, n_,
            coefficients.data(), q_
        );
        return projected_source_leakage_from_coefficients(
            sources, source_elements, coefficients.data(), coefficient_elements
        );
    }

    void validate_scales(const std::vector<double>& scale_x,
                         const std::vector<double>& scale_w) const {
        if (scale_x.size() != scale_w.size()) {
            throw std::runtime_error("GxE native target scale snapshots disagree");
        }
        for (size_t index = 0; index < scale_x.size(); ++index) {
            if (!std::isfinite(scale_x[index]) || !std::isfinite(scale_w[index]) ||
                scale_x[index] <= 0.0 || scale_w[index] <= 0.0) {
                throw std::runtime_error("GxE native target feature scale is invalid");
            }
        }
    }

    void scale_target_outputs(double* work_x,
                              double* work_w,
                              int rows,
                              int columns,
                              const std::vector<double>& scale_x,
                              const std::vector<double>& scale_w) const {
        for (int column = 0; column < columns; ++column) {
            double* x = work_x + static_cast<size_t>(column) * static_cast<size_t>(rows);
            double* w = work_w + static_cast<size_t>(column) * static_cast<size_t>(rows);
            for (int row = 0; row < rows; ++row) {
                x[row] *= scale_x[static_cast<size_t>(row)];
                w[row] *= scale_w[static_cast<size_t>(row)];
            }
        }
        const size_t output_elements = checked_mul(
            static_cast<size_t>(rows), static_cast<size_t>(columns), "target output"
        );
        validate_finite_output(work_x, output_elements, "work_x");
        validate_finite_output(work_w, output_elements, "work_w");
    }

    void validate_projected_panel_handle(const ProjectedPanel& panel) const {
        if (panel.context_id_ != context_id_ || panel.rows_ != n_ ||
            panel.columns_ <= 0 ||
            panel.data_.size() != checked_mul(
                static_cast<size_t>(n_), static_cast<size_t>(panel.columns_),
                "opaque projected panel"
            ) || panel.environment_data_.size() != panel.data_.size()) {
            throw std::runtime_error(
                "GxE native projected panel does not belong to this context"
            );
        }
    }

    nb::tuple target_projected_panels(
        int blk_start,
        int blk_end,
        nb_vec1_ro<double> scale_x,
        nb_vec1_ro<double> scale_w,
        const ProjectedPanel& first,
        const ProjectedPanel* second,
        bool require_missing_free) const {
        auto guard = acquire_call_lock();
        validate_block(blk_start, blk_end);
        check_files_unchanged();
        validate_projected_panel_handle(first);
        if (second != nullptr) validate_projected_panel_handle(*second);
        const int l = blk_end - blk_start;
        const size_t columns_size = checked_add(
            static_cast<size_t>(first.columns_),
            second == nullptr ? 0U : static_cast<size_t>(second->columns_),
            "projected target columns"
        );
        const int columns = checked_blas_dim(columns_size, "projected target columns");
        if (checked_blas_dim(scale_x.shape(0), "target scale_x") != l ||
            checked_blas_dim(scale_w.shape(0), "target scale_w") != l) {
            throw std::runtime_error("GxE native target input shape mismatch");
        }
        size_t elements = checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "target genotype");
        elements = checked_add(elements, checked_mul(2U, static_cast<size_t>(l), "target scale snapshots"), "target input snapshots");
        elements = checked_add(elements, checked_mul(2U, checked_mul(static_cast<size_t>(l), columns_size, "target outputs"), "target outputs"), "target workspace");
        ensure_workspace(elements, "projected target block");

        const std::vector<double> scale_x_snapshot(
            scale_x.data(), scale_x.data() + static_cast<size_t>(l)
        );
        const std::vector<double> scale_w_snapshot(
            scale_w.data(), scale_w.data() + static_cast<size_t>(l)
        );
        double* work_x = nullptr;
        double* work_w = nullptr;
        auto work_x_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), columns_size, &work_x);
        auto work_w_out = make_owned_numpy_mat2f<double>(static_cast<size_t>(l), columns_size, &work_w);
        int64_t missing = 0;
        {
            nb::gil_scoped_release release;
            validate_scales(scale_x_snapshot, scale_w_snapshot);
            std::vector<double> geno;
            std::vector<int> observed;
            missing = decode_block(blk_start, blk_end, require_missing_free, geno, observed);
            size_t output_offset = 0;
            auto consume = [&](const ProjectedPanel& panel) {
                for (int c0 = 0; c0 < panel.columns_; c0 += target_panel_columns_) {
                    const int count = std::min(target_panel_columns_, panel.columns_ - c0);
                    const size_t source_offset =
                        static_cast<size_t>(c0) * static_cast<size_t>(n_);
                    const size_t output_column = output_offset + static_cast<size_t>(c0);
                    dgemm_tn(
                        l, count, n_, geno.data(), n_,
                        panel.data_.data() + source_offset, n_,
                        work_x + output_column * static_cast<size_t>(l), l
                    );
                    dgemm_tn(
                        l, count, n_, geno.data(), n_,
                        panel.environment_data_.data() + source_offset, n_,
                        work_w + output_column * static_cast<size_t>(l), l
                    );
                }
                output_offset += static_cast<size_t>(panel.columns_);
            };
            consume(first);
            if (second != nullptr) consume(*second);
            scale_target_outputs(
                work_x, work_w, l, columns, scale_x_snapshot, scale_w_snapshot
            );
            check_files_unchanged();
        }
        const double leakage = std::max(
            first.leakage_, second == nullptr ? 0.0 : second->leakage_
        );
        return nb::make_tuple(work_x_out, work_w_out, missing, leakage);
    }

    std::unique_lock<std::mutex> acquire_call_lock() const {
        nb::gil_scoped_release release;
        return std::unique_lock<std::mutex>(call_mutex_);
    }

    void ensure_open() const {
        if (closed_ || bed_base_ == nullptr) {
            throw std::runtime_error("GxE native context is closed");
        }
    }

    void validate_block(int blk_start, int blk_end) const {
        ensure_open();
        if (blk_start < 0 || blk_end <= blk_start || blk_end > m_total_) {
            throw std::runtime_error("GxE native BED block is outside the BIM axis");
        }
    }

    void ensure_workspace(size_t elements, const char* label) const {
        const size_t bytes = checked_mul(elements, sizeof(double), label);
        if (static_cast<uint64_t>(bytes) > max_workspace_bytes_) {
            throw std::runtime_error(
                std::string("GxE native ") + label + " requires " +
                std::to_string(bytes) + " workspace bytes, exceeding the configured limit " +
                std::to_string(max_workspace_bytes_)
            );
        }
    }

    void parse_rows(nb::object row_sel_obj) {
        if (row_sel_obj.is_none()) {
            rows_.resize(static_cast<size_t>(n_total_));
            for (int i = 0; i < n_total_; ++i) rows_[static_cast<size_t>(i)] = i;
        } else {
            nb_any_array_ro idx = nb::cast<nb_any_array_ro>(row_sel_obj);
            if (idx.ndim() != 1 || idx.shape(0) > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("GxE native row_sel must be a bounded one-dimensional integer array");
            }
            rows_.resize(idx.shape(0));
            if (idx.dtype() == nb::dtype<int32_t>()) {
                auto view = idx.view<const int32_t, nb::ndim<1>>();
                for (size_t i = 0; i < rows_.size(); ++i) rows_[i] = static_cast<int>(view(i));
            } else if (idx.dtype() == nb::dtype<int64_t>()) {
                auto view = idx.view<const int64_t, nb::ndim<1>>();
                for (size_t i = 0; i < rows_.size(); ++i) {
                    const int64_t value = view(i);
                    if (value < 0 || value > std::numeric_limits<int>::max()) {
                        throw std::runtime_error("GxE native row_sel contains an out-of-range sample index");
                    }
                    rows_[i] = static_cast<int>(value);
                }
            } else {
                throw std::runtime_error("GxE native row_sel must have dtype int32 or int64");
            }
        }
        if (rows_.size() < 3) {
            throw std::runtime_error("GxE native context requires at least three selected samples");
        }
        int previous = -1;
        for (int row : rows_) {
            if (row < 0 || row >= n_total_ || row <= previous) {
                throw std::runtime_error("GxE native row_sel must be strictly increasing and within the FAM axis");
            }
            previous = row;
        }
        n_ = checked_blas_dim(rows_.size(), "selected samples");
    }

    void copy_and_validate_design(nb_vec1_ro<double> env, nb_mat2f_ro<double> q_basis) {
        if (checked_blas_dim(env.shape(0), "environment") != n_ ||
            checked_blas_dim(q_basis.shape(0), "projection rows") != n_) {
            throw std::runtime_error("GxE native environment/projection row mismatch");
        }
        q_ = checked_blas_dim(q_basis.shape(1), "projection rank");
        if (q_ < 1 || q_ >= n_) {
            throw std::runtime_error("GxE native projection basis has invalid rank");
        }
        env_.assign(env.data(), env.data() + static_cast<size_t>(n_));
        q_basis_.assign(
            q_basis.data(),
            q_basis.data() + checked_mul(static_cast<size_t>(n_), static_cast<size_t>(q_), "projection basis")
        );
        long double env_sum = 0.0L;
        long double env_ss = 0.0L;
        for (double value : env_) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("GxE native environment contains a non-finite value");
            }
            env_sum += static_cast<long double>(value);
            env_ss += static_cast<long double>(value) * static_cast<long double>(value);
        }
        environment_mean_ = static_cast<double>(env_sum / static_cast<long double>(n_));
        environment_variance_ = static_cast<double>(env_ss / static_cast<long double>(n_ - ddof_));
        if (std::abs(environment_mean_) > kStandardizedEnvTolerance ||
            !std::isfinite(environment_variance_) ||
            std::abs(environment_variance_ - 1.0) > kStandardizedEnvTolerance) {
            throw std::runtime_error("GxE native environment must be nonconstant, centered, and standardized for the configured ddof");
        }
        q_gram_.assign(checked_mul(static_cast<size_t>(q_), static_cast<size_t>(q_), "Q Gram"), 0.0);
        q_e2_q_.assign(q_gram_.size(), 0.0);
        max_q_gram_error_ = 0.0;
        for (int b = 0; b < q_; ++b) {
            for (int a = 0; a < q_; ++a) {
                double gram = 0.0;
                double e2gram = 0.0;
                for (int i = 0; i < n_; ++i) {
                    const double qa = q_basis_[static_cast<size_t>(a) * static_cast<size_t>(n_) + static_cast<size_t>(i)];
                    const double qb = q_basis_[static_cast<size_t>(b) * static_cast<size_t>(n_) + static_cast<size_t>(i)];
                    if (!std::isfinite(qa) || !std::isfinite(qb)) {
                        throw std::runtime_error("GxE native projection basis contains a non-finite value");
                    }
                    gram += qa * qb;
                    e2gram += qa * qb * env_[static_cast<size_t>(i)] * env_[static_cast<size_t>(i)];
                }
                const size_t index = static_cast<size_t>(b) * static_cast<size_t>(q_) + static_cast<size_t>(a);
                q_gram_[index] = gram;
                q_e2_q_[index] = e2gram;
                max_q_gram_error_ = std::max(max_q_gram_error_, std::abs(gram - ((a == b) ? 1.0 : 0.0)));
            }
        }
        if (max_q_gram_error_ > kOrthonormalTolerance) {
            throw std::runtime_error("GxE native projection basis is not orthonormal");
        }
        std::vector<double> intercept_coeff(static_cast<size_t>(q_), 0.0);
        std::vector<double> env_coeff(static_cast<size_t>(q_), 0.0);
        const double root_n = std::sqrt(static_cast<double>(n_));
        for (int a = 0; a < q_; ++a) {
            const double* column = q_basis_.data() + static_cast<size_t>(a) * static_cast<size_t>(n_);
            for (int i = 0; i < n_; ++i) {
                intercept_coeff[static_cast<size_t>(a)] += column[i] / root_n;
                env_coeff[static_cast<size_t>(a)] += column[i] * env_[static_cast<size_t>(i)];
            }
        }
        double intercept_resid_ss = 0.0;
        double env_resid_ss = 0.0;
        for (int i = 0; i < n_; ++i) {
            double fitted_intercept = 0.0;
            double fitted_env = 0.0;
            for (int a = 0; a < q_; ++a) {
                const double value = q_basis_[static_cast<size_t>(a) * static_cast<size_t>(n_) + static_cast<size_t>(i)];
                fitted_intercept += value * intercept_coeff[static_cast<size_t>(a)];
                fitted_env += value * env_coeff[static_cast<size_t>(a)];
            }
            const double ri = 1.0 / root_n - fitted_intercept;
            const double re = env_[static_cast<size_t>(i)] - fitted_env;
            intercept_resid_ss += ri * ri;
            env_resid_ss += re * re;
        }
        if (std::sqrt(intercept_resid_ss) > 1.0e-9 ||
            std::sqrt(env_resid_ss / std::max(static_cast<double>(env_ss), std::numeric_limits<double>::min())) > 1.0e-9) {
            throw std::runtime_error("GxE native projection basis must span the intercept and environment");
        }
    }

    int64_t decode_block(int blk_start,
                         int blk_end,
                         bool require_missing_free,
                         std::vector<double>& geno,
                         std::vector<int>& observed) const {
        const int l = blk_end - blk_start;
        geno.assign(checked_mul(static_cast<size_t>(n_), static_cast<size_t>(l), "decoded genotype"), 0.0);
        observed.assign(static_cast<size_t>(l), 0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(decode_threads_)
#endif
        for (int j = 0; j < l; ++j) {
            const unsigned char* bytes = bed_base_ + 3 +
                static_cast<size_t>(blk_start + j) * bytes_per_snp_;
            int nobs = 0;
            int64_t sum = 0;
            int64_t sumsq = 0;
            for (int i = 0; i < n_; ++i) {
                const int row = rows_[static_cast<size_t>(i)];
                const uint8_t bits = static_cast<uint8_t>((bytes[static_cast<size_t>(row >> 2)] >> ((row & 3) << 1)) & 0x3U);
                if (bits != 1U) {
                    const int value = (bits == 0U) ? 0 : (bits == 2U) ? 1 : 2;
                    ++nobs;
                    sum += value;
                    sumsq += value * value;
                }
            }
            observed[static_cast<size_t>(j)] = nobs;
            const double mean = (nobs > 0) ? static_cast<double>(sum) / static_cast<double>(nobs) : 0.0;
            double m2 = (nobs > 0)
                ? static_cast<double>(sumsq) - static_cast<double>(sum) * static_cast<double>(sum) / static_cast<double>(nobs)
                : 0.0;
            if (m2 < 0.0 && m2 > -1.0e-12) m2 = 0.0;
            const int denom = n_ - ddof_;
            const double inverse_sd = (denom > 0 && m2 > 0.0)
                ? std::sqrt(static_cast<double>(denom) / m2)
                : 1.0;
            double* column = geno.data() + static_cast<size_t>(j) * static_cast<size_t>(n_);
            for (int i = 0; i < n_; ++i) {
                const int row = rows_[static_cast<size_t>(i)];
                const uint8_t bits = static_cast<uint8_t>((bytes[static_cast<size_t>(row >> 2)] >> ((row & 3) << 1)) & 0x3U);
                if (bits == 1U) {
                    column[i] = 0.0;
                } else {
                    const int value = (bits == 0U) ? 0 : (bits == 2U) ? 1 : 2;
                    column[i] = -(static_cast<double>(value) - mean) * inverse_sd;
                }
            }
        }
        int64_t missing = 0;
        for (int count : observed) missing += static_cast<int64_t>(n_ - count);
        if (require_missing_free && missing != 0) {
            throw std::runtime_error("GxE native production backend requires a missing-free selected BED block");
        }
        return missing;
    }

    void project_panel_inplace(double* panel, int columns) const {
        if (columns <= 0) return;
        std::vector<double> coefficients(
            checked_mul(static_cast<size_t>(q_), static_cast<size_t>(columns), "projection coefficients"), 0.0
        );
        dgemm_tn(q_, columns, n_, q_basis_.data(), n_, panel, n_, coefficients.data(), q_);
        dgemm_nn(n_, columns, q_, q_basis_.data(), n_, coefficients.data(), q_, panel, n_, -1.0, 1.0);
    }

    void validate_finite_output(const double* values, size_t count, const char* label) const {
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error(std::string("GxE native ") + label + " contains a non-finite value");
            }
        }
    }

    void check_files_unchanged() const {
#if defined(__linux__)
        ensure_open();
        if (!same_state(validate_regular_fd(bed_fd_, "BED"), bed_state_) ||
            !same_state(validate_regular_fd(bim_fd_, "BIM"), bim_state_) ||
            !same_state(validate_regular_fd(fam_fd_, "FAM"), fam_state_)) {
            throw std::runtime_error("GxE native PLINK input changed after context construction");
        }
#endif
    }

    void close_internal() noexcept {
#if defined(__linux__)
        if (bed_base_ != nullptr) {
            ::munmap(bed_base_, bed_size_);
            bed_base_ = nullptr;
        }
        for (int* descriptor : {&bed_fd_, &bim_fd_, &fam_fd_}) {
            if (*descriptor >= 0) {
                ::close(*descriptor);
                *descriptor = -1;
            }
        }
#endif
        closed_ = true;
    }

    const uint64_t context_id_;
    int ddof_ = 1;
    int decode_threads_ = 1;
    uint64_t max_workspace_bytes_ = 0;
    int target_panel_columns_ = 64;
    int n_total_ = 0;
    int m_total_ = 0;
    int n_ = 0;
    int q_ = 0;
    std::vector<int> rows_;
    std::vector<double> env_;
    std::vector<double> q_basis_;
    std::vector<double> q_gram_;
    std::vector<double> q_e2_q_;
    double environment_mean_ = 0.0;
    double environment_variance_ = 0.0;
    double max_q_gram_error_ = 0.0;
    bool closed_ = false;
    mutable std::mutex call_mutex_;
#if defined(__linux__)
    int bed_fd_ = -1;
    int bim_fd_ = -1;
    int fam_fd_ = -1;
    FileState bed_state_{};
    FileState bim_state_{};
    FileState fam_state_{};
    unsigned char* bed_base_ = nullptr;
    size_t bed_size_ = 0;
    size_t bytes_per_snp_ = 0;
#else
    unsigned char* bed_base_ = nullptr;
#endif
};

}  // namespace

NB_MODULE(gxeldcore, module) {
    module.doc() = "Bounded double-precision native context for standardized one-environment GxE sketches";
    module.attr("__version__") = "1.0";
    module.def("build_info", []() {
        nb::dict result;
        result["backend_name"] = "gxeldcore_direct";
        result["backend_version"] = "1.0";
        result["api_version"] = 2;
        result["source_commit"] = GWLDCORE_SOURCE_COMMIT;
        result["source_tree_sha256"] = GWLDCORE_SOURCE_TREE_SHA256;
        result["compiler_id"] = GWLDCORE_COMPILER_ID;
        result["compiler_version"] = GWLDCORE_COMPILER_VERSION;
        result["build_type"] = GWLDCORE_BUILD_TYPE;
        result["blas_vendor"] = GWLDCORE_BLAS_VENDOR;
        result["cxx_standard"] = 17;
        result["optimization"] = "-O3";
        result["architecture_tuning"] = (
            bool(GWLDCORE_NATIVE_OPT) ? "-march=native" : "portable"
        );
        result["openmp_enabled"] = bool(GWLDCORE_OPENMP_ENABLED);
        result["native_optimization_enabled"] = bool(GWLDCORE_NATIVE_OPT);
        result["platform"] = "linux";
        return result;
    });
    nb::class_<ProjectedPanel>(module, "ProjectedPanel")
        .def_prop_ro("columns", &ProjectedPanel::columns)
        .def_prop_ro("leakage", &ProjectedPanel::leakage);
    nb::class_<DirectContext>(module, "DirectContext")
        .def(
            nb::init<int, int, int, nb::object, int, nb_vec1_ro<double>, nb_mat2f_ro<double>, int, uint64_t, int>(),
            nb::arg("bed_descriptor"), nb::arg("bim_descriptor"), nb::arg("fam_descriptor"),
            nb::arg("row_sel") = nb::none(), nb::arg("ddof") = 1,
            nb::arg("env"), nb::arg("q_basis"), nb::arg("decode_threads"),
            nb::arg("max_workspace_bytes"), nb::arg("target_panel_columns") = 64
        )
        .def("close", &DirectContext::close)
        .def("info", &DirectContext::info)
        .def(
            "feature_block", &DirectContext::feature_block,
            nb::arg("blk_start"), nb::arg("blk_end"), nb::arg("eps_var") = 1.0e-10,
            nb::arg("require_missing_free") = true
        )
        .def(
            "source_block", &DirectContext::source_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"), nb::arg("sqrt_annotation"),
            nb::arg("probes"), nb::arg("group_ids"), nb::arg("num_groups"),
            nb::arg("require_missing_free") = true
        )
        .def(
            "prepare_projected_sources", &DirectContext::prepare_projected_sources,
            nb::arg("sources"), nb::arg("tolerance") = 1.0e-10
        )
        .def(
            "validate_projected_sources", &DirectContext::validate_projected_sources,
            nb::arg("sources"), nb::arg("tolerance") = 1.0e-10
        )
        .def(
            "target_block", &DirectContext::target_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"), nb::arg("sources"),
            nb::arg("require_missing_free") = true
        )
        .def(
            "target_projected_block", &DirectContext::target_projected_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"), nb::arg("sources"),
            nb::arg("require_missing_free") = true
        )
        .def(
            "target_projected_pair_block", &DirectContext::target_projected_pair_block,
            nb::arg("blk_start"), nb::arg("blk_end"),
            nb::arg("scale_x"), nb::arg("scale_w"),
            nb::arg("first"), nb::arg("second"),
            nb::arg("require_missing_free") = true
        );
}
