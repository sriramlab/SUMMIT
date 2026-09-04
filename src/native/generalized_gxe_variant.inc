// Descriptor-owned generalized per-variant GxE LD-score executor.
//
// This file is included inside gxeldcore.cpp's anonymous namespace.  It uses
// DirectContext only as the mature descriptor/decode operator; all pair,
// annotation semantics remain in this generalized owner.  Jackknife blocks
// are deliberately absent: inference reduces the completed per-variant panel
// after both reference-genotype passes have finished.

struct GeneralizedGxEGemmShapeRecord {
    std::string operation;
    bool transpose_left = false;
    int rows = 0;
    int columns = 0;
    int reduction = 0;
    int64_t calls = 0;
    double wall_seconds = 0.0;
    double process_cpu_seconds = 0.0;
};

// std::vector<double>::resize value-initializes newly allocated storage. The
// dense decoder and singleton compaction loops overwrite every element, so
// that initialization is pure memory traffic and defeats parallel first-touch
// on large NUMA hosts. Keep ownership and logical-size checks without it.
struct GeneralizedUninitializedDoubleBuffer {
    void reserve(size_t requested) {
        if (requested <= capacity_) return;
        storage_.reset(new double[requested]);
        capacity_ = requested;
    }

    void resize(size_t requested) {
        reserve(requested);
        size_ = requested;
    }

    double* data() noexcept { return storage_.get(); }
    const double* data() const noexcept { return storage_.get(); }
    size_t size() const noexcept { return size_; }
    double& operator[](size_t index) noexcept { return storage_[index]; }
    const double& operator[](size_t index) const noexcept {
        return storage_[index];
    }

    void clear() noexcept {
        storage_.reset();
        capacity_ = 0;
        size_ = 0;
    }

private:
    std::unique_ptr<double[]> storage_;
    size_t capacity_ = 0;
    size_t size_ = 0;
};

double generalized_elapsed_seconds(
    const std::chrono::steady_clock::time_point& begin
) noexcept {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin
    ).count();
}

uint64_t generalized_process_peak_rss_bytes() noexcept {
#if defined(__linux__)
    try {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.compare(0, 6, "VmHWM:") != 0) continue;
            std::istringstream values(line.substr(6));
            uint64_t kibibytes = 0;
            std::string unit;
            if (!(values >> kibibytes >> unit) || unit != "kB" ||
                kibibytes > std::numeric_limits<uint64_t>::max() / 1024U) {
                return 0;
            }
            return kibibytes * 1024U;
        }
    } catch (...) {
    }
#endif
    return 0;
}

class GeneralizedGxELDScoreDirectContext {
public:
    GeneralizedGxELDScoreDirectContext(
        int bed_descriptor,
        int bim_descriptor,
        int fam_descriptor,
        nb::object row_sel,
        int ddof,
        nb_mat2f_ro<double> basis,
        nb_mat2f_ro<double> fixed_effect_basis,
        nb_mat2c_ro<double> annotations,
        nb_vec1_ro<double> annotation_masses,
        nb_mat2c_ro<int64_t> pair_table,
        nb_mat2c_ro<int64_t> component_table,
        nb_vec1_ro<int64_t> product_offsets,
        nb_mat2c_ro<int64_t> product_terms,
        uint64_t root_seed,
        uint64_t namespace_key,
        int64_t probe_offset,
        int probe_count,
        int variant_block_width,
        int source_probe_tile_width,
        int source_annotation_batch_width,
        int probe_tile_width,
        int target_annotation_batch_width,
        int same_person_sample_tile_width,
        uint64_t max_workspace_bytes,
        int decode_threads,
        int requested_threads,
        bool retain_base_sources,
        bool publish_component_kernel_diagonal,
        bool retain_contextual_sources,
        bool dense_blas_hybrid,
        bool precompute_rhs,
        const std::string& qualification_fault_phase,
        int qualification_fault_row,
        int qualification_fault_column,
        double qualification_fault_delta
    )
        : reader_(std::make_unique<DirectContext>(
              bed_descriptor, bim_descriptor, fam_descriptor,
              std::move(row_sel), ddof, decode_threads,
              max_workspace_bytes, requested_threads,
              DescriptorOnlyDirectContextTag{}
          )),
          rows_(reader_->n_),
          variants_(reader_->m_total_),
          q_(checked_blas_dim(basis.shape(1),
                              "generalized basis columns")),
          fixed_rank_(checked_blas_dim(
              fixed_effect_basis.shape(1),
              "generalized fixed-effect columns"
          )),
          annotations_count_(checked_blas_dim(
              annotations.shape(1), "generalized annotation columns"
          )),
          probes_(probe_count),
          root_seed_(root_seed),
          namespace_key_(namespace_key),
          probe_offset_(probe_offset),
          variant_block_width_(variant_block_width),
          source_probe_tile_width_(source_probe_tile_width),
          source_annotation_batch_width_(source_annotation_batch_width),
          probe_tile_width_(probe_tile_width),
          target_annotation_batch_width_(target_annotation_batch_width),
          same_person_sample_tile_width_(same_person_sample_tile_width),
          max_workspace_bytes_(max_workspace_bytes),
          threads_(requested_threads),
          decode_threads_(decode_threads),
          retain_base_sources_(retain_base_sources),
          publish_component_kernel_diagonal_(
              publish_component_kernel_diagonal
          ),
          retain_contextual_sources_(retain_contextual_sources),
          dense_blas_hybrid_(dense_blas_hybrid),
          precompute_rhs_(precompute_rhs),
          qualification_fault_phase_(qualification_fault_phase),
          qualification_fault_row_(qualification_fault_row),
          qualification_fault_column_(qualification_fault_column),
          qualification_fault_delta_(qualification_fault_delta) {
        validate_protected_gemm_threads(threads_);
        if (rows_ != checked_blas_dim(
                basis.shape(0), "generalized basis rows") ||
            rows_ != checked_blas_dim(
                fixed_effect_basis.shape(0),
                "generalized fixed-effect rows") ||
            variants_ != checked_blas_dim(
                annotations.shape(0), "generalized annotation rows") ||
            q_ <= 0 || q_ >= rows_ || fixed_rank_ < 0 ||
            fixed_rank_ >= rows_ || annotations_count_ <= 0 || probes_ < 2 ||
            probe_offset_ < 0 || variant_block_width_ <= 0 ||
            source_probe_tile_width_ <= 0 ||
            source_probe_tile_width_ > probes_ ||
            source_annotation_batch_width_ <= 0 ||
            source_annotation_batch_width_ > annotations_count_ ||
            probe_tile_width_ <= 0 || probe_tile_width_ > probes_ ||
            target_annotation_batch_width_ <= 0 ||
            target_annotation_batch_width_ > annotations_count_ ||
            same_person_sample_tile_width_ <= 0 ||
            max_workspace_bytes_ == 0) {
            throw std::runtime_error(
                "Generalized descriptor-native dimensions are inconsistent"
            );
        }
        const bool fault_enabled = qualification_fault_phase_ != "none";
        if ((qualification_fault_phase_ != "none" &&
             qualification_fault_phase_ != "pass1_nn" &&
             qualification_fault_phase_ != "pass2_tn") ||
            (fault_enabled && (!dense_blas_hybrid_ ||
                qualification_fault_row_ < 0 ||
                qualification_fault_column_ < 0 ||
                !std::isfinite(qualification_fault_delta_) ||
                qualification_fault_delta_ == 0.0)) ||
            (!fault_enabled && (qualification_fault_row_ != -1 ||
                qualification_fault_column_ != -1 ||
                qualification_fault_delta_ != 0.0))) {
            throw std::runtime_error(
                "Generalized qualification fault injection is invalid"
            );
        }
        basis_.assign(
            basis.data(),
            basis.data() + checked_mul(
                static_cast<size_t>(rows_), static_cast<size_t>(q_),
                "generalized basis"
            )
        );
        fixed_.assign(
            fixed_effect_basis.data(),
            fixed_effect_basis.data() + checked_mul(
                static_cast<size_t>(rows_),
                static_cast<size_t>(fixed_rank_),
                "generalized fixed effects"
            )
        );
        annotation_values_.assign(
            annotations.data(),
            annotations.data() + checked_mul(
                static_cast<size_t>(variants_),
                static_cast<size_t>(annotations_count_),
                "generalized annotations"
            )
        );
        annotation_masses_.assign(
            annotation_masses.data(),
            annotation_masses.data() + annotation_masses.shape(0)
        );
        pair_table_.assign(
            pair_table.data(), pair_table.data() + checked_mul(
                pair_table.shape(0), pair_table.shape(1),
                "generalized pair table"
            )
        );
        component_table_.assign(
            component_table.data(), component_table.data() + checked_mul(
                component_table.shape(0), component_table.shape(1),
                "generalized component table"
            )
        );
        product_offsets_.assign(
            product_offsets.data(),
            product_offsets.data() + product_offsets.shape(0)
        );
        product_terms_.assign(
            product_terms.data(), product_terms.data() + checked_mul(
                product_terms.shape(0), product_terms.shape(1),
                "generalized product terms"
            )
        );
        validate_inputs(
            pair_table, component_table, product_offsets, product_terms
        );
        singleton_annotation_rows_ = true;
        singleton_annotation_index_.resize(static_cast<size_t>(variants_));
        singleton_annotation_weight_.resize(static_cast<size_t>(variants_));
        for (int variant = 0; variant < variants_; ++variant) {
            int observed_annotation = -1;
            double observed_weight = 0.0;
            for (int annotation = 0; annotation < annotations_count_;
                 ++annotation) {
                const double value = annotation_values_[
                    static_cast<size_t>(variant)
                        * static_cast<size_t>(annotations_count_)
                    + static_cast<size_t>(annotation)
                ];
                if (value == 0.0) continue;
                if (observed_annotation >= 0) {
                    singleton_annotation_rows_ = false;
                    break;
                }
                observed_annotation = annotation;
                observed_weight = value;
            }
            if (observed_annotation < 0 || !singleton_annotation_rows_) {
                singleton_annotation_rows_ = false;
                break;
            }
            singleton_annotation_index_[static_cast<size_t>(variant)] =
                observed_annotation;
            singleton_annotation_weight_[static_cast<size_t>(variant)] =
                observed_weight;
        }
        if (!singleton_annotation_rows_) {
            singleton_annotation_index_.clear();
            singleton_annotation_weight_.clear();
        }
        reader_->check_files_unchanged();
    }

    GeneralizedGxELDScoreDirectContext(
        const GeneralizedGxELDScoreDirectContext&) = delete;
    GeneralizedGxELDScoreDirectContext& operator=(
        const GeneralizedGxELDScoreDirectContext&) = delete;

    nb::dict progress() const {
        int64_t generation_before = 0;
        int64_t generation_after = 0;
        int phase = 0;
        int subphase = 0;
        int64_t completed_blocks = 0;
        int64_t total_blocks = 0;
        int64_t current_block = -1;
        int64_t variant_start = -1;
        int64_t variant_stop = -1;
        int64_t completed_units = 0;
        int64_t total_units = 0;
        int64_t started_nanoseconds = 0;
        int64_t heartbeat_nanoseconds = 0;
        do {
            generation_before = progress_generation_.load(
                std::memory_order_acquire
            );
            if ((generation_before & 1LL) != 0) {
                std::this_thread::yield();
                continue;
            }
            phase = progress_phase_.load(std::memory_order_relaxed);
            subphase = progress_subphase_.load(std::memory_order_relaxed);
            completed_blocks = progress_completed_blocks_.load(
                std::memory_order_relaxed
            );
            total_blocks = progress_total_blocks_.load(
                std::memory_order_relaxed
            );
            current_block = progress_current_block_.load(
                std::memory_order_relaxed
            );
            variant_start = progress_variant_start_.load(
                std::memory_order_relaxed
            );
            variant_stop = progress_variant_stop_.load(
                std::memory_order_relaxed
            );
            completed_units = progress_completed_units_.load(
                std::memory_order_relaxed
            );
            total_units = progress_total_units_.load(
                std::memory_order_relaxed
            );
            started_nanoseconds = progress_started_nanoseconds_.load(
                std::memory_order_relaxed
            );
            heartbeat_nanoseconds = progress_heartbeat_nanoseconds_.load(
                std::memory_order_relaxed
            );
            generation_after = progress_generation_.load(
                std::memory_order_acquire
            );
        } while (generation_before != generation_after ||
                 (generation_after & 1LL) != 0);

        const int64_t now = steady_nanoseconds_now();
        nb::dict result;
        result["schema"] =
            "summit.generalized_gxe.native_progress.v1";
        result["phase"] = progress_phase_name(
            static_cast<ProgressPhase>(phase)
        );
        result["subphase"] = progress_subphase_name(
            static_cast<ProgressSubphase>(subphase)
        );
        result["active"] = progress_active_.load(
            std::memory_order_acquire
        );
        result["cancel_requested"] = cancel_requested_.load(
            std::memory_order_acquire
        );
        result["completed_blocks"] = completed_blocks;
        result["total_blocks"] = total_blocks;
        result["current_block"] = current_block;
        result["variant_start"] = variant_start;
        result["variant_stop"] = variant_stop;
        result["completed_units"] = completed_units;
        result["total_units"] = total_units;
        result["elapsed_seconds"] = started_nanoseconds > 0
            ? static_cast<double>(std::max<int64_t>(
                  0, now - started_nanoseconds
              )) / 1.0e9
            : 0.0;
        result["heartbeat_age_seconds"] = heartbeat_nanoseconds > 0
            ? static_cast<double>(std::max<int64_t>(
                  0, now - heartbeat_nanoseconds
              )) / 1.0e9
            : 0.0;
        result["peak_rss_bytes"] = generalized_process_peak_rss_bytes();
        return result;
    }

    void request_cancel() noexcept {
        if (progress_active_.load(std::memory_order_acquire)) {
            cancel_requested_.store(true, std::memory_order_release);
        }
    }

    nb::dict info() const {
        std::lock_guard<std::mutex> guard(mutex_);
        nb::dict result;
        result["schema"] =
            "summit.generalized_gxe.descriptor_native_context.v1";
        result["state"] = state_name(state_);
        result["n_selected"] = rows_;
        result["m_variants"] = variants_;
        result["q_basis"] = q_;
        result["fixed_rank"] = fixed_rank_;
        result["annotations"] = annotations_count_;
        result["pairs"] = pairs_;
        result["components"] = components_;
        result["probes"] = probes_;
        result["variant_block_width"] = variant_block_width_;
        result["source_probe_tile_width"] = source_probe_tile_width_;
        result["source_annotation_batch_width"] =
            source_annotation_batch_width_;
        result["target_probe_tile_width"] = probe_tile_width_;
        result["target_annotation_batch_width"] =
            target_annotation_batch_width_;
        result["probe_tile_width"] = probe_tile_width_;
        result["planned_complete_passes"] = 2;
        result["observed_complete_passes"] = observed_passes_;
        result["observed_variant_visits"] = observed_variant_visits_;
        result["observed_block_reads"] = observed_block_reads_;
        result["source_nn_calls"] = source_nn_calls_;
        result["target_tn_calls"] = target_tn_calls_;
        result["repaired_columns"] = repaired_columns_;
        result["retry_count"] = retry_count_;
        result["fallback_count"] = fallback_count_;
        result["integrity_audit_count"] = integrity_audit_count_;
        result["protected_integrity_audit_count"] =
            protected_integrity_audit_count_;
        result["independent_integrity_audit_count"] =
            independent_integrity_audit_count_;
        result["integrity_failures"] = integrity_failures_;
        result["qualification_fault_injected"] =
            qualification_fault_injected_;
        result["packed_backend_used"] = !dense_blas_hybrid_;
        result["rhs_precomputed"] = precompute_rhs_;
        result["component_kernel_diagonal_published"] =
            publish_component_kernel_diagonal_;
        result["singleton_annotation_rows"] = singleton_annotation_rows_;
        result["contextual_sources_published"] =
            retain_contextual_sources_;
        result["backend"] = dense_blas_hybrid_
            ? "descriptor_dense_protected_nn_tn"
            : "descriptor_packed_mailman";
        result["genotype_scale"] =
            "mean_imputed_single_affine_ddof_fp64_v1";
        result["internal_source_layout"] =
            "annotation_coordinate_probe_sample_v1";
        return result;
    }

    nb::dict run() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (state_ != State::constructed) {
            throw std::runtime_error(
                "Generalized descriptor-native context is single-use"
            );
        }
        cancel_requested_.store(false, std::memory_order_release);
        progress_started_nanoseconds_.store(
            steady_nanoseconds_now(), std::memory_order_relaxed
        );
        progress_total_blocks_.store(
            (static_cast<int64_t>(variants_) + variant_block_width_ - 1) /
                variant_block_width_,
            std::memory_order_relaxed
        );
        progress_active_.store(true, std::memory_order_release);
        set_progress(
            ProgressPhase::constructed, ProgressSubphase::idle,
            0, -1, -1, -1, 0, 0
        );
        try {
            reader_->check_files_unchanged();
            initialize_outputs();
            {
                nb::gil_scoped_release release;
                {
                    const auto wall = std::chrono::steady_clock::now();
                    const double cpu = process_cpu_seconds_now();
                    execute_pass1();
                    phase_pass1_wall_seconds_ +=
                        generalized_elapsed_seconds(wall);
                    phase_pass1_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - cpu
                    );
                }
                {
                    const auto wall = std::chrono::steady_clock::now();
                    const double cpu = process_cpu_seconds_now();
                    seal_sources();
                    phase_barrier_wall_seconds_ +=
                        generalized_elapsed_seconds(wall);
                    phase_barrier_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - cpu
                    );
                }
                {
                    const auto wall = std::chrono::steady_clock::now();
                    const double cpu = process_cpu_seconds_now();
                    execute_pass2();
                    phase_pass2_wall_seconds_ +=
                        generalized_elapsed_seconds(wall);
                    phase_pass2_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - cpu
                    );
                }
                {
                    const auto wall = std::chrono::steady_clock::now();
                    const double cpu = process_cpu_seconds_now();
                    finalize_scientific_outputs();
                    reader_->check_files_unchanged();
                    phase_finalize_wall_seconds_ +=
                        generalized_elapsed_seconds(wall);
                    phase_finalize_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - cpu
                    );
                }
            }
            if (observed_passes_ != 2 ||
                observed_variant_visits_ != 2LL * variants_) {
                throw std::runtime_error(
                    "Generalized descriptor-native traversal count mismatch"
                );
            }
            state_ = State::finalized;
            nb::dict result = publish_result();
            const int64_t total_blocks = progress_total_blocks_.load(
                std::memory_order_relaxed
            );
            set_progress(
                ProgressPhase::complete, ProgressSubphase::idle,
                total_blocks, -1, variants_, variants_, 0, 0
            );
            progress_active_.store(false, std::memory_order_release);
            return result;
        } catch (...) {
            state_ = State::failed;
            const bool cancelled = cancel_requested_.load(
                std::memory_order_acquire
            );
            set_progress(
                cancelled ? ProgressPhase::cancelled
                          : ProgressPhase::failed,
                ProgressSubphase::idle,
                progress_completed_blocks_.load(std::memory_order_relaxed),
                progress_current_block_.load(std::memory_order_relaxed),
                progress_variant_start_.load(std::memory_order_relaxed),
                progress_variant_stop_.load(std::memory_order_relaxed),
                progress_completed_units_.load(std::memory_order_relaxed),
                progress_total_units_.load(std::memory_order_relaxed)
            );
            progress_active_.store(false, std::memory_order_release);
            clear_publishable_outputs();
            throw;
        }
    }

private:
    enum class ProgressPhase : int {
        constructed,
        pass1,
        barrier,
        pass2,
        finalize,
        complete,
        cancelled,
        failed,
    };

    enum class ProgressSubphase : int {
        idle,
        decode,
        source_sketch,
        source_projection,
        component_diagonal,
        rhs_construction,
        target_product,
        row_reduction,
        output_reduction,
        scientific_reduction,
        rhs_precompute,
    };

    enum class State {
        constructed,
        pass1_running,
        sources_sealed,
        pass2_running,
        finalized,
        failed,
    };

    static const char* state_name(State state) noexcept {
        switch (state) {
            case State::constructed: return "constructed";
            case State::pass1_running: return "pass1_running";
            case State::sources_sealed: return "sources_sealed";
            case State::pass2_running: return "pass2_running";
            case State::finalized: return "finalized";
            case State::failed: return "failed";
        }
        return "invalid";
    }

    static const char* progress_phase_name(ProgressPhase phase) noexcept {
        switch (phase) {
            case ProgressPhase::constructed: return "constructed";
            case ProgressPhase::pass1: return "pass1";
            case ProgressPhase::barrier: return "barrier";
            case ProgressPhase::pass2: return "pass2";
            case ProgressPhase::finalize: return "finalize";
            case ProgressPhase::complete: return "complete";
            case ProgressPhase::cancelled: return "cancelled";
            case ProgressPhase::failed: return "failed";
        }
        return "invalid";
    }

    static const char* progress_subphase_name(
        ProgressSubphase subphase
    ) noexcept {
        switch (subphase) {
            case ProgressSubphase::idle: return "idle";
            case ProgressSubphase::decode: return "decode";
            case ProgressSubphase::source_sketch: return "source_sketch";
            case ProgressSubphase::source_projection:
                return "source_projection";
            case ProgressSubphase::component_diagonal:
                return "component_diagonal";
            case ProgressSubphase::rhs_construction:
                return "rhs_construction";
            case ProgressSubphase::target_product: return "target_product";
            case ProgressSubphase::row_reduction: return "row_reduction";
            case ProgressSubphase::output_reduction:
                return "output_reduction";
            case ProgressSubphase::scientific_reduction:
                return "scientific_reduction";
            case ProgressSubphase::rhs_precompute: return "rhs_precompute";
        }
        return "invalid";
    }

    static int64_t steady_nanoseconds_now() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    void set_progress(
        ProgressPhase phase,
        ProgressSubphase subphase,
        int64_t completed_blocks,
        int64_t current_block,
        int64_t variant_start,
        int64_t variant_stop,
        int64_t completed_units,
        int64_t total_units
    ) noexcept {
        progress_generation_.fetch_add(1, std::memory_order_acq_rel);
        progress_phase_.store(
            static_cast<int>(phase), std::memory_order_relaxed
        );
        progress_subphase_.store(
            static_cast<int>(subphase), std::memory_order_relaxed
        );
        progress_completed_blocks_.store(
            completed_blocks, std::memory_order_relaxed
        );
        progress_current_block_.store(
            current_block, std::memory_order_relaxed
        );
        progress_variant_start_.store(
            variant_start, std::memory_order_relaxed
        );
        progress_variant_stop_.store(
            variant_stop, std::memory_order_relaxed
        );
        progress_completed_units_.store(
            completed_units, std::memory_order_relaxed
        );
        progress_total_units_.store(total_units, std::memory_order_relaxed);
        progress_heartbeat_nanoseconds_.store(
            steady_nanoseconds_now(), std::memory_order_relaxed
        );
        progress_generation_.fetch_add(1, std::memory_order_release);
    }

    void set_progress_units(
        int64_t completed_units, int64_t total_units
    ) noexcept {
        progress_generation_.fetch_add(1, std::memory_order_acq_rel);
        progress_completed_units_.store(
            completed_units, std::memory_order_relaxed
        );
        progress_total_units_.store(total_units, std::memory_order_relaxed);
        progress_heartbeat_nanoseconds_.store(
            steady_nanoseconds_now(), std::memory_order_relaxed
        );
        progress_generation_.fetch_add(1, std::memory_order_release);
    }

    void check_cancelled() const {
        if (cancel_requested_.load(std::memory_order_acquire)) {
            throw std::runtime_error(
                "Generalized descriptor-native execution was cancelled"
            );
        }
    }

    static nb_numpy_vec1<double> move_vector(
        std::vector<double>& source
    ) {
        auto* storage = new std::vector<double>();
        storage->swap(source);
        const size_t elements = storage->size();
        nb::capsule owner(storage, [](void* pointer) noexcept {
            delete static_cast<std::vector<double>*>(pointer);
        });
        return nb_numpy_vec1<double>(storage->data(), {elements}, owner);
    }

    static nb_numpy_mat2c<double> move_matrix(
        std::vector<double>& source, size_t rows, size_t columns
    ) {
        if (source.size() != checked_mul(rows, columns, "published matrix")) {
            throw std::runtime_error("Published generalized matrix is mis-sized");
        }
        auto* storage = new std::vector<double>();
        storage->swap(source);
        nb::capsule owner(storage, [](void* pointer) noexcept {
            delete static_cast<std::vector<double>*>(pointer);
        });
        return nb_numpy_mat2c<double>(
            storage->data(), {rows, columns}, owner
        );
    }

    nb_numpy_vec1<double> copy_base_sources_for_publication() const {
        double* destination = nullptr;
        auto result = make_owned_numpy_vec1<double>(
            base_sources_.size(), &destination
        );
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_)
#endif
        for (int annotation = 0; annotation < annotations_count_;
             ++annotation) {
            for (int row = 0; row < rows_; ++row) {
                for (int probe = 0; probe < probes_; ++probe) {
                    destination[
                        (static_cast<size_t>(annotation)
                             * static_cast<size_t>(rows_)
                         + static_cast<size_t>(row))
                            * static_cast<size_t>(probes_)
                        + static_cast<size_t>(probe)
                    ] = base_sources_[base_index(annotation, row, probe)];
                }
            }
        }
        return result;
    }

    nb_numpy_vec1<double> copy_contextual_sources_for_publication() const {
        double* destination = nullptr;
        auto result = make_owned_numpy_vec1<double>(
            contextual_sources_.size(), &destination
        );
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) num_threads(threads_)
#endif
        for (int annotation = 0; annotation < annotations_count_;
             ++annotation) {
            for (int coordinate = 0; coordinate < q_; ++coordinate) {
                for (int row = 0; row < rows_; ++row) {
                    for (int probe = 0; probe < probes_; ++probe) {
                        destination[
                            ((static_cast<size_t>(annotation)
                                  * static_cast<size_t>(q_)
                              + static_cast<size_t>(coordinate))
                                 * static_cast<size_t>(rows_)
                             + static_cast<size_t>(row))
                                * static_cast<size_t>(probes_)
                            + static_cast<size_t>(probe)
                        ] = contextual_sources_[source_index(
                            annotation, coordinate, row, probe
                        )];
                    }
                }
            }
        }
        return result;
    }

    size_t base_index(int annotation, int row, int probe) const noexcept {
        return (static_cast<size_t>(annotation) * static_cast<size_t>(probes_)
                + static_cast<size_t>(probe)) * static_cast<size_t>(rows_)
            + static_cast<size_t>(row);
    }

    size_t source_index(
        int annotation, int coordinate, int row, int probe
    ) const noexcept {
        return (((static_cast<size_t>(annotation) * static_cast<size_t>(q_)
                  + static_cast<size_t>(coordinate))
                 * static_cast<size_t>(probes_)
                 + static_cast<size_t>(probe))
                * static_cast<size_t>(rows_))
            + static_cast<size_t>(row);
    }

    size_t directional_index(
        int variant, int target_pair, int component
    ) const noexcept {
        return (static_cast<size_t>(variant) * static_cast<size_t>(pairs_)
                + static_cast<size_t>(target_pair))
                * static_cast<size_t>(components_)
            + static_cast<size_t>(component);
    }

    int pair_q(int pair) const noexcept {
        return static_cast<int>(pair_table_[2U * static_cast<size_t>(pair)]);
    }

    int pair_r(int pair) const noexcept {
        return static_cast<int>(
            pair_table_[2U * static_cast<size_t>(pair) + 1U]
        );
    }

    double pair_factor(int pair) const noexcept {
        return pair_q(pair) == pair_r(pair) ? 1.0 : 2.0;
    }

    struct ComponentDiagonalWorkspace {
        std::vector<double> coefficients;
        std::vector<double> fixed_tile;
        GeneralizedUninitializedDoubleBuffer genotype_tile;
        GeneralizedUninitializedDoubleBuffer feature_tile;
        GeneralizedUninitializedDoubleBuffer pair_tile;
        GeneralizedUninitializedDoubleBuffer annotation_product;
    };

    void validate_inputs(
        nb_mat2c_ro<int64_t> pair_table,
        nb_mat2c_ro<int64_t> component_table,
        nb_vec1_ro<int64_t> product_offsets,
        nb_mat2c_ro<int64_t> product_terms
    );
    void initialize_outputs();
    void decode_block(
        int start,
        int stop,
        GeneralizedUninitializedDoubleBuffer& genotype
    );
    void read_packed_block(int start, int stop, MailmanPackedBlock& packed);
    void packed_source_product(
        const MailmanPackedBlock& packed,
        const std::vector<double>& rhs,
        int columns,
        std::vector<double>& output
    ) const;
    void packed_target_product(
        const MailmanPackedBlock& packed,
        const double* rhs,
        int columns,
        std::vector<double>& output
    ) const;
    void build_precomputed_rhs();
    void accumulate_component_kernel_diagonal(
        int start,
        int stop,
        const GeneralizedUninitializedDoubleBuffer* genotype,
        const MailmanPackedBlock* packed,
        ComponentDiagonalWorkspace& workspace
    );
    void execute_pass1();
    void seal_sources();
    void execute_pass2();
    void finalize_scientific_outputs();
    void record_gemm_shape(
        const char* operation,
        bool transpose_left,
        int rows,
        int columns,
        int reduction,
        double wall_seconds,
        double process_cpu_seconds
    );
    void record_integrity_resolution(
        const GemmIntegrityResolution& resolution
    ) noexcept;
    void audit_nn_product(
        const double* left,
        int rows,
        int reduction,
        const double* right,
        int columns,
        const double* output,
        int call_index
    );
    void audit_tn_product(
        const double* left,
        int rows,
        int reduction,
        const double* right,
        int columns,
        const double* output,
        int call_index
    );
    nb::dict publish_result();
    void clear_publishable_outputs() noexcept;

    std::unique_ptr<DirectContext> reader_;
    int rows_ = 0;
    int variants_ = 0;
    int q_ = 0;
    int fixed_rank_ = 0;
    int annotations_count_ = 0;
    int pairs_ = 0;
    int components_ = 0;
    int probes_ = 0;
    int residual_rank_ = 0;
    uint64_t root_seed_ = 0;
    uint64_t namespace_key_ = 0;
    int64_t probe_offset_ = 0;
    int variant_block_width_ = 0;
    int source_probe_tile_width_ = 0;
    int source_annotation_batch_width_ = 1;
    int probe_tile_width_ = 0;
    int target_annotation_batch_width_ = 1;
    int same_person_sample_tile_width_ = 0;
    uint64_t max_workspace_bytes_ = 0;
    int threads_ = 1;
    int decode_threads_ = 1;
    bool retain_base_sources_ = false;
    bool publish_component_kernel_diagonal_ = false;
    bool retain_contextual_sources_ = false;
    bool dense_blas_hybrid_ = true;
    bool precompute_rhs_ = false;
    std::string qualification_fault_phase_ = "none";
    int qualification_fault_row_ = -1;
    int qualification_fault_column_ = -1;
    double qualification_fault_delta_ = 0.0;
    bool qualification_fault_injected_ = false;
    State state_ = State::constructed;
    mutable std::mutex mutex_;
    std::atomic<int64_t> progress_generation_{0};
    std::atomic<int> progress_phase_{
        static_cast<int>(ProgressPhase::constructed)
    };
    std::atomic<int> progress_subphase_{
        static_cast<int>(ProgressSubphase::idle)
    };
    std::atomic<bool> progress_active_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<int64_t> progress_completed_blocks_{0};
    std::atomic<int64_t> progress_total_blocks_{0};
    std::atomic<int64_t> progress_current_block_{-1};
    std::atomic<int64_t> progress_variant_start_{-1};
    std::atomic<int64_t> progress_variant_stop_{-1};
    std::atomic<int64_t> progress_completed_units_{0};
    std::atomic<int64_t> progress_total_units_{0};
    std::atomic<int64_t> progress_started_nanoseconds_{0};
    std::atomic<int64_t> progress_heartbeat_nanoseconds_{0};

    std::vector<double> basis_;
    std::vector<double> fixed_;
    std::vector<double> annotation_values_;
    std::vector<double> sqrt_annotation_values_;
    std::vector<double> annotation_masses_;
    bool singleton_annotation_rows_ = false;
    std::vector<int> singleton_annotation_index_;
    std::vector<double> singleton_annotation_weight_;
    std::vector<int64_t> pair_table_;
    std::vector<int64_t> component_table_;
    std::vector<int64_t> product_offsets_;
    std::vector<int64_t> product_terms_;
    GeneralizedUninitializedDoubleBuffer base_sources_;
    GeneralizedUninitializedDoubleBuffer contextual_sources_;
    std::vector<double> weighted_fixed_;
    std::vector<double> component_kernel_diagonal_;
    std::vector<double> same_person_;
    std::vector<double> directional_;
    std::vector<double> directed_;
    std::vector<double> symmetric_;
    std::vector<double> genetic_gram_;
    std::vector<double> affine_mean_;
    std::vector<double> affine_inverse_scale_;
    std::unique_ptr<ReadOnlyDoubleMapping> precomputed_rhs_;

    int64_t observed_passes_ = 0;
    int64_t observed_variant_visits_ = 0;
    int64_t observed_block_reads_ = 0;
    int64_t missing_calls_ = 0;
    int64_t source_nn_calls_ = 0;
    int64_t target_tn_calls_ = 0;
    int64_t repaired_columns_ = 0;
    int64_t checksum_recomputed_columns_ = 0;
    int64_t roundoff_only_columns_ = 0;
    int64_t integrity_audit_count_ = 0;
    int64_t protected_integrity_audit_count_ = 0;
    int64_t independent_integrity_audit_count_ = 0;
    int64_t integrity_failures_ = 0;
    int64_t retry_count_ = 0;
    int64_t fallback_count_ = 0;
    int component_pair_batch_width_ = 0;
    int rhs_precomputed_columns_ = 0;
    uint64_t rhs_precomputed_bytes_ = 0;
    double maximum_projection_leakage_ = 0.0;
    double maximum_relative_projection_leakage_ = 0.0;
    double same_person_presymmetry_error_ = 0.0;
    double presymmetry_absolute_error_ = 0.0;
    double presymmetry_relative_error_ = 0.0;
    double phase_pass1_wall_seconds_ = 0.0;
    double phase_barrier_wall_seconds_ = 0.0;
    double phase_pass2_wall_seconds_ = 0.0;
    double phase_finalize_wall_seconds_ = 0.0;
    double phase_pass1_cpu_seconds_ = 0.0;
    double phase_barrier_cpu_seconds_ = 0.0;
    double phase_pass2_cpu_seconds_ = 0.0;
    double phase_finalize_cpu_seconds_ = 0.0;
    double decode_wall_seconds_ = 0.0;
    double decode_cpu_seconds_ = 0.0;
    double rhs_wall_seconds_ = 0.0;
    double rhs_cpu_seconds_ = 0.0;
    double rhs_precompute_wall_seconds_ = 0.0;
    double rhs_precompute_cpu_seconds_ = 0.0;
    double source_nn_wall_seconds_ = 0.0;
    double source_nn_cpu_seconds_ = 0.0;
    double projection_wall_seconds_ = 0.0;
    double projection_cpu_seconds_ = 0.0;
    double same_person_wall_seconds_ = 0.0;
    double same_person_cpu_seconds_ = 0.0;
    double component_diagonal_wall_seconds_ = 0.0;
    double component_diagonal_cpu_seconds_ = 0.0;
    double target_tn_wall_seconds_ = 0.0;
    double target_tn_cpu_seconds_ = 0.0;
    double reduction_wall_seconds_ = 0.0;
    double reduction_cpu_seconds_ = 0.0;
    double directed_aggregation_wall_seconds_ = 0.0;
    double directed_aggregation_cpu_seconds_ = 0.0;
    double independent_audit_wall_seconds_ = 0.0;
    double independent_audit_cpu_seconds_ = 0.0;
    double source_independent_audit_wall_seconds_ = 0.0;
    double source_independent_audit_cpu_seconds_ = 0.0;
    double target_independent_audit_wall_seconds_ = 0.0;
    double target_independent_audit_cpu_seconds_ = 0.0;
    double publication_wall_seconds_ = 0.0;
    double publication_cpu_seconds_ = 0.0;
    std::vector<GeneralizedGxEGemmShapeRecord> gemm_shapes_;
    uint64_t gemm_output_numa_call_count_ = 0;
    std::vector<std::shared_ptr<SharedNativeGemmOutputNumaEvidence>>
        gemm_output_numa_evidence_;
};

void GeneralizedGxELDScoreDirectContext::validate_inputs(
    nb_mat2c_ro<int64_t> pair_table,
    nb_mat2c_ro<int64_t> component_table,
    nb_vec1_ro<int64_t> product_offsets,
    nb_mat2c_ro<int64_t> product_terms
) {
    const size_t expected_pairs = checked_add(
        static_cast<size_t>(q_),
        checked_mul(
            static_cast<size_t>(q_), static_cast<size_t>(q_ - 1),
            "generalized off-diagonal pairs"
        ) / 2U,
        "generalized pairs"
    );
    pairs_ = checked_blas_dim(expected_pairs, "generalized pair count");
    components_ = checked_blas_dim(
        checked_mul(
            static_cast<size_t>(annotations_count_), expected_pairs,
            "generalized component count"
        ),
        "generalized component count"
    );
    if (pair_table.shape(0) != expected_pairs || pair_table.shape(1) != 2 ||
        component_table.shape(0) != static_cast<size_t>(components_) ||
        component_table.shape(1) != 2 ||
        product_offsets.shape(0) != checked_add(
            checked_mul(expected_pairs, expected_pairs,
                        "generalized product cells"),
            1U, "generalized product offsets"
        ) || product_terms.shape(1) != 4) {
        throw std::runtime_error(
            "Generalized immutable scientific tables are mis-sized"
        );
    }

    int expected_pair = 0;
    for (int coordinate = 0; coordinate < q_; ++coordinate) {
        if (pair_q(expected_pair) != coordinate ||
            pair_r(expected_pair) != coordinate) {
            throw std::runtime_error(
                "Generalized pair table is not in canonical order"
            );
        }
        ++expected_pair;
    }
    for (int left = 0; left < q_; ++left) {
        for (int right = left + 1; right < q_; ++right) {
            if (pair_q(expected_pair) != left ||
                pair_r(expected_pair) != right) {
                throw std::runtime_error(
                    "Generalized pair table is not in canonical order"
                );
            }
            ++expected_pair;
        }
    }
    for (int annotation = 0; annotation < annotations_count_; ++annotation) {
        for (int pair = 0; pair < pairs_; ++pair) {
            const int component = annotation * pairs_ + pair;
            if (component_table_[2U * static_cast<size_t>(component)] !=
                    annotation ||
                component_table_[2U * static_cast<size_t>(component) + 1U] !=
                    pair) {
                throw std::runtime_error(
                    "Generalized component table is not annotation-major"
                );
            }
        }
    }

    if (product_offsets_[0] != 0 ||
        product_offsets_.back() !=
            static_cast<int64_t>(product_terms.shape(0))) {
        throw std::runtime_error(
            "Generalized product offsets do not delimit the term table"
        );
    }
    for (int target_pair = 0; target_pair < pairs_; ++target_pair) {
        const int tq = pair_q(target_pair);
        const int tr = pair_r(target_pair);
        std::array<std::pair<int, int>, 2> target_orientations{{
            {tq, tq}, {tr, tq}
        }};
        const int target_count = tq == tr ? 1 : 2;
        if (target_count == 2) {
            target_orientations[0] = {tq, tr};
            target_orientations[1] = {tr, tq};
        }
        for (int source_pair = 0; source_pair < pairs_; ++source_pair) {
            const int sq = pair_q(source_pair);
            const int sr = pair_r(source_pair);
            std::array<std::pair<int, int>, 2> source_orientations{{
                {sq, sq}, {sr, sq}
            }};
            const int source_count = sq == sr ? 1 : 2;
            if (source_count == 2) {
                source_orientations[0] = {sq, sr};
                source_orientations[1] = {sr, sq};
            }
            const int cell = target_pair * pairs_ + source_pair;
            const int64_t start = product_offsets_[
                static_cast<size_t>(cell)
            ];
            const int64_t stop = product_offsets_[
                static_cast<size_t>(cell + 1)
            ];
            if (start < 0 || stop < start ||
                stop - start != target_count * source_count) {
                throw std::runtime_error(
                    "Generalized product multiplicity is invalid"
                );
            }
            int64_t term_index = start;
            for (int target = 0; target < target_count; ++target) {
                for (int source = 0; source < source_count;
                     ++source, ++term_index) {
                    const size_t offset =
                        4U * static_cast<size_t>(term_index);
                    const std::array<int64_t, 4> expected{{
                        target_orientations[static_cast<size_t>(target)].second,
                        source_orientations[static_cast<size_t>(source)].first,
                        target_orientations[static_cast<size_t>(target)].first,
                        source_orientations[static_cast<size_t>(source)].second,
                    }};
                    for (size_t coordinate = 0; coordinate < 4; ++coordinate) {
                        if (product_terms_[offset + coordinate] !=
                            expected[coordinate]) {
                            throw std::runtime_error(
                                "Generalized product term table is not canonical"
                            );
                        }
                    }
                }
            }
        }
    }

    for (double value : basis_) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Generalized basis is non-finite");
        }
    }
    for (double value : fixed_) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Generalized fixed effects are non-finite");
        }
    }
    for (double value : annotation_values_) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error(
                "Generalized annotations must be finite and nonnegative"
            );
        }
    }
    if (annotation_masses_.size() !=
        static_cast<size_t>(annotations_count_)) {
        throw std::runtime_error(
            "Generalized annotation masses are mis-sized"
        );
    }
    for (int annotation = 0; annotation < annotations_count_; ++annotation) {
        long double total = 0.0L;
        long double magnitude = 0.0L;
        for (int variant = 0; variant < variants_; ++variant) {
            const double value = annotation_values_[
                static_cast<size_t>(variant)
                    * static_cast<size_t>(annotations_count_)
                + static_cast<size_t>(annotation)
            ];
            total += static_cast<long double>(value);
            magnitude += static_cast<long double>(std::abs(value));
        }
        const long double supplied = annotation_masses_[
            static_cast<size_t>(annotation)
        ];
        const long double tolerance = 64.0L
            * static_cast<long double>(std::numeric_limits<double>::epsilon())
            * magnitude
            + static_cast<long double>(std::numeric_limits<double>::min());
        if (!std::isfinite(static_cast<double>(supplied)) || supplied <= 0.0L ||
            std::abs(supplied - total) > tolerance) {
            throw std::runtime_error(
                "Generalized annotation masses disagree with annotations"
            );
        }
    }

    residual_rank_ = rows_ - fixed_rank_;
    if (residual_rank_ < 1) {
        throw std::runtime_error(
            "Generalized fixed effects leave no residual rank"
        );
    }
    if (fixed_rank_ > 0) {
        for (int left = 0; left < fixed_rank_; ++left) {
            for (int right = 0; right < fixed_rank_; ++right) {
                long double dot = 0.0L;
                for (int row = 0; row < rows_; ++row) {
                    dot += static_cast<long double>(fixed_[
                        static_cast<size_t>(left) * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ]) * static_cast<long double>(fixed_[
                        static_cast<size_t>(right) * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ]);
                }
                const long double expected = left == right ? 1.0L : 0.0L;
                if (std::abs(dot - expected) > 2.0e-10L) {
                    throw std::runtime_error(
                        "Generalized fixed-effect basis is not orthonormal"
                    );
                }
            }
        }
    }
}

void GeneralizedGxELDScoreDirectContext::record_gemm_shape(
    const char* operation,
    bool transpose_left,
    int rows,
    int columns,
    int reduction,
    double wall_seconds,
    double process_cpu_seconds
) {
    for (GeneralizedGxEGemmShapeRecord& record : gemm_shapes_) {
        if (record.operation == operation &&
            record.transpose_left == transpose_left &&
            record.rows == rows && record.columns == columns &&
            record.reduction == reduction) {
            ++record.calls;
            record.wall_seconds += wall_seconds;
            record.process_cpu_seconds += process_cpu_seconds;
            return;
        }
    }
    GeneralizedGxEGemmShapeRecord record;
    record.operation = operation;
    record.transpose_left = transpose_left;
    record.rows = rows;
    record.columns = columns;
    record.reduction = reduction;
    record.calls = 1;
    record.wall_seconds = wall_seconds;
    record.process_cpu_seconds = process_cpu_seconds;
    gemm_shapes_.push_back(std::move(record));
}

void GeneralizedGxELDScoreDirectContext::record_integrity_resolution(
    const GemmIntegrityResolution& resolution
) noexcept {
    checksum_recomputed_columns_ += resolution.checksum_recomputed_columns;
    repaired_columns_ += resolution.materially_repaired_columns;
    roundoff_only_columns_ += resolution.roundoff_only_columns();
}

void GeneralizedGxELDScoreDirectContext::audit_nn_product(
    const double* left,
    int rows,
    int reduction,
    const double* right,
    int columns,
    const double* output,
    int call_index
) {
    const bool protected_checksum = gemm_requires_integrity_checks(
        rows, columns, reduction
    );
    if (protected_checksum) {
        ++protected_integrity_audit_count_;
        ++integrity_audit_count_;
    }
    const bool qualification_fault_pending =
        qualification_fault_phase_ == "pass1_nn" &&
        !qualification_fault_injected_;
    if (protected_checksum && !qualification_fault_pending) return;
    if (call_index % 64 != 0 && !qualification_fault_pending) return;
    if (qualification_fault_pending) {
        if (qualification_fault_row_ >= rows ||
            qualification_fault_column_ >= columns) {
            throw std::runtime_error(
                "Generalized pass-1 qualification fault coordinate is out of range"
            );
        }
        const_cast<double*>(output)[
            static_cast<size_t>(qualification_fault_column_)
                * static_cast<size_t>(rows)
            + static_cast<size_t>(qualification_fault_row_)
        ] += qualification_fault_delta_;
        qualification_fault_injected_ = true;
    }
    const auto coefficient = [](int check, int index) noexcept {
        if (check == 0) return index % 2 == 0 ? 1.0L : -1.0L;
        const int value = ((index * 5 + 3) % 11) - 5;
        return static_cast<long double>(value == 0 ? 2 : value);
    };
    for (int check = 0; check < 2; ++check) {
        std::vector<long double> projected(
            static_cast<size_t>(reduction), 0.0L
        );
        for (int inner = 0; inner < reduction; ++inner) {
            for (int row = 0; row < rows; ++row) {
                projected[static_cast<size_t>(inner)] +=
                    coefficient(check, row) * static_cast<long double>(left[
                        static_cast<size_t>(inner) * static_cast<size_t>(rows)
                        + static_cast<size_t>(row)
                    ]);
            }
        }
        for (int column = 0; column < columns; ++column) {
            long double expected = 0.0L;
            long double observed = 0.0L;
            long double absolute_bound = 0.0L;
            for (int inner = 0; inner < reduction; ++inner) {
                const long double term =
                    projected[static_cast<size_t>(inner)]
                    * static_cast<long double>(right[
                        static_cast<size_t>(column)
                            * static_cast<size_t>(reduction)
                        + static_cast<size_t>(inner)
                    ]);
                expected += term;
                absolute_bound += std::abs(term);
            }
            for (int row = 0; row < rows; ++row) {
                observed += coefficient(check, row)
                    * static_cast<long double>(output[
                        static_cast<size_t>(column)
                            * static_cast<size_t>(rows)
                        + static_cast<size_t>(row)
                    ]);
            }
            const long double scale = std::max(
                1.0L,
                std::max(
                    absolute_bound,
                    std::max(std::abs(expected), std::abs(observed))
                )
            );
            const long double tolerance = 4096.0L
                * static_cast<long double>(
                    std::numeric_limits<double>::epsilon()
                )
                * static_cast<long double>(rows + reduction + 1) * scale;
            if (!std::isfinite(static_cast<double>(observed)) ||
                std::abs(expected - observed) > tolerance) {
                ++integrity_failures_;
                throw std::runtime_error(
                    "Generalized pass-1 algebraic checksum detected corrupted output"
                );
            }
        }
    }
    ++independent_integrity_audit_count_;
    ++integrity_audit_count_;
}

void GeneralizedGxELDScoreDirectContext::audit_tn_product(
    const double* left,
    int rows,
    int reduction,
    const double* right,
    int columns,
    const double* output,
    int call_index
) {
    const bool protected_checksum = gemm_requires_integrity_checks(
        rows, columns, reduction
    );
    if (protected_checksum) {
        ++protected_integrity_audit_count_;
        ++integrity_audit_count_;
    }
    const bool qualification_fault_pending =
        qualification_fault_phase_ == "pass2_tn" &&
        !qualification_fault_injected_;
    if (protected_checksum && !qualification_fault_pending) return;
    if (call_index % 64 != 0 && !qualification_fault_pending) return;
    if (qualification_fault_pending) {
        if (qualification_fault_row_ >= rows ||
            qualification_fault_column_ >= columns) {
            throw std::runtime_error(
                "Generalized pass-2 qualification fault coordinate is out of range"
            );
        }
        const_cast<double*>(output)[
            static_cast<size_t>(qualification_fault_column_)
                * static_cast<size_t>(rows)
            + static_cast<size_t>(qualification_fault_row_)
        ] += qualification_fault_delta_;
        qualification_fault_injected_ = true;
    }
    const auto coefficient = [](int check, int index) noexcept {
        if (check == 0) return index % 2 == 0 ? 1.0L : -1.0L;
        const int value = ((index * 7 + 1) % 13) - 6;
        return static_cast<long double>(value == 0 ? -3 : value);
    };
    for (int check = 0; check < 2; ++check) {
        std::vector<long double> projected(
            static_cast<size_t>(reduction), 0.0L
        );
        for (int row = 0; row < rows; ++row) {
            const long double weight = coefficient(check, row);
            for (int inner = 0; inner < reduction; ++inner) {
                projected[static_cast<size_t>(inner)] += weight
                    * static_cast<long double>(left[
                        static_cast<size_t>(row)
                            * static_cast<size_t>(reduction)
                        + static_cast<size_t>(inner)
                    ]);
            }
        }
        for (int column = 0; column < columns; ++column) {
            long double expected = 0.0L;
            long double observed = 0.0L;
            long double absolute_bound = 0.0L;
            for (int inner = 0; inner < reduction; ++inner) {
                const long double term =
                    projected[static_cast<size_t>(inner)]
                    * static_cast<long double>(right[
                        static_cast<size_t>(column)
                            * static_cast<size_t>(reduction)
                        + static_cast<size_t>(inner)
                    ]);
                expected += term;
                absolute_bound += std::abs(term);
            }
            for (int row = 0; row < rows; ++row) {
                observed += coefficient(check, row)
                    * static_cast<long double>(output[
                        static_cast<size_t>(column)
                            * static_cast<size_t>(rows)
                        + static_cast<size_t>(row)
                    ]);
            }
            const long double scale = std::max(
                1.0L,
                std::max(
                    absolute_bound,
                    std::max(std::abs(expected), std::abs(observed))
                )
            );
            const long double tolerance = 4096.0L
                * static_cast<long double>(
                    std::numeric_limits<double>::epsilon()
                )
                * static_cast<long double>(rows + reduction + 1) * scale;
            if (!std::isfinite(static_cast<double>(observed)) ||
                std::abs(expected - observed) > tolerance) {
                ++integrity_failures_;
                throw std::runtime_error(
                    "Generalized pass-2 algebraic checksum detected corrupted output"
                );
            }
        }
    }
    ++independent_integrity_audit_count_;
    ++integrity_audit_count_;
}

void GeneralizedGxELDScoreDirectContext::initialize_outputs() {
    const size_t base_elements = checked_mul(
        checked_mul(
            static_cast<size_t>(annotations_count_),
            static_cast<size_t>(rows_), "generalized base sources"
        ),
        static_cast<size_t>(probes_), "generalized base sources"
    );
    const size_t source_elements = checked_mul(
        base_elements, static_cast<size_t>(q_),
        "generalized contextual sources"
    );
    const size_t component_square = checked_mul(
        static_cast<size_t>(components_), static_cast<size_t>(components_),
        "generalized component square"
    );
    const size_t component_sample_elements = checked_mul(
        static_cast<size_t>(components_), static_cast<size_t>(rows_),
        "generalized component kernel diagonal"
    );
    sqrt_annotation_values_.resize(annotation_values_.size());
    std::transform(
        annotation_values_.begin(), annotation_values_.end(),
        sqrt_annotation_values_.begin(),
        [](double value) { return std::sqrt(value); }
    );
    base_sources_.resize(base_elements);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (int64_t index = 0; index < static_cast<int64_t>(base_elements);
         ++index) {
        base_sources_[static_cast<size_t>(index)] = 0.0;
    }
    // Every contextual source cell is assigned at the pass barrier. Avoid a
    // serial zero-fill so the parallel publication loop establishes NUMA
    // placement on first touch.
    contextual_sources_.resize(source_elements);
    weighted_fixed_.assign(checked_mul(
        checked_mul(
            static_cast<size_t>(q_), static_cast<size_t>(fixed_rank_),
            "generalized weighted fixed columns"
        ),
        static_cast<size_t>(rows_),
        "generalized weighted fixed basis"
    ), 0.0);
    for (int coordinate = 0; coordinate < q_; ++coordinate) {
        for (int fixed_column = 0; fixed_column < fixed_rank_;
             ++fixed_column) {
            double* destination = weighted_fixed_.data()
                + (static_cast<size_t>(fixed_column)
                       * static_cast<size_t>(q_)
                   + static_cast<size_t>(coordinate))
                    * static_cast<size_t>(rows_);
            const double* fixed_source = fixed_.data()
                + static_cast<size_t>(fixed_column)
                    * static_cast<size_t>(rows_);
            const double* basis_source = basis_.data()
                + static_cast<size_t>(coordinate)
                    * static_cast<size_t>(rows_);
            for (int row = 0; row < rows_; ++row) {
                destination[row] = basis_source[row] * fixed_source[row];
            }
        }
    }
    component_kernel_diagonal_.assign(component_sample_elements, 0.0);
    same_person_.assign(component_square, 0.0);
    directional_.assign(
        checked_mul(
            checked_mul(
                static_cast<size_t>(variants_), static_cast<size_t>(pairs_),
                "generalized directional panel"
            ),
            static_cast<size_t>(components_),
            "generalized directional panel"
        ),
        0.0
    );
    directed_.assign(component_square, 0.0);
    symmetric_.assign(component_square, 0.0);
    genetic_gram_.assign(component_square, 0.0);
    affine_mean_.assign(
        static_cast<size_t>(variants_),
        std::numeric_limits<double>::quiet_NaN()
    );
    affine_inverse_scale_.assign(
        static_cast<size_t>(variants_),
        std::numeric_limits<double>::quiet_NaN()
    );
}

void GeneralizedGxELDScoreDirectContext::decode_block(
    int start,
    int stop,
    GeneralizedUninitializedDoubleBuffer& genotype
) {
    const auto wall = std::chrono::steady_clock::now();
    const double cpu = process_cpu_seconds_now();
    const int width = stop - start;
    genotype.resize(checked_mul(
        static_cast<size_t>(rows_), static_cast<size_t>(width),
        "generalized decoded block"
    ));
    std::vector<int> observed;
    missing_calls_ += reader_->decode_block_into(
        start, stop, false, genotype.data(), observed,
        affine_mean_.data() + static_cast<size_t>(start),
        affine_inverse_scale_.data() + static_cast<size_t>(start)
    );
    observed_variant_visits_ += width;
    ++observed_block_reads_;
    decode_wall_seconds_ += generalized_elapsed_seconds(wall);
    decode_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - cpu
    );
}

void GeneralizedGxELDScoreDirectContext::read_packed_block(
    int start, int stop, MailmanPackedBlock& packed
) {
    const auto wall = std::chrono::steady_clock::now();
    const double cpu = process_cpu_seconds_now();
    read_block_mailman_mean_memory(
        reader_->bed_base_, reader_->bed_size_, reader_->n_total_,
        reader_->bytes_per_snp_, start, stop, reader_->rows_,
        reader_->ddof_, packed
    );
    if (packed.N != rows_ || packed.L != stop - start) {
        throw std::runtime_error(
            "Generalized packed decoder returned an inconsistent block"
        );
    }
    for (int observed : packed.observed) {
        missing_calls_ += static_cast<int64_t>(rows_ - observed);
    }
    for (int variant = 0; variant < packed.L; ++variant) {
        const size_t local = static_cast<size_t>(variant);
        const size_t global = static_cast<size_t>(start + variant);
        const int observed = packed.observed[local];
        // Mailman retains the compact-code mean. Recover its exact integer
        // sum, then seal the canonical BIM A1 mean just as the dense decoder
        // does. Retained sample counts are far below the exact-integer range
        // of FP64.
        const int64_t compact_sum = observed > 0
            ? static_cast<int64_t>(std::llround(
                packed.mean[local] * static_cast<double>(observed)
            ))
            : 0;
        affine_mean_[global] = observed > 0
            ? static_cast<double>(2LL * observed - compact_sum)
                / static_cast<double>(observed)
            : 0.0;
        affine_inverse_scale_[global] = packed.inv_std[local];
    }
    observed_variant_visits_ += stop - start;
    ++observed_block_reads_;
    decode_wall_seconds_ += generalized_elapsed_seconds(wall);
    decode_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - cpu
    );
}

void GeneralizedGxELDScoreDirectContext::packed_source_product(
    const MailmanPackedBlock& packed,
    const std::vector<double>& rhs,
    int columns,
    std::vector<double>& output
) const {
    const int width = packed.L;
    if (packed.N != rows_ || columns <= 0 ||
        rhs.size() != checked_mul(
            static_cast<size_t>(width), static_cast<size_t>(columns),
            "generalized packed source RHS"
        )) {
        throw std::runtime_error(
            "Generalized packed source product is mis-sized"
        );
    }
    std::vector<double> weights(checked_mul(
        static_cast<size_t>(width), static_cast<size_t>(columns),
        "generalized packed source weights"
    ));
    std::vector<double> mean_correction(
        static_cast<size_t>(columns), 0.0
    );
    for (int variant = 0; variant < width; ++variant) {
        const double inverse = packed.inv_std[static_cast<size_t>(variant)];
        const double mean = packed.mean[static_cast<size_t>(variant)];
        for (int column = 0; column < columns; ++column) {
            const double value = inverse * rhs[
                static_cast<size_t>(column) * static_cast<size_t>(width)
                + static_cast<size_t>(variant)
            ];
            weights[
                static_cast<size_t>(variant) * static_cast<size_t>(columns)
                + static_cast<size_t>(column)
            ] = value;
            mean_correction[static_cast<size_t>(column)] += mean * value;
        }
    }
    output.assign(checked_mul(
        static_cast<size_t>(rows_), static_cast<size_t>(columns),
        "generalized packed source output"
    ), 0.0);
    const int qpanel = summit::mailman::qpanel_width<double>(
        packed.table_size, columns, packed.segment_size
    );
    for (int q0 = 0; q0 < columns; q0 += qpanel) {
        const int qwidth = std::min(qpanel, columns - q0);
        {
            const size_t work_elements = checked_mul(
                static_cast<size_t>(packed.table_size),
                static_cast<size_t>(qwidth),
                "generalized packed source lookup table"
            );
            std::vector<double> work(work_elements, 0.0);
            for (int64_t segment = 0; segment < packed.n_segments; ++segment) {
                const int base = static_cast<int>(
                    segment * static_cast<int64_t>(packed.segment_size)
                );
                const int actual = std::min(
                    packed.segment_size, width - base
                );
                if (packed.use_u16) {
                    summit::mailman::post_multiply_colmajor_subset(
                        packed.packed16.data()
                            + static_cast<size_t>(segment)
                                * static_cast<size_t>(rows_),
                        actual, 0, rows_, qwidth,
                        weights.data()
                            + static_cast<size_t>(base)
                                * static_cast<size_t>(columns)
                            + static_cast<size_t>(q0),
                        columns,
                        output.data()
                            + static_cast<size_t>(q0)
                                * static_cast<size_t>(rows_),
                        rows_, work.data()
                    );
                } else {
                    summit::mailman::post_multiply_colmajor_subset(
                        packed.packed32.data()
                            + static_cast<size_t>(segment)
                                * static_cast<size_t>(rows_),
                        actual, 0, rows_, qwidth,
                        weights.data()
                            + static_cast<size_t>(base)
                                * static_cast<size_t>(columns)
                            + static_cast<size_t>(q0),
                        columns,
                        output.data()
                            + static_cast<size_t>(q0)
                                * static_cast<size_t>(rows_),
                        rows_, work.data()
                    );
                }
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (int column = 0; column < columns; ++column) {
        double* destination = output.data()
            + static_cast<size_t>(column) * static_cast<size_t>(rows_);
        const double correction = mean_correction[static_cast<size_t>(column)];
        for (int row = 0; row < rows_; ++row) {
            destination[row] = correction - destination[row];
        }
        for (int variant = 0; variant < width; ++variant) {
            const double missing_correction = packed.mean[
                static_cast<size_t>(variant)
            ] * weights[
                static_cast<size_t>(variant) * static_cast<size_t>(columns)
                + static_cast<size_t>(column)
            ];
            for (int missing_row : packed.missing_rows[
                    static_cast<size_t>(variant)]) {
                destination[missing_row] -= missing_correction;
            }
        }
    }
}

void GeneralizedGxELDScoreDirectContext::packed_target_product(
    const MailmanPackedBlock& packed,
    const double* rhs,
    int columns,
    std::vector<double>& output
) const {
    const int width = packed.L;
    if (packed.N != rows_ || columns <= 0 || rhs == nullptr) {
        throw std::runtime_error(
            "Generalized packed target product is mis-sized"
        );
    }
    output.assign(checked_mul(
        static_cast<size_t>(width), static_cast<size_t>(columns),
        "generalized packed target output"
    ), 0.0);
    const int qpanel = summit::mailman::qpanel_width<double>(
        packed.table_size, columns, packed.segment_size
    );
    for (int q0 = 0; q0 < columns; q0 += qpanel) {
        const int qwidth = std::min(qpanel, columns - q0);
        std::vector<double> rhs_sums(static_cast<size_t>(qwidth), 0.0);
        for (int local_column = 0; local_column < qwidth; ++local_column) {
            long double total = 0.0L;
            const double* column = rhs
                + static_cast<size_t>(q0 + local_column)
                    * static_cast<size_t>(rows_);
            for (int row = 0; row < rows_; ++row) total += column[row];
            rhs_sums[static_cast<size_t>(local_column)] =
                static_cast<double>(total);
        }
        const auto rhs_value = [&](int row, int local_column) {
            return rhs[
                static_cast<size_t>(q0 + local_column)
                    * static_cast<size_t>(rows_)
                + static_cast<size_t>(row)
            ];
        };
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
        {
            const size_t table_elements = checked_mul(
                static_cast<size_t>(packed.table_size),
                static_cast<size_t>(qwidth),
                "generalized packed target lookup table"
            );
            const size_t segment_elements = checked_mul(
                static_cast<size_t>(packed.segment_size),
                static_cast<size_t>(qwidth),
                "generalized packed target segment"
            );
            std::vector<double> work(table_elements, 0.0);
            std::vector<double> raw(segment_elements, 0.0);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t segment = 0; segment < packed.n_segments; ++segment) {
                const int base = static_cast<int>(
                    segment * static_cast<int64_t>(packed.segment_size)
                );
                const int actual = std::min(
                    packed.segment_size, width - base
                );
                if (packed.use_u16) {
                    summit::mailman::pre_multiply_generated(
                        packed.packed16.data()
                            + static_cast<size_t>(segment)
                                * static_cast<size_t>(rows_),
                        actual, rows_, qwidth, rhs_value,
                        raw.data(), work.data()
                    );
                } else {
                    summit::mailman::pre_multiply_generated(
                        packed.packed32.data()
                            + static_cast<size_t>(segment)
                                * static_cast<size_t>(rows_),
                        actual, rows_, qwidth, rhs_value,
                        raw.data(), work.data()
                    );
                }
                for (int local_variant = 0; local_variant < actual;
                     ++local_variant) {
                    const int variant = base + local_variant;
                    const double mean = packed.mean[
                        static_cast<size_t>(variant)
                    ];
                    const double inverse = packed.inv_std[
                        static_cast<size_t>(variant)
                    ];
                    for (int local_column = 0; local_column < qwidth;
                         ++local_column) {
                        long double missing_rhs = 0.0L;
                        for (int missing_row : packed.missing_rows[
                                static_cast<size_t>(variant)]) {
                            missing_rhs += rhs_value(
                                missing_row, local_column
                            );
                        }
                        output[
                            static_cast<size_t>(q0 + local_column)
                                * static_cast<size_t>(width)
                            + static_cast<size_t>(variant)
                        ] = inverse * (
                            mean * rhs_sums[
                                static_cast<size_t>(local_column)
                            ] - raw[
                                static_cast<size_t>(local_variant)
                                    * static_cast<size_t>(qwidth)
                                + static_cast<size_t>(local_column)
                            ] - mean * static_cast<double>(missing_rhs)
                        );
                    }
                }
            }
        }
    }
}

void GeneralizedGxELDScoreDirectContext::execute_pass1() {
    if (state_ != State::constructed) {
        throw std::runtime_error("Invalid generalized pass-1 transition");
    }
    state_ = State::pass1_running;
    ++observed_passes_;
    GeneralizedUninitializedDoubleBuffer genotype;
    MailmanPackedBlock packed;
    std::vector<double> probes;
    std::vector<double> rhs;
    std::vector<double> contribution;
    std::vector<int> singleton_variants;
    GeneralizedUninitializedDoubleBuffer singleton_genotype;
    std::vector<double> singleton_rhs;
    std::unique_ptr<NativeGemmOutputAllocation> dense_contribution;
    genotype.reserve(checked_mul(
        static_cast<size_t>(rows_),
        static_cast<size_t>(std::min(variants_, variant_block_width_)),
        "generalized pass-1 genotype arena"
    ));
    probes.reserve(checked_mul(
        static_cast<size_t>(std::min(variants_, variant_block_width_)),
        static_cast<size_t>(source_probe_tile_width_),
        "generalized pass-1 probe arena"
    ));
    rhs.reserve(checked_mul(
        probes.capacity(),
        static_cast<size_t>(source_annotation_batch_width_),
        "generalized pass-1 batched RHS arena"
    ));
    contribution.reserve(checked_mul(
        checked_mul(
            static_cast<size_t>(rows_),
            static_cast<size_t>(source_probe_tile_width_),
            "generalized pass-1 output arena"
        ),
        static_cast<size_t>(source_annotation_batch_width_),
        "generalized pass-1 output arena"
    ));
    const int64_t source_tiles =
        (static_cast<int64_t>(probes_) + source_probe_tile_width_ - 1) /
        source_probe_tile_width_;
    const int64_t units_per_block = source_tiles * annotations_count_;
    for (int start = 0; start < variants_; start += variant_block_width_) {
        const int stop = std::min(variants_, start + variant_block_width_);
        const int width = stop - start;
        const int64_t block = start / variant_block_width_;
        check_cancelled();
        set_progress(
            ProgressPhase::pass1, ProgressSubphase::decode,
            block, block, start, stop, 0, units_per_block
        );
        if (dense_blas_hybrid_) {
            decode_block(start, stop, genotype);
        } else {
            read_packed_block(start, stop, packed);
        }
        set_progress(
            ProgressPhase::pass1, ProgressSubphase::source_sketch,
            block, block, start, stop, 0, units_per_block
        );
        int64_t completed_units = 0;
        for (int probe_start = 0; probe_start < probes_;
             probe_start += source_probe_tile_width_) {
            const int probe_stop = std::min(
                probes_, probe_start + source_probe_tile_width_
            );
            const int probe_width = probe_stop - probe_start;
            const auto probe_wall = std::chrono::steady_clock::now();
            const double probe_cpu = process_cpu_seconds_now();
            probes.resize(checked_mul(
                static_cast<size_t>(width),
                static_cast<size_t>(probe_width),
                "generalized source probes"
            ));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
            for (int probe = 0; probe < probe_width; ++probe) {
                for (int variant = 0; variant < width; ++variant) {
                    probes[
                        static_cast<size_t>(probe)
                            * static_cast<size_t>(width)
                        + static_cast<size_t>(variant)
                    ] = generalized_global_probe_sign(
                        root_seed_, static_cast<uint64_t>(start + variant),
                        static_cast<uint64_t>(
                            probe_offset_ + probe_start + probe
                        ),
                        namespace_key_
                    );
                }
            }
            rhs_wall_seconds_ += generalized_elapsed_seconds(probe_wall);
            rhs_cpu_seconds_ += std::max(
                0.0, process_cpu_seconds_now() - probe_cpu
            );
            if (dense_blas_hybrid_ && singleton_annotation_rows_) {
                for (int annotation = 0;
                     annotation < annotations_count_; ++annotation) {
                    singleton_variants.clear();
                    for (int variant = 0; variant < width; ++variant) {
                        if (singleton_annotation_index_[
                                static_cast<size_t>(start + variant)
                            ] == annotation) {
                            singleton_variants.push_back(variant);
                        }
                    }
                    const int singleton_width = checked_blas_dim(
                        singleton_variants.size(),
                        "generalized singleton annotation width"
                    );
                    if (singleton_width == 0) {
                        ++completed_units;
                        set_progress_units(
                            completed_units, units_per_block
                        );
                        check_cancelled();
                        continue;
                    }
                    const bool reuse_decoded_genotype =
                        singleton_width == width;
                    if (!reuse_decoded_genotype) {
                        singleton_genotype.resize(checked_mul(
                            static_cast<size_t>(rows_),
                            static_cast<size_t>(singleton_width),
                            "generalized singleton annotation genotype"
                        ));
                    }
                    const double* singleton_genotype_data =
                        reuse_decoded_genotype
                            ? genotype.data()
                            : singleton_genotype.data();
                    singleton_rhs.resize(checked_mul(
                        static_cast<size_t>(singleton_width),
                        static_cast<size_t>(probe_width),
                        "generalized singleton annotation RHS"
                    ));
                    const auto rhs_wall = std::chrono::steady_clock::now();
                    const double rhs_cpu = process_cpu_seconds_now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                    for (int position = 0; position < singleton_width;
                         ++position) {
                        const int variant = singleton_variants[
                            static_cast<size_t>(position)
                        ];
                        if (!reuse_decoded_genotype) {
                            std::memcpy(
                                singleton_genotype.data()
                                    + static_cast<size_t>(position)
                                        * static_cast<size_t>(rows_),
                                genotype.data()
                                    + static_cast<size_t>(variant)
                                        * static_cast<size_t>(rows_),
                                static_cast<size_t>(rows_) * sizeof(double)
                            );
                        }
                        const double weight = sqrt_annotation_values_[
                            static_cast<size_t>(start + variant)
                                * static_cast<size_t>(annotations_count_)
                            + static_cast<size_t>(annotation)
                        ];
                        for (int probe = 0; probe < probe_width; ++probe) {
                            singleton_rhs[
                                static_cast<size_t>(probe)
                                    * static_cast<size_t>(singleton_width)
                                + static_cast<size_t>(position)
                            ] = weight * probes[
                                static_cast<size_t>(probe)
                                    * static_cast<size_t>(width)
                                + static_cast<size_t>(variant)
                            ];
                        }
                    }
                    rhs_wall_seconds_ +=
                        generalized_elapsed_seconds(rhs_wall);
                    rhs_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - rhs_cpu
                    );

                    NativeGemmOutputTelemetryScope output_scope;
                    const size_t logical_elements = checked_mul(
                        static_cast<size_t>(rows_),
                        static_cast<size_t>(probe_width),
                        "generalized singleton source output"
                    );
                    const size_t logical_bytes = checked_mul(
                        logical_elements, sizeof(double),
                        "generalized singleton source output bytes"
                    );
                    const size_t capacity_bytes = checked_mul(
                        checked_mul(
                            static_cast<size_t>(rows_),
                            static_cast<size_t>(source_probe_tile_width_),
                            "generalized singleton source capacity"
                        ),
                        sizeof(double),
                        "generalized singleton source capacity bytes"
                    );
                    if (dense_contribution == nullptr) {
                        dense_contribution =
                            std::make_unique<NativeGemmOutputAllocation>(
                                static_cast<size_t>(rows_),
                                static_cast<size_t>(probe_width),
                                "column_major", logical_bytes,
                                output_scope.request(),
                                output_scope.evidence(), capacity_bytes
                            );
                    } else {
                        dense_contribution->reuse_for_call(
                            static_cast<size_t>(rows_),
                            static_cast<size_t>(probe_width),
                            "column_major", logical_bytes,
                            output_scope.request(), output_scope.evidence()
                        );
                    }
                    double* contribution_data = dense_contribution->data();
                    output_scope.register_output(
                        contribution_data, logical_bytes, CblasColMajor
                    );
                    GemmIntegrityResolution resolution;
                    const auto product_wall =
                        std::chrono::steady_clock::now();
                    const double product_cpu = process_cpu_seconds_now();
                    try {
                        (void)dgemm_nn_partitioned_rows(
                            rows_, probe_width, singleton_width,
                            singleton_genotype_data, rows_,
                            singleton_rhs.data(), singleton_width,
                            contribution_data, rows_, threads_, 1.0, 0.0,
                            &resolution
                        );
                    } catch (const RetryableGemmInputMutation&) {
                        ++retry_count_;
                        if (reuse_decoded_genotype) {
                            decode_block(start, stop, genotype);
                            singleton_genotype_data = genotype.data();
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                        for (int position = 0;
                             position < singleton_width; ++position) {
                            if (reuse_decoded_genotype) continue;
                            const int variant = singleton_variants[
                                static_cast<size_t>(position)
                            ];
                            std::memcpy(
                                singleton_genotype.data()
                                    + static_cast<size_t>(position)
                                        * static_cast<size_t>(rows_),
                                genotype.data()
                                    + static_cast<size_t>(variant)
                                        * static_cast<size_t>(rows_),
                                static_cast<size_t>(rows_) * sizeof(double)
                            );
                        }
                        dgemm_nn_tiled(
                            rows_, probe_width, singleton_width,
                            singleton_genotype_data, rows_,
                            singleton_rhs.data(), singleton_width,
                            contribution_data, rows_, threads_
                        );
                        ++fallback_count_;
                    }
                    const double product_wall_seconds =
                        generalized_elapsed_seconds(product_wall);
                    const double product_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - product_cpu
                    );
                    source_nn_wall_seconds_ += product_wall_seconds;
                    source_nn_cpu_seconds_ += product_cpu_seconds;
                    record_gemm_shape(
                        "pass1_source_nn_singleton", false,
                        rows_, probe_width, singleton_width,
                        product_wall_seconds, product_cpu_seconds
                    );
                    record_integrity_resolution(resolution);
                    dense_contribution->verify_after_repair();
                    const auto audit_wall =
                        std::chrono::steady_clock::now();
                    const double audit_cpu = process_cpu_seconds_now();
                    audit_nn_product(
                        singleton_genotype_data, rows_, singleton_width,
                        singleton_rhs.data(), probe_width,
                        contribution_data,
                        static_cast<int>(source_nn_calls_)
                    );
                    const double audit_wall_seconds =
                        generalized_elapsed_seconds(audit_wall);
                    const double audit_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - audit_cpu
                    );
                    independent_audit_wall_seconds_ += audit_wall_seconds;
                    independent_audit_cpu_seconds_ += audit_cpu_seconds;
                    source_independent_audit_wall_seconds_ +=
                        audit_wall_seconds;
                    source_independent_audit_cpu_seconds_ +=
                        audit_cpu_seconds;
                    ++gemm_output_numa_call_count_;
                    if (gemm_output_numa_evidence_.size() < 64U) {
                        gemm_output_numa_evidence_.push_back(
                            output_scope.evidence()
                        );
                    }
                    output_scope.complete();
                    ++source_nn_calls_;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                    for (int probe = 0; probe < probe_width; ++probe) {
                        for (int row = 0; row < rows_; ++row) {
                            base_sources_[base_index(
                                annotation, row, probe_start + probe
                            )] += contribution_data[
                                static_cast<size_t>(probe)
                                    * static_cast<size_t>(rows_)
                                + static_cast<size_t>(row)
                            ];
                        }
                    }
                    ++completed_units;
                    set_progress_units(completed_units, units_per_block);
                    check_cancelled();
                }
                continue;
            }
            for (int annotation_start = 0;
                 annotation_start < annotations_count_;
                 annotation_start += source_annotation_batch_width_) {
                const int annotation_stop = std::min(
                    annotations_count_,
                    annotation_start + source_annotation_batch_width_
                );
                const int annotation_width =
                    annotation_stop - annotation_start;
                const int rhs_columns = annotation_width * probe_width;
                rhs.resize(checked_mul(
                    static_cast<size_t>(width),
                    static_cast<size_t>(rhs_columns),
                    "generalized batched source RHS"
                ));
                if (!dense_blas_hybrid_) {
                    contribution.resize(checked_mul(
                        static_cast<size_t>(rows_),
                        static_cast<size_t>(rhs_columns),
                        "generalized batched source contribution"
                    ));
                }
                const auto rhs_wall = std::chrono::steady_clock::now();
                const double rhs_cpu = process_cpu_seconds_now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int column = 0; column < rhs_columns; ++column) {
                    const int local_annotation = column / probe_width;
                    const int probe = column % probe_width;
                    const int annotation =
                        annotation_start + local_annotation;
                    for (int variant = 0; variant < width; ++variant) {
                        rhs[
                            static_cast<size_t>(column)
                                * static_cast<size_t>(width)
                            + static_cast<size_t>(variant)
                        ] = sqrt_annotation_values_[
                            static_cast<size_t>(start + variant)
                                * static_cast<size_t>(annotations_count_)
                            + static_cast<size_t>(annotation)
                        ] * probes[
                            static_cast<size_t>(probe)
                                * static_cast<size_t>(width)
                            + static_cast<size_t>(variant)
                        ];
                    }
                }
                rhs_wall_seconds_ += generalized_elapsed_seconds(rhs_wall);
                rhs_cpu_seconds_ += std::max(
                    0.0, process_cpu_seconds_now() - rhs_cpu
                );
                double* contribution_data = contribution.data();
                if (dense_blas_hybrid_) {
                    NativeGemmOutputTelemetryScope output_scope;
                    const size_t logical_elements = checked_mul(
                        static_cast<size_t>(rows_),
                        static_cast<size_t>(rhs_columns),
                        "generalized source logical output"
                    );
                    const size_t logical_bytes = checked_mul(
                        logical_elements, sizeof(double),
                        "generalized source logical output bytes"
                    );
                    const size_t capacity_elements = checked_mul(
                        checked_mul(
                            static_cast<size_t>(rows_),
                            static_cast<size_t>(source_probe_tile_width_),
                            "generalized source output capacity"
                        ),
                        static_cast<size_t>(
                            source_annotation_batch_width_
                        ),
                        "generalized source output capacity"
                    );
                    const size_t capacity_bytes = checked_mul(
                        capacity_elements,
                        sizeof(double),
                        "generalized source output capacity bytes"
                    );
                    if (dense_contribution == nullptr) {
                        dense_contribution =
                            std::make_unique<NativeGemmOutputAllocation>(
                                static_cast<size_t>(rows_),
                                static_cast<size_t>(rhs_columns),
                                "column_major", logical_bytes,
                                output_scope.request(),
                                output_scope.evidence(), capacity_bytes
                            );
                    } else {
                        dense_contribution->reuse_for_call(
                            static_cast<size_t>(rows_),
                            static_cast<size_t>(rhs_columns),
                            "column_major", logical_bytes,
                            output_scope.request(), output_scope.evidence()
                        );
                    }
                    contribution_data = dense_contribution->data();
                    output_scope.register_output(
                        contribution_data, logical_bytes, CblasColMajor
                    );
                    GemmIntegrityResolution resolution;
                    const auto product_wall = std::chrono::steady_clock::now();
                    const double product_cpu = process_cpu_seconds_now();
                    try {
                        (void)dgemm_nn_partitioned_rows(
                            rows_, rhs_columns, width,
                            genotype.data(), rows_, rhs.data(), width,
                            contribution_data, rows_, threads_, 1.0, 0.0,
                            &resolution
                        );
                    } catch (const RetryableGemmInputMutation&) {
                        ++retry_count_;
                        dgemm_nn_tiled(
                            rows_, rhs_columns, width,
                            genotype.data(), rows_, rhs.data(), width,
                            contribution_data, rows_, threads_
                        );
                        ++fallback_count_;
                    }
                    const double product_wall_seconds =
                        generalized_elapsed_seconds(product_wall);
                    const double product_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - product_cpu
                    );
                    source_nn_wall_seconds_ += product_wall_seconds;
                    source_nn_cpu_seconds_ += product_cpu_seconds;
                    record_gemm_shape(
                        "pass1_source_nn", false,
                        rows_, rhs_columns, width,
                        product_wall_seconds, product_cpu_seconds
                    );
                    record_integrity_resolution(resolution);
                    dense_contribution->verify_after_repair();
                    const auto audit_wall = std::chrono::steady_clock::now();
                    const double audit_cpu = process_cpu_seconds_now();
                    audit_nn_product(
                        genotype.data(), rows_, width, rhs.data(),
                        rhs_columns, contribution_data,
                        static_cast<int>(source_nn_calls_)
                    );
                    const double audit_wall_seconds =
                        generalized_elapsed_seconds(audit_wall);
                    const double audit_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - audit_cpu
                    );
                    independent_audit_wall_seconds_ += audit_wall_seconds;
                    independent_audit_cpu_seconds_ += audit_cpu_seconds;
                    source_independent_audit_wall_seconds_ +=
                        audit_wall_seconds;
                    source_independent_audit_cpu_seconds_ +=
                        audit_cpu_seconds;
                    ++gemm_output_numa_call_count_;
                    if (gemm_output_numa_evidence_.size() < 64U) {
                        gemm_output_numa_evidence_.push_back(
                            output_scope.evidence()
                        );
                    }
                    output_scope.complete();
                } else {
                    const auto product_wall = std::chrono::steady_clock::now();
                    const double product_cpu = process_cpu_seconds_now();
                    packed_source_product(
                        packed, rhs, rhs_columns, contribution
                    );
                    contribution_data = contribution.data();
                    const double product_wall_seconds =
                        generalized_elapsed_seconds(product_wall);
                    const double product_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - product_cpu
                    );
                    source_nn_wall_seconds_ += product_wall_seconds;
                    source_nn_cpu_seconds_ += product_cpu_seconds;
                    record_gemm_shape(
                        "pass1_source_packed", false,
                        rows_, rhs_columns, width,
                        product_wall_seconds, product_cpu_seconds
                    );
                }
                ++source_nn_calls_;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int column = 0; column < rhs_columns; ++column) {
                    const int local_annotation = column / probe_width;
                    const int probe = column % probe_width;
                    const int annotation =
                        annotation_start + local_annotation;
                    for (int row = 0; row < rows_; ++row) {
                        base_sources_[base_index(
                            annotation, row, probe_start + probe
                        )] += contribution_data[
                            static_cast<size_t>(column)
                                * static_cast<size_t>(rows_)
                            + static_cast<size_t>(row)
                        ];
                    }
                }
                completed_units += annotation_width;
                set_progress_units(completed_units, units_per_block);
                check_cancelled();
            }
        }
        set_progress(
            ProgressPhase::pass1, ProgressSubphase::source_sketch,
            block + 1, block, start, stop,
            units_per_block, units_per_block
        );
    }
    reader_->check_files_unchanged();
    if (observed_variant_visits_ != variants_) {
        throw std::runtime_error(
            "Generalized pass 1 did not visit every variant exactly once"
        );
    }
}

void GeneralizedGxELDScoreDirectContext::seal_sources() {
    if (state_ != State::pass1_running || observed_passes_ != 1 ||
        observed_variant_visits_ != variants_) {
        throw std::runtime_error(
            "Generalized pass-1 barrier is not ready to seal"
        );
    }
    GeneralizedUninitializedDoubleBuffer weighted;
    weighted.resize(checked_mul(
        static_cast<size_t>(rows_), static_cast<size_t>(probes_),
        "generalized contextual weighted source"
    ));
    std::vector<double> coefficients(checked_mul(
        static_cast<size_t>(fixed_rank_), static_cast<size_t>(probes_),
        "generalized projection coefficients"
    ));
    std::vector<double> leakage(coefficients.size());
    const int64_t total_blocks = progress_total_blocks_.load(
        std::memory_order_relaxed
    );
    set_progress(
        ProgressPhase::barrier, ProgressSubphase::source_projection,
        total_blocks, -1, variants_, variants_, 0,
        static_cast<int64_t>(annotations_count_) * q_
    );
    const auto projection_wall = std::chrono::steady_clock::now();
    const double projection_cpu = process_cpu_seconds_now();
    for (int annotation = 0; annotation < annotations_count_; ++annotation) {
        for (int coordinate = 0; coordinate < q_; ++coordinate) {
            check_cancelled();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
            for (int probe = 0; probe < probes_; ++probe) {
                for (int row = 0; row < rows_; ++row) {
                    weighted[
                        static_cast<size_t>(probe)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ] = basis_[
                        static_cast<size_t>(coordinate)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ] * base_sources_[base_index(annotation, row, probe)];
                }
            }
            if (fixed_rank_ > 0) {
                GemmIntegrityResolution tn_resolution;
                const auto tn_wall = std::chrono::steady_clock::now();
                const double tn_cpu = process_cpu_seconds_now();
                (void)dgemm_tn_partitioned_columns(
                    fixed_rank_, probes_, rows_, fixed_.data(), rows_,
                    weighted.data(), rows_, coefficients.data(), fixed_rank_,
                    threads_, &tn_resolution
                );
                record_gemm_shape(
                    "barrier_projection_tn", true,
                    fixed_rank_, probes_, rows_,
                    generalized_elapsed_seconds(tn_wall),
                    std::max(0.0, process_cpu_seconds_now() - tn_cpu)
                );
                record_integrity_resolution(tn_resolution);
                GemmIntegrityResolution nn_resolution;
                const auto nn_wall = std::chrono::steady_clock::now();
                const double nn_cpu = process_cpu_seconds_now();
                (void)dgemm_nn_partitioned_rows(
                    rows_, probes_, fixed_rank_, fixed_.data(), rows_,
                    coefficients.data(), fixed_rank_, weighted.data(), rows_,
                    threads_, -1.0, 1.0, &nn_resolution
                );
                record_gemm_shape(
                    "barrier_projection_nn", false,
                    rows_, probes_, fixed_rank_,
                    generalized_elapsed_seconds(nn_wall),
                    std::max(0.0, process_cpu_seconds_now() - nn_cpu)
                );
                record_integrity_resolution(nn_resolution);
                GemmIntegrityResolution leakage_resolution;
                const auto leakage_wall = std::chrono::steady_clock::now();
                const double leakage_cpu = process_cpu_seconds_now();
                (void)dgemm_tn_partitioned_columns(
                    fixed_rank_, probes_, rows_, fixed_.data(), rows_,
                    weighted.data(), rows_, leakage.data(), fixed_rank_,
                    threads_, &leakage_resolution
                );
                record_gemm_shape(
                    "barrier_projection_audit_tn", true,
                    fixed_rank_, probes_, rows_,
                    generalized_elapsed_seconds(leakage_wall),
                    std::max(0.0, process_cpu_seconds_now() - leakage_cpu)
                );
                record_integrity_resolution(leakage_resolution);
            }
            std::vector<long double> projected_squared_by_probe(
                static_cast<size_t>(probes_), 0.0L
            );
            int invalid_projected_source = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) reduction(|:invalid_projected_source)
#endif
            for (int probe = 0; probe < probes_; ++probe) {
                long double probe_squared = 0.0L;
                for (int row = 0; row < rows_; ++row) {
                    const double value = weighted[
                        static_cast<size_t>(probe)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ];
                    invalid_projected_source |= !std::isfinite(value);
                    probe_squared += static_cast<long double>(value) * value;
                    contextual_sources_[source_index(
                        annotation, coordinate, row, probe
                    )] = value;
                }
                projected_squared_by_probe[static_cast<size_t>(probe)] =
                    probe_squared;
            }
            if (invalid_projected_source != 0) {
                throw std::runtime_error(
                    "Generalized contextual source is non-finite"
                );
            }
            long double projected_squared = 0.0L;
            for (long double value : projected_squared_by_probe) {
                projected_squared += value;
            }
            long double leakage_squared = 0.0L;
            for (double value : leakage) {
                maximum_projection_leakage_ = std::max(
                    maximum_projection_leakage_, std::abs(value)
                );
                leakage_squared += static_cast<long double>(value) * value;
            }
            maximum_relative_projection_leakage_ = std::max(
                maximum_relative_projection_leakage_,
                std::sqrt(static_cast<double>(leakage_squared)) /
                    std::max(
                        1.0,
                        std::sqrt(static_cast<double>(projected_squared))
                    )
            );
            set_progress_units(
                static_cast<int64_t>(annotation) * q_ + coordinate + 1,
                static_cast<int64_t>(annotations_count_) * q_
            );
        }
    }
    if (maximum_relative_projection_leakage_ > 2.0e-11) {
        throw std::runtime_error(
            "Generalized contextual projection leakage exceeds tolerance"
        );
    }
    projection_wall_seconds_ += generalized_elapsed_seconds(projection_wall);
    projection_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - projection_cpu
    );

    // Same-person moments are formed exactly from the component-kernel
    // diagonals accumulated during pass 2.
    for (int variant = 0; variant < variants_; ++variant) {
        const double mean = affine_mean_[static_cast<size_t>(variant)];
        const double inverse =
            affine_inverse_scale_[static_cast<size_t>(variant)];
        if (!std::isfinite(mean) || !std::isfinite(inverse) ||
            inverse <= 0.0) {
            throw std::runtime_error(
                "Generalized genotype scale identity is non-finite"
            );
        }
    }
    if (!retain_base_sources_) {
        base_sources_.clear();
    }
    state_ = State::sources_sealed;
    build_precomputed_rhs();
    std::vector<double>().swap(sqrt_annotation_values_);
    if (precompute_rhs_ && !retain_contextual_sources_) {
        contextual_sources_.clear();
    }
}

void GeneralizedGxELDScoreDirectContext::build_precomputed_rhs() {
    if (!precompute_rhs_) return;
    if (state_ != State::sources_sealed || precomputed_rhs_ != nullptr) {
        throw std::runtime_error(
            "Generalized RHS precomputation has an invalid lifecycle"
        );
    }
    const int family_count = q_ * q_;
    rhs_precomputed_columns_ = checked_blas_dim(
        checked_mul(
            checked_mul(
                static_cast<size_t>(annotations_count_),
                static_cast<size_t>(probes_),
                "generalized precomputed RHS columns"
            ),
            static_cast<size_t>(family_count),
            "generalized precomputed RHS columns"
        ),
        "generalized precomputed RHS columns"
    );
    const size_t elements = checked_mul(
        static_cast<size_t>(rows_),
        static_cast<size_t>(rhs_precomputed_columns_),
        "generalized precomputed RHS"
    );
    rhs_precomputed_bytes_ = checked_mul(
        static_cast<uint64_t>(elements),
        static_cast<uint64_t>(sizeof(double)),
        "generalized precomputed RHS bytes"
    );
    if (rhs_precomputed_bytes_ > max_workspace_bytes_) {
        throw std::runtime_error(
            "Generalized precomputed RHS exceeds the native workspace limit"
        );
    }
    const int target_tiles =
        (probes_ + probe_tile_width_ - 1) / probe_tile_width_;
    set_progress(
        ProgressPhase::barrier, ProgressSubphase::rhs_precompute,
        progress_total_blocks_.load(std::memory_order_relaxed),
        -1, variants_, variants_, 0, target_tiles
    );
    const auto wall = std::chrono::steady_clock::now();
    const double cpu = process_cpu_seconds_now();
    auto mapping = std::make_unique<ReadOnlyDoubleMapping>(elements);
    for (int probe_start = 0; probe_start < probes_;
         probe_start += probe_tile_width_) {
        check_cancelled();
        const int probe_stop = std::min(
            probes_, probe_start + probe_tile_width_
        );
        const int probe_width = probe_stop - probe_start;
        const int columns_in_tile = checked_blas_dim(
            checked_mul(
                checked_mul(
                    static_cast<size_t>(annotations_count_),
                    static_cast<size_t>(probe_width),
                    "generalized precomputed RHS tile columns"
                ),
                static_cast<size_t>(family_count),
                "generalized precomputed RHS tile columns"
            ),
            "generalized precomputed RHS tile columns"
        );
        int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) reduction(|:invalid)
#endif
        for (int local_column = 0;
             local_column < columns_in_tile; ++local_column) {
            const int annotation_stride = probe_width * family_count;
            const int annotation = local_column / annotation_stride;
            const int within_annotation =
                local_column % annotation_stride;
            const int probe =
                probe_start + within_annotation / family_count;
            const int family = within_annotation % family_count;
            const int target_coordinate = family / q_;
            const int source_coordinate = family % q_;
            const size_t column = checked_mul(
                static_cast<size_t>(probe_start),
                checked_mul(
                    static_cast<size_t>(annotations_count_),
                    static_cast<size_t>(family_count),
                    "generalized precomputed RHS tile offset"
                ),
                "generalized precomputed RHS tile offset"
            ) + static_cast<size_t>(local_column);
            double* destination = mapping->data()
                + column * static_cast<size_t>(rows_);
            const double* phi = basis_.data()
                + static_cast<size_t>(target_coordinate)
                    * static_cast<size_t>(rows_);
            const double* source = contextual_sources_.data()
                + source_index(
                    annotation, source_coordinate, 0, probe
                );
            for (int row = 0; row < rows_; ++row) {
                destination[row] = phi[row] * source[row];
                invalid |= !std::isfinite(destination[row]);
            }
        }
        if (invalid != 0) {
            for (int local_column = 0;
                 local_column < columns_in_tile; ++local_column) {
                const int annotation_stride = probe_width * family_count;
                const int annotation = local_column / annotation_stride;
                const int within_annotation =
                    local_column % annotation_stride;
                const int probe =
                    probe_start + within_annotation / family_count;
                const int family = within_annotation % family_count;
                const int target_coordinate = family / q_;
                const int source_coordinate = family % q_;
                const size_t column = checked_mul(
                    static_cast<size_t>(probe_start),
                    checked_mul(
                        static_cast<size_t>(annotations_count_),
                        static_cast<size_t>(family_count),
                        "generalized precomputed RHS diagnostic offset"
                    ),
                    "generalized precomputed RHS diagnostic offset"
                ) + static_cast<size_t>(local_column);
                const double* destination = mapping->data()
                    + column * static_cast<size_t>(rows_);
                const double* phi = basis_.data()
                    + static_cast<size_t>(target_coordinate)
                        * static_cast<size_t>(rows_);
                const double* source = contextual_sources_.data()
                    + source_index(
                        annotation, source_coordinate, 0, probe
                    );
                for (int row = 0; row < rows_; ++row) {
                    if (std::isfinite(destination[row])) continue;
                    std::ostringstream message;
                    message
                        << "Generalized precomputed RHS is non-finite at "
                        << "annotation=" << annotation
                        << ", probe=" << probe
                        << ", target_coordinate=" << target_coordinate
                        << ", source_coordinate=" << source_coordinate
                        << ", row=" << row
                        << "; basis_finite=" << std::isfinite(phi[row])
                        << ", source_finite=" << std::isfinite(source[row]);
                    throw std::runtime_error(message.str());
                }
            }
            throw std::runtime_error(
                "Generalized precomputed RHS finite reduction was inconsistent"
            );
        }
        set_progress_units(
            probe_start / probe_tile_width_ + 1, target_tiles
        );
    }
    mapping->seal_read_only();
    precomputed_rhs_ = std::move(mapping);
    rhs_precompute_wall_seconds_ += generalized_elapsed_seconds(wall);
    rhs_precompute_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - cpu
    );
}

void GeneralizedGxELDScoreDirectContext::accumulate_component_kernel_diagonal(
    int start,
    int stop,
    const GeneralizedUninitializedDoubleBuffer* genotype,
    const MailmanPackedBlock* packed,
    ComponentDiagonalWorkspace& workspace
) {
    const auto wall = std::chrono::steady_clock::now();
    const double cpu = process_cpu_seconds_now();
    const int width = stop - start;
    if (width <= 0 || (genotype == nullptr) == (packed == nullptr)) {
        throw std::runtime_error(
            "Generalized component-diagonal block is invalid"
        );
    }
    if (genotype != nullptr && genotype->size() != checked_mul(
            static_cast<size_t>(rows_), static_cast<size_t>(width),
            "generalized component-diagonal dense block")) {
        throw std::runtime_error(
            "Generalized component-diagonal dense block is mis-sized"
        );
    }
    if (packed != nullptr && (packed->N != rows_ || packed->L != width)) {
        throw std::runtime_error(
            "Generalized component-diagonal packed block is mis-sized"
        );
    }

    const int projection_columns = q_ * fixed_rank_;
    std::vector<double>& coefficients = workspace.coefficients;
    if (projection_columns > 0) {
        coefficients.resize(checked_mul(
            static_cast<size_t>(width),
            static_cast<size_t>(projection_columns),
            "generalized component-diagonal projection coefficients"
        ));
        const auto product_wall = std::chrono::steady_clock::now();
        const double product_cpu = process_cpu_seconds_now();
        if (genotype != nullptr) {
            GemmIntegrityResolution resolution;
            (void)dgemm_tn_partitioned_rows(
                width, projection_columns, rows_, genotype->data(), rows_,
                weighted_fixed_.data(), rows_, coefficients.data(), width,
                threads_, 1.0, 0.0, nullptr, false, &resolution
            );
            record_integrity_resolution(resolution);
        } else {
            packed_target_product(
                *packed, weighted_fixed_.data(), projection_columns,
                coefficients
            );
        }
        record_gemm_shape(
            genotype != nullptr
                ? "pass2_component_diagonal_projection_tn"
                : "pass2_component_diagonal_projection_packed",
            true, width, projection_columns, rows_,
            generalized_elapsed_seconds(product_wall),
            std::max(0.0, process_cpu_seconds_now() - product_cpu)
        );
    }

    const int maximum_sample_width = std::min(
        rows_, same_person_sample_tile_width_
    );
    std::vector<double>& fixed_tile = workspace.fixed_tile;
    fixed_tile.resize(checked_mul(
        static_cast<size_t>(maximum_sample_width),
        static_cast<size_t>(fixed_rank_),
        "generalized component-diagonal fixed tile"
    ));
    GeneralizedUninitializedDoubleBuffer& genotype_tile =
        workspace.genotype_tile;
    genotype_tile.resize(checked_mul(
        static_cast<size_t>(maximum_sample_width),
        static_cast<size_t>(width),
        "generalized component-diagonal genotype tile"
    ));
    GeneralizedUninitializedDoubleBuffer& feature_tile =
        workspace.feature_tile;
    feature_tile.resize(checked_mul(
        checked_mul(
            static_cast<size_t>(q_),
            static_cast<size_t>(maximum_sample_width),
            "generalized component-diagonal feature tile"
        ),
        static_cast<size_t>(width),
        "generalized component-diagonal feature tile"
    ));
    constexpr size_t kMaximumPairBatchBytes = 256U * 1024U * 1024U;
    constexpr int kMaximumPairBatchWidth = 8;
    const size_t one_pair_tile_elements = checked_mul(
        static_cast<size_t>(maximum_sample_width),
        static_cast<size_t>(width),
        "generalized component-diagonal pair tile"
    );
    const size_t one_pair_tile_bytes = checked_mul(
        one_pair_tile_elements, sizeof(double),
        "generalized component-diagonal pair tile bytes"
    );
    const int pair_batch_width = singleton_annotation_rows_
        ? pairs_
        : std::max(1, std::min(
              std::min(pairs_, kMaximumPairBatchWidth),
              static_cast<int>(std::max<size_t>(
                  1U, kMaximumPairBatchBytes / one_pair_tile_bytes
              ))
          ));
    component_pair_batch_width_ = std::max(
        component_pair_batch_width_, pair_batch_width
    );
    GeneralizedUninitializedDoubleBuffer& pair_tile = workspace.pair_tile;
    if (!singleton_annotation_rows_) {
        pair_tile.resize(checked_mul(
            one_pair_tile_elements,
            static_cast<size_t>(pair_batch_width),
            "generalized component-diagonal pair tile"
        ));
    }
    GeneralizedUninitializedDoubleBuffer& annotation_product =
        workspace.annotation_product;
    annotation_product.resize(checked_mul(
        checked_mul(
            static_cast<size_t>(maximum_sample_width),
            static_cast<size_t>(annotations_count_),
            "generalized component-diagonal annotation product"
        ),
        static_cast<size_t>(pair_batch_width),
        "generalized component-diagonal annotation product"
    ));

    for (int sample_start = 0; sample_start < rows_;
         sample_start += same_person_sample_tile_width_) {
        const int sample_stop = std::min(
            rows_, sample_start + same_person_sample_tile_width_
        );
        const int sample_width = sample_stop - sample_start;
        if (genotype != nullptr) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
            for (int sample = 0; sample < sample_width; ++sample) {
                const int global_sample = sample_start + sample;
                for (int variant = 0; variant < width; ++variant) {
                    genotype_tile[
                        static_cast<size_t>(sample)
                            * static_cast<size_t>(width)
                        + static_cast<size_t>(variant)
                    ] = (*genotype)[
                        static_cast<size_t>(variant)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(global_sample)
                    ];
                }
            }
        } else {
            for (int64_t segment = 0; segment < packed->n_segments;
                 ++segment) {
                const int base = static_cast<int>(
                    segment * static_cast<int64_t>(packed->segment_size)
                );
                const int actual = std::min(
                    packed->segment_size, width - base
                );
                int64_t divisor = compute_mailman_table_size(actual);
                for (int local_variant = 0; local_variant < actual;
                     ++local_variant) {
                    divisor /= 3;
                    const int variant = base + local_variant;
                    const double mean = packed->mean[
                        static_cast<size_t>(variant)
                    ];
                    const double inverse = packed->inv_std[
                        static_cast<size_t>(variant)
                    ];
                    for (int sample = 0; sample < sample_width; ++sample) {
                        const int global_sample = sample_start + sample;
                        int code = 0;
                        if (packed->use_u16) {
                            code = static_cast<int>((packed->packed16[
                                static_cast<size_t>(segment)
                                    * static_cast<size_t>(rows_)
                                + static_cast<size_t>(global_sample)
                            ] / divisor) % 3);
                        } else {
                            code = static_cast<int>((packed->packed32[
                                static_cast<size_t>(segment)
                                    * static_cast<size_t>(rows_)
                                + static_cast<size_t>(global_sample)
                            ] / divisor) % 3);
                        }
                        genotype_tile[
                            static_cast<size_t>(sample)
                                * static_cast<size_t>(width)
                            + static_cast<size_t>(variant)
                        ] = (mean - static_cast<double>(code)) * inverse;
                    }
                    for (int missing_row : packed->missing_rows[
                            static_cast<size_t>(variant)]) {
                        if (missing_row >= sample_start &&
                            missing_row < sample_stop) {
                            genotype_tile[
                                static_cast<size_t>(missing_row - sample_start)
                                    * static_cast<size_t>(width)
                                + static_cast<size_t>(variant)
                            ] = 0.0;
                        }
                    }
                }
            }
        }

        const int feature_rows = q_ * width;
        if (fixed_rank_ > 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
            for (int sample = 0; sample < sample_width; ++sample) {
                const int global_sample = sample_start + sample;
                for (int fixed_column = 0; fixed_column < fixed_rank_;
                     ++fixed_column) {
                    fixed_tile[
                        static_cast<size_t>(sample)
                            * static_cast<size_t>(fixed_rank_)
                        + static_cast<size_t>(fixed_column)
                    ] = fixed_[
                        static_cast<size_t>(fixed_column)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(global_sample)
                    ];
                }
            }
            GemmIntegrityResolution feature_resolution;
            const auto feature_wall = std::chrono::steady_clock::now();
            const double feature_cpu = process_cpu_seconds_now();
            (void)dgemm_nn_partitioned_rows(
                feature_rows, sample_width, fixed_rank_,
                coefficients.data(), feature_rows,
                fixed_tile.data(), fixed_rank_,
                feature_tile.data(), feature_rows, threads_,
                -1.0, 0.0, &feature_resolution
            );
            record_gemm_shape(
                "pass2_component_diagonal_residual_nn", false,
                feature_rows, sample_width, fixed_rank_,
                generalized_elapsed_seconds(feature_wall),
                std::max(0.0, process_cpu_seconds_now() - feature_cpu)
            );
            record_integrity_resolution(feature_resolution);
        }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_)
#endif
        for (int sample = 0; sample < sample_width; ++sample) {
            for (int coordinate = 0; coordinate < q_; ++coordinate) {
                const int global_sample = sample_start + sample;
                double* destination = feature_tile.data()
                    + (static_cast<size_t>(sample)
                           * static_cast<size_t>(q_)
                       + static_cast<size_t>(coordinate))
                        * static_cast<size_t>(width);
                const double phi = basis_[
                    static_cast<size_t>(coordinate)
                        * static_cast<size_t>(rows_)
                    + static_cast<size_t>(global_sample)
                ];
                const double* genotype_values = genotype_tile.data()
                    + static_cast<size_t>(sample)
                        * static_cast<size_t>(width);
                if (fixed_rank_ > 0) {
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (int variant = 0; variant < width; ++variant) {
                        destination[variant] +=
                            phi * genotype_values[variant];
                    }
                } else {
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (int variant = 0; variant < width; ++variant) {
                        destination[variant] =
                            phi * genotype_values[variant];
                    }
                }
            }
        }

        for (int pair_start = 0; pair_start < pairs_;
             pair_start += pair_batch_width) {
            const int pair_stop = std::min(
                pairs_, pair_start + pair_batch_width
            );
            const int pair_width = pair_stop - pair_start;
            GemmIntegrityResolution resolution;
            const auto annotation_wall = std::chrono::steady_clock::now();
            const double annotation_cpu = process_cpu_seconds_now();
            if (singleton_annotation_rows_) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int sample = 0; sample < sample_width; ++sample) {
                    for (int local_pair = 0; local_pair < pair_width;
                         ++local_pair) {
                        double* destination = annotation_product.data()
                            + (static_cast<size_t>(local_pair)
                                   * static_cast<size_t>(sample_width)
                               + static_cast<size_t>(sample))
                                * static_cast<size_t>(annotations_count_);
                        std::fill(
                            destination,
                            destination
                                + static_cast<size_t>(annotations_count_),
                            0.0
                        );
                    }
                    for (int variant = 0; variant < width; ++variant) {
                        const size_t global_variant =
                            static_cast<size_t>(start + variant);
                        const int annotation = singleton_annotation_index_[
                            global_variant
                        ];
                        const double weight = singleton_annotation_weight_[
                            global_variant
                        ];
                        for (int local_pair = 0; local_pair < pair_width;
                             ++local_pair) {
                            const int pair = pair_start + local_pair;
                            const int left = pair_q(pair);
                            const int right = pair_r(pair);
                            const double* left_values = feature_tile.data()
                                + (static_cast<size_t>(sample)
                                       * static_cast<size_t>(q_)
                                   + static_cast<size_t>(left))
                                    * static_cast<size_t>(width);
                            const double* right_values = feature_tile.data()
                                + (static_cast<size_t>(sample)
                                       * static_cast<size_t>(q_)
                                   + static_cast<size_t>(right))
                                    * static_cast<size_t>(width);
                            const double pair_value = pair_factor(pair)
                                * left_values[variant]
                                * right_values[variant];
                            double* destination = annotation_product.data()
                                + (static_cast<size_t>(local_pair)
                                       * static_cast<size_t>(sample_width)
                                   + static_cast<size_t>(sample))
                                    * static_cast<size_t>(
                                        annotations_count_
                                    );
                            destination[annotation] += weight * pair_value;
                        }
                    }
                }
                record_gemm_shape(
                    "pass2_component_diagonal_singleton_fused", false,
                    1, sample_width * pair_width, width,
                    generalized_elapsed_seconds(annotation_wall),
                    std::max(
                        0.0, process_cpu_seconds_now() - annotation_cpu
                    )
                );
            } else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_)
#endif
                for (int local_pair = 0; local_pair < pair_width;
                     ++local_pair) {
                    for (int sample = 0; sample < sample_width; ++sample) {
                        const int pair = pair_start + local_pair;
                        const int left = pair_q(pair);
                        const int right = pair_r(pair);
                        const double factor = pair_factor(pair);
                        const double* left_values = feature_tile.data()
                            + (static_cast<size_t>(sample)
                                   * static_cast<size_t>(q_)
                               + static_cast<size_t>(left))
                                * static_cast<size_t>(width);
                        const double* right_values = feature_tile.data()
                            + (static_cast<size_t>(sample)
                                   * static_cast<size_t>(q_)
                               + static_cast<size_t>(right))
                                * static_cast<size_t>(width);
                        double* destination = pair_tile.data()
                            + (static_cast<size_t>(local_pair)
                                   * static_cast<size_t>(sample_width)
                               + static_cast<size_t>(sample))
                                * static_cast<size_t>(width);
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (int variant = 0; variant < width; ++variant) {
                            destination[variant] = factor
                                * left_values[variant]
                                * right_values[variant];
                        }
                    }
                }
                (void)dgemm_nn_partitioned_rows(
                    annotations_count_, sample_width * pair_width, width,
                    annotation_values_.data()
                        + static_cast<size_t>(start)
                            * static_cast<size_t>(annotations_count_),
                    annotations_count_, pair_tile.data(), width,
                    annotation_product.data(), annotations_count_, threads_,
                    1.0, 0.0, &resolution
                );
                record_gemm_shape(
                    "pass2_component_diagonal_annotation_nn", false,
                    annotations_count_, sample_width * pair_width, width,
                    generalized_elapsed_seconds(annotation_wall),
                    std::max(
                        0.0, process_cpu_seconds_now() - annotation_cpu
                    )
                );
                record_integrity_resolution(resolution);
            }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads_)
#endif
            for (int local_pair = 0; local_pair < pair_width; ++local_pair) {
                for (int annotation = 0;
                     annotation < annotations_count_; ++annotation) {
                    const int pair = pair_start + local_pair;
                    const int component = annotation * pairs_ + pair;
                    double* destination = component_kernel_diagonal_.data()
                        + static_cast<size_t>(component)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(sample_start);
                    const double* source = annotation_product.data()
                        + static_cast<size_t>(local_pair)
                            * static_cast<size_t>(sample_width)
                            * static_cast<size_t>(annotations_count_)
                        + static_cast<size_t>(annotation);
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (int sample = 0; sample < sample_width; ++sample) {
                        destination[sample] += source[
                            static_cast<size_t>(sample)
                                * static_cast<size_t>(annotations_count_)
                        ];
                    }
                }
            }
        }
        set_progress_units(
            (static_cast<int64_t>(sample_stop)
                + same_person_sample_tile_width_ - 1) /
                same_person_sample_tile_width_,
            (static_cast<int64_t>(rows_)
                + same_person_sample_tile_width_ - 1) /
                same_person_sample_tile_width_
        );
        check_cancelled();
    }
    component_diagonal_wall_seconds_ += generalized_elapsed_seconds(wall);
    component_diagonal_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - cpu
    );
}

void GeneralizedGxELDScoreDirectContext::execute_pass2() {
    if (state_ != State::sources_sealed || observed_passes_ != 1 ||
        observed_variant_visits_ != variants_) {
        throw std::runtime_error("Invalid generalized pass-2 transition");
    }
    state_ = State::pass2_running;
    ++observed_passes_;
    std::fill(
        affine_mean_.begin(), affine_mean_.end(),
        std::numeric_limits<double>::quiet_NaN()
    );
    std::fill(
        affine_inverse_scale_.begin(), affine_inverse_scale_.end(),
        std::numeric_limits<double>::quiet_NaN()
    );
    const int family_count = q_ * q_;
    GeneralizedUninitializedDoubleBuffer genotype;
    MailmanPackedBlock packed;
    std::vector<double> rhs;
    std::vector<double> cross;
    std::vector<double> lrow;
    std::vector<double> lsum;
    std::vector<double> tile_sum;
    std::vector<double> directed_block;
    ComponentDiagonalWorkspace component_workspace;
    std::unique_ptr<NativeGemmOutputAllocation> dense_cross;
    const int maximum_width = std::min(variants_, variant_block_width_);
    const int maximum_rhs_columns = checked_blas_dim(
        checked_mul(
            checked_mul(
                static_cast<size_t>(probe_tile_width_),
                static_cast<size_t>(family_count),
                "generalized pass-2 batched RHS columns"
            ),
            static_cast<size_t>(target_annotation_batch_width_),
            "generalized pass-2 batched RHS columns"
        ),
        "generalized pass-2 batched RHS columns"
    );
    genotype.reserve(checked_mul(
        static_cast<size_t>(rows_), static_cast<size_t>(maximum_width),
        "generalized pass-2 genotype arena"
    ));
    if (!precompute_rhs_) {
        rhs.reserve(checked_mul(
            static_cast<size_t>(rows_),
            static_cast<size_t>(maximum_rhs_columns),
            "generalized pass-2 RHS arena"
        ));
    }
    cross.reserve(checked_mul(
        static_cast<size_t>(maximum_width),
        static_cast<size_t>(maximum_rhs_columns),
        "generalized pass-2 cross arena"
    ));
    lrow.reserve(checked_mul(
        checked_mul(
            static_cast<size_t>(maximum_width),
            static_cast<size_t>(pairs_),
            "generalized pass-2 row arena"
        ),
        static_cast<size_t>(components_),
        "generalized pass-2 row arena"
    ));
    lsum.reserve(checked_mul(
        checked_mul(
            checked_mul(
                static_cast<size_t>(maximum_width),
                static_cast<size_t>(pairs_),
                "generalized pass-2 reduction arena"
            ),
            static_cast<size_t>(pairs_),
            "generalized pass-2 reduction arena"
        ),
        static_cast<size_t>(target_annotation_batch_width_),
        "generalized pass-2 reduction arena"
    ));
    tile_sum.reserve(lsum.capacity());
    directed_block.reserve(checked_mul(
        checked_mul(
            static_cast<size_t>(annotations_count_),
            static_cast<size_t>(pairs_),
            "generalized directed block"
        ),
        static_cast<size_t>(components_),
        "generalized directed block"
    ));
    const int64_t sample_tiles =
        (static_cast<int64_t>(rows_) + same_person_sample_tile_width_ - 1) /
        same_person_sample_tile_width_;
    const int64_t target_tiles =
        (static_cast<int64_t>(probes_) + probe_tile_width_ - 1) /
        probe_tile_width_;
    const int64_t target_units = target_tiles * annotations_count_;
    for (int start = 0; start < variants_; start += variant_block_width_) {
        const int stop = std::min(variants_, start + variant_block_width_);
        const int width = stop - start;
        const int64_t block = start / variant_block_width_;
        check_cancelled();
        set_progress(
            ProgressPhase::pass2, ProgressSubphase::decode,
            block, block, start, stop, 0, 1
        );
        if (dense_blas_hybrid_) {
            decode_block(start, stop, genotype);
        } else {
            read_packed_block(start, stop, packed);
        }
        set_progress(
            ProgressPhase::pass2, ProgressSubphase::component_diagonal,
            block, block, start, stop, 0, sample_tiles
        );
        accumulate_component_kernel_diagonal(
            start, stop,
            dense_blas_hybrid_ ? &genotype : nullptr,
            dense_blas_hybrid_ ? nullptr : &packed,
            component_workspace
        );
        lrow.assign(checked_mul(
            checked_mul(
                static_cast<size_t>(width), static_cast<size_t>(pairs_),
                "generalized target row block"
            ),
            static_cast<size_t>(components_),
            "generalized target row block"
        ), 0.0);
        int64_t completed_target_units = 0;
        for (int annotation_start = 0;
             annotation_start < annotations_count_;
             annotation_start += target_annotation_batch_width_) {
            const int annotation_stop = std::min(
                annotations_count_,
                annotation_start + target_annotation_batch_width_
            );
            const int annotation_width =
                annotation_stop - annotation_start;
            const int cells = pairs_ * pairs_;
            lsum.assign(checked_mul(
                checked_mul(
                    checked_mul(
                        static_cast<size_t>(width),
                        static_cast<size_t>(pairs_),
                        "generalized target pair products"
                    ),
                    static_cast<size_t>(pairs_),
                    "generalized target pair products"
                ),
                static_cast<size_t>(annotation_width),
                "generalized target pair products"
            ), 0.0);
            for (int probe_start = 0; probe_start < probes_;
                 probe_start += probe_tile_width_) {
                const int probe_stop = std::min(
                    probes_, probe_start + probe_tile_width_
                );
                const int probe_width = probe_stop - probe_start;
                const int columns_per_annotation =
                    probe_width * family_count;
                const int rhs_columns = checked_blas_dim(
                    checked_mul(
                        static_cast<size_t>(annotation_width),
                        static_cast<size_t>(columns_per_annotation),
                        "generalized target batched RHS columns"
                    ),
                    "generalized target batched RHS columns"
                );
                set_progress(
                    ProgressPhase::pass2,
                    ProgressSubphase::rhs_construction,
                    block, block, start, stop,
                    completed_target_units, target_units
                );
                const double* rhs_data = nullptr;
                if (precompute_rhs_) {
                    if (precomputed_rhs_ == nullptr) {
                        throw std::runtime_error(
                            "Generalized precomputed RHS is unavailable"
                        );
                    }
                    const size_t first_column = checked_mul(
                        static_cast<size_t>(probe_start),
                        checked_mul(
                            static_cast<size_t>(annotations_count_),
                            static_cast<size_t>(family_count),
                            "generalized precomputed RHS tile offset"
                        ),
                        "generalized precomputed RHS tile offset"
                    ) + checked_mul(
                        static_cast<size_t>(annotation_start),
                        static_cast<size_t>(columns_per_annotation),
                        "generalized precomputed RHS annotation offset"
                    );
                    rhs_data = precomputed_rhs_->data()
                        + first_column * static_cast<size_t>(rows_);
                } else {
                    const auto rhs_wall = std::chrono::steady_clock::now();
                    const double rhs_cpu = process_cpu_seconds_now();
                    rhs.resize(checked_mul(
                        static_cast<size_t>(rows_),
                        static_cast<size_t>(rhs_columns),
                        "generalized target RHS"
                    ));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                    for (int column = 0; column < rhs_columns; ++column) {
                        const int local_annotation =
                            column / columns_per_annotation;
                        const int within_annotation =
                            column % columns_per_annotation;
                        const int local_probe =
                            within_annotation / family_count;
                        const int family = within_annotation % family_count;
                        const int target_coordinate = family / q_;
                        const int source_coordinate = family % q_;
                        const int global_probe = probe_start + local_probe;
                        const int annotation =
                            annotation_start + local_annotation;
                        const double* source = contextual_sources_.data()
                            + source_index(
                                annotation, source_coordinate, 0,
                                global_probe
                            );
                        const double* phi = basis_.data()
                            + static_cast<size_t>(target_coordinate)
                                * static_cast<size_t>(rows_);
                        double* destination = rhs.data()
                            + static_cast<size_t>(column)
                                * static_cast<size_t>(rows_);
                        for (int row = 0; row < rows_; ++row) {
                            destination[row] = phi[row] * source[row];
                        }
                    }
                    rhs_wall_seconds_ += generalized_elapsed_seconds(rhs_wall);
                    rhs_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - rhs_cpu
                    );
                    rhs_data = rhs.data();
                }
                double* cross_data = cross.data();
                set_progress(
                    ProgressPhase::pass2,
                    ProgressSubphase::target_product,
                    block, block, start, stop,
                    completed_target_units, target_units
                );
                if (dense_blas_hybrid_) {
                    NativeGemmOutputTelemetryScope output_scope;
                    const size_t logical_elements = checked_mul(
                        static_cast<size_t>(width),
                        static_cast<size_t>(rhs_columns),
                        "generalized target logical output"
                    );
                    const size_t logical_bytes = checked_mul(
                        logical_elements, sizeof(double),
                        "generalized target logical output bytes"
                    );
                    const size_t capacity_bytes = checked_mul(
                        checked_mul(
                            static_cast<size_t>(maximum_width),
                            static_cast<size_t>(maximum_rhs_columns),
                            "generalized target output capacity"
                        ),
                        sizeof(double),
                        "generalized target output capacity bytes"
                    );
                    if (dense_cross == nullptr) {
                        dense_cross =
                            std::make_unique<NativeGemmOutputAllocation>(
                                static_cast<size_t>(width),
                                static_cast<size_t>(rhs_columns),
                                "column_major", logical_bytes,
                                output_scope.request(),
                                output_scope.evidence(), capacity_bytes
                            );
                    } else {
                        dense_cross->reuse_for_call(
                            static_cast<size_t>(width),
                            static_cast<size_t>(rhs_columns),
                            "column_major", logical_bytes,
                            output_scope.request(), output_scope.evidence()
                        );
                    }
                    cross_data = dense_cross->data();
                    output_scope.register_output(
                        cross_data, logical_bytes, CblasColMajor
                    );
                    GemmIntegrityResolution resolution;
                    const auto product_wall = std::chrono::steady_clock::now();
                    const double product_cpu = process_cpu_seconds_now();
                    try {
                        (void)dgemm_tn_partitioned_rows(
                            width, rhs_columns, rows_, genotype.data(), rows_,
                            rhs_data, rows_, cross_data, width, threads_,
                            1.0, 0.0, nullptr, precompute_rhs_, &resolution
                        );
                    } catch (const RetryableGemmInputMutation&) {
                        ++retry_count_;
                        decode_block(start, stop, genotype);
                        dgemm_tn_tiled(
                            width, rhs_columns, rows_, genotype.data(), rows_,
                            rhs_data, rows_, cross_data, width, threads_
                        );
                        ++fallback_count_;
                    }
                    const double product_wall_seconds =
                        generalized_elapsed_seconds(product_wall);
                    const double product_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - product_cpu
                    );
                    target_tn_wall_seconds_ += product_wall_seconds;
                    target_tn_cpu_seconds_ += product_cpu_seconds;
                    record_gemm_shape(
                        "pass2_target_tn", true,
                        width, rhs_columns, rows_,
                        product_wall_seconds, product_cpu_seconds
                    );
                    record_integrity_resolution(resolution);
                    dense_cross->verify_after_repair();
                    const auto audit_wall = std::chrono::steady_clock::now();
                    const double audit_cpu = process_cpu_seconds_now();
                    audit_tn_product(
                        genotype.data(), width, rows_, rhs_data,
                        rhs_columns, cross_data,
                        static_cast<int>(target_tn_calls_)
                    );
                    const double audit_wall_seconds =
                        generalized_elapsed_seconds(audit_wall);
                    const double audit_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - audit_cpu
                    );
                    independent_audit_wall_seconds_ += audit_wall_seconds;
                    independent_audit_cpu_seconds_ += audit_cpu_seconds;
                    target_independent_audit_wall_seconds_ +=
                        audit_wall_seconds;
                    target_independent_audit_cpu_seconds_ +=
                        audit_cpu_seconds;
                    ++gemm_output_numa_call_count_;
                    if (gemm_output_numa_evidence_.size() < 64U) {
                        gemm_output_numa_evidence_.push_back(
                            output_scope.evidence()
                        );
                    }
                    output_scope.complete();
                } else {
                    cross.resize(checked_mul(
                        static_cast<size_t>(width),
                        static_cast<size_t>(rhs_columns),
                        "generalized target cross sketch"
                    ));
                    const auto product_wall = std::chrono::steady_clock::now();
                    const double product_cpu = process_cpu_seconds_now();
                    packed_target_product(
                        packed, rhs_data, rhs_columns, cross
                    );
                    cross_data = cross.data();
                    const double product_wall_seconds =
                        generalized_elapsed_seconds(product_wall);
                    const double product_cpu_seconds = std::max(
                        0.0, process_cpu_seconds_now() - product_cpu
                    );
                    target_tn_wall_seconds_ += product_wall_seconds;
                    target_tn_cpu_seconds_ += product_cpu_seconds;
                    record_gemm_shape(
                        "pass2_target_packed", true,
                        width, rhs_columns, rows_,
                        product_wall_seconds, product_cpu_seconds
                    );
                }
                ++target_tn_calls_;
                set_progress(
                    ProgressPhase::pass2,
                    ProgressSubphase::row_reduction,
                    block, block, start, stop,
                    completed_target_units, target_units
                );
                const auto reduction_wall = std::chrono::steady_clock::now();
                const double reduction_cpu = process_cpu_seconds_now();
                const double inverse_rank = 1.0 /
                    static_cast<double>(residual_rank_);
                const size_t cross_elements = checked_mul(
                    static_cast<size_t>(width),
                    static_cast<size_t>(rhs_columns),
                    "generalized target cross scaling"
                );
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int64_t index = 0;
                     index < static_cast<int64_t>(cross_elements); ++index) {
                    cross_data[index] *= inverse_rank;
                }
                const int batched_cells = annotation_width * cells;
                const size_t cell_elements = checked_mul(
                    static_cast<size_t>(batched_cells),
                    static_cast<size_t>(width),
                    "generalized target cell-major tile products"
                );
                tile_sum.assign(cell_elements, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_)
#endif
                for (int batched_cell = 0;
                     batched_cell < batched_cells; ++batched_cell) {
                    const int local_annotation = batched_cell / cells;
                    const int cell = batched_cell % cells;
                    double* destination = tile_sum.data()
                        + static_cast<size_t>(batched_cell)
                            * static_cast<size_t>(width);
                    const int64_t term_start = product_offsets_[
                        static_cast<size_t>(cell)
                    ];
                    const int64_t term_stop = product_offsets_[
                        static_cast<size_t>(cell + 1)
                    ];
                    for (int64_t term = term_start; term < term_stop;
                         ++term) {
                        const size_t term_offset =
                            4U * static_cast<size_t>(term);
                        const int first_target = static_cast<int>(
                            product_terms_[term_offset]
                        );
                        const int first_source = static_cast<int>(
                            product_terms_[term_offset + 1U]
                        );
                        const int second_target = static_cast<int>(
                            product_terms_[term_offset + 2U]
                        );
                        const int second_source = static_cast<int>(
                            product_terms_[term_offset + 3U]
                        );
                        for (int probe = 0; probe < probe_width; ++probe) {
                            const int first_column =
                                local_annotation * columns_per_annotation
                                + probe * family_count
                                + first_target * q_ + first_source;
                            const int second_column =
                                local_annotation * columns_per_annotation
                                + probe * family_count
                                + second_target * q_ + second_source;
                            const double* first = cross_data
                                + static_cast<size_t>(first_column)
                                    * static_cast<size_t>(width);
                            const double* second = cross_data
                                + static_cast<size_t>(second_column)
                                    * static_cast<size_t>(width);
#ifdef _OPENMP
#pragma omp simd
#endif
                            for (int variant = 0; variant < width; ++variant) {
                                destination[variant] +=
                                    first[variant] * second[variant];
                            }
                        }
                    }
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int64_t index = 0;
                     index < static_cast<int64_t>(cell_elements); ++index) {
                    lsum[static_cast<size_t>(index)] +=
                        tile_sum[static_cast<size_t>(index)];
                }
                reduction_wall_seconds_ +=
                    generalized_elapsed_seconds(reduction_wall);
                reduction_cpu_seconds_ += std::max(
                    0.0, process_cpu_seconds_now() - reduction_cpu
                );
                completed_target_units += annotation_width;
                set_progress_units(completed_target_units, target_units);
                check_cancelled();
            }
            const auto block_reduction_wall =
                std::chrono::steady_clock::now();
            const double block_reduction_cpu = process_cpu_seconds_now();
            for (int local_annotation = 0;
                 local_annotation < annotation_width;
                 ++local_annotation) {
                const int annotation =
                    annotation_start + local_annotation;
                for (int variant = 0; variant < width; ++variant) {
                    for (int target_pair = 0; target_pair < pairs_;
                         ++target_pair) {
                        for (int source_pair = 0; source_pair < pairs_;
                             ++source_pair) {
                            const int component = annotation * pairs_
                                + source_pair;
                            lrow[
                                (static_cast<size_t>(variant)
                                    * static_cast<size_t>(pairs_)
                                 + static_cast<size_t>(target_pair))
                                    * static_cast<size_t>(components_)
                                + static_cast<size_t>(component)
                            ] = lsum[
                                (static_cast<size_t>(local_annotation)
                                    * static_cast<size_t>(cells)
                                 + static_cast<size_t>(target_pair * pairs_)
                                 + static_cast<size_t>(source_pair))
                                    * static_cast<size_t>(width)
                                + static_cast<size_t>(variant)
                            ] / static_cast<double>(probes_);
                        }
                    }
                }
            }
            reduction_wall_seconds_ +=
                generalized_elapsed_seconds(block_reduction_wall);
            reduction_cpu_seconds_ += std::max(
                0.0, process_cpu_seconds_now() - block_reduction_cpu
            );
        }

        set_progress(
            ProgressPhase::pass2, ProgressSubphase::output_reduction,
            block, block, start, stop, 0, width
        );
        const auto output_reduction_wall = std::chrono::steady_clock::now();
        const double output_reduction_cpu = process_cpu_seconds_now();
        const int directed_columns = checked_blas_dim(
            checked_mul(
                static_cast<size_t>(pairs_),
                static_cast<size_t>(components_),
                "generalized directed aggregation columns"
            ),
            "generalized directed aggregation columns"
        );
        const size_t directional_block_offset = checked_mul(
            static_cast<size_t>(start),
            static_cast<size_t>(directed_columns),
            "generalized directional block offset"
        );
        int invalid_lrow = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) reduction(|:invalid_lrow)
#endif
        for (int64_t index = 0;
             index < static_cast<int64_t>(lrow.size()); ++index) {
            const double value = lrow[static_cast<size_t>(index)];
            invalid_lrow |= !std::isfinite(value);
            directional_[
                directional_block_offset + static_cast<size_t>(index)
            ] = value;
        }
        if (invalid_lrow != 0) {
            throw std::runtime_error(
                "Generalized directional target row is non-finite"
            );
        }
        directed_block.resize(checked_mul(
            static_cast<size_t>(annotations_count_),
            static_cast<size_t>(directed_columns),
            "generalized directed block"
        ));
        const MatrixFingerprint lrow_fingerprint =
            fingerprint_row_major_matrix(
                width, directed_columns, lrow.data(), directed_columns,
                threads_
            );
        const auto directed_wall = std::chrono::steady_clock::now();
        const double directed_cpu = process_cpu_seconds_now();
        const int64_t directed_repairs = dgemm_row_tn_partitioned(
            annotations_count_, directed_columns, width,
            annotation_values_.data()
                + static_cast<size_t>(start)
                    * static_cast<size_t>(annotations_count_),
            annotations_count_, lrow.data(), directed_columns,
            directed_block.data(), directed_columns, threads_,
            lrow_fingerprint
        );
        const double directed_wall_seconds =
            generalized_elapsed_seconds(directed_wall);
        const double directed_cpu_seconds = std::max(
            0.0, process_cpu_seconds_now() - directed_cpu
        );
        directed_aggregation_wall_seconds_ += directed_wall_seconds;
        directed_aggregation_cpu_seconds_ += directed_cpu_seconds;
        record_gemm_shape(
            "pass2_directed_aggregation_tn", true,
            annotations_count_, directed_columns, width,
            directed_wall_seconds, directed_cpu_seconds
        );
        repaired_columns_ += directed_repairs;
        for (size_t index = 0; index < directed_block.size(); ++index) {
            directed_[index] += directed_block[index];
        }
        reduction_wall_seconds_ +=
            generalized_elapsed_seconds(output_reduction_wall);
        reduction_cpu_seconds_ += std::max(
            0.0, process_cpu_seconds_now() - output_reduction_cpu
        );
        set_progress(
            ProgressPhase::pass2, ProgressSubphase::output_reduction,
            block + 1, block, start, stop, width, width
        );
    }
    reader_->check_files_unchanged();
    if (observed_variant_visits_ != 2LL * variants_) {
        throw std::runtime_error(
            "Generalized pass 2 did not visit every variant exactly once"
        );
    }
    precomputed_rhs_.reset();
}

void GeneralizedGxELDScoreDirectContext::finalize_scientific_outputs() {
    if (state_ != State::pass2_running || observed_passes_ != 2 ||
        observed_variant_visits_ != 2LL * variants_) {
        throw std::runtime_error(
            "Generalized scientific finalization lacks two complete passes"
        );
    }
    check_cancelled();
    const int64_t total_blocks = progress_total_blocks_.load(
        std::memory_order_relaxed
    );
    set_progress(
        ProgressPhase::finalize, ProgressSubphase::scientific_reduction,
        total_blocks, -1, variants_, variants_, 0, components_
    );
    for (int row = 0; row < components_; ++row) {
        for (int column = 0; column < components_; ++column) {
            const size_t index = static_cast<size_t>(row)
                * static_cast<size_t>(components_)
                + static_cast<size_t>(column);
            const size_t reverse = static_cast<size_t>(column)
                * static_cast<size_t>(components_)
                + static_cast<size_t>(row);
            presymmetry_absolute_error_ = std::max(
                presymmetry_absolute_error_,
                std::abs(directed_[index] - directed_[reverse])
            );
        }
    }
    double directed_maximum = 0.0;
    for (double value : directed_) {
        directed_maximum = std::max(directed_maximum, std::abs(value));
    }
    presymmetry_relative_error_ = presymmetry_absolute_error_ /
        std::max(1.0, directed_maximum);
    const double rank_squared = static_cast<double>(
        static_cast<int64_t>(residual_rank_) * residual_rank_
    );
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (int component = 0; component < components_; ++component) {
        const int annotation = static_cast<int>(component_table_[
            2U * static_cast<size_t>(component)
        ]);
        const double inverse_mass = 1.0 / annotation_masses_[
            static_cast<size_t>(annotation)
        ];
        double* diagonal = component_kernel_diagonal_.data()
            + static_cast<size_t>(component) * static_cast<size_t>(rows_);
        for (int row = 0; row < rows_; ++row) {
            diagonal[row] *= inverse_mass;
        }
    }
    const auto same_person_wall = std::chrono::steady_clock::now();
    const double same_person_cpu = process_cpu_seconds_now();
    GemmIntegrityResolution same_person_resolution;
    (void)dgemm_tn_partitioned_rows(
        components_, components_, rows_,
        component_kernel_diagonal_.data(), rows_,
        component_kernel_diagonal_.data(), rows_,
        same_person_.data(), components_, threads_,
        1.0, 0.0, nullptr, false, &same_person_resolution
    );
    record_gemm_shape(
        "finalize_same_person_tn", true,
        components_, components_, rows_,
        generalized_elapsed_seconds(same_person_wall),
        std::max(0.0, process_cpu_seconds_now() - same_person_cpu)
    );
    record_integrity_resolution(same_person_resolution);
    for (int left = 0; left < components_; ++left) {
        for (int right = left; right < components_; ++right) {
            const size_t forward = static_cast<size_t>(left)
                * static_cast<size_t>(components_)
                + static_cast<size_t>(right);
            const size_t reverse = static_cast<size_t>(right)
                * static_cast<size_t>(components_)
                + static_cast<size_t>(left);
            same_person_presymmetry_error_ = std::max(
                same_person_presymmetry_error_,
                std::abs(same_person_[forward] - same_person_[reverse])
            );
            const double value = 0.5
                * (same_person_[forward] + same_person_[reverse]);
            same_person_[forward] = value;
            same_person_[reverse] = value;
        }
    }
    same_person_wall_seconds_ += generalized_elapsed_seconds(same_person_wall);
    same_person_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - same_person_cpu
    );
    for (int row = 0; row < components_; ++row) {
        const int row_annotation = static_cast<int>(component_table_[
            2U * static_cast<size_t>(row)
        ]);
        for (int column = 0; column < components_; ++column) {
            const int column_annotation = static_cast<int>(component_table_[
                2U * static_cast<size_t>(column)
            ]);
            const size_t index = static_cast<size_t>(row)
                * static_cast<size_t>(components_)
                + static_cast<size_t>(column);
            const size_t reverse = static_cast<size_t>(column)
                * static_cast<size_t>(components_)
                + static_cast<size_t>(row);
            symmetric_[index] = 0.5
                * (directed_[index] + directed_[reverse]);
            genetic_gram_[index] = rank_squared * symmetric_[index]
                / (annotation_masses_[static_cast<size_t>(row_annotation)]
                   * annotation_masses_[
                       static_cast<size_t>(column_annotation)
                   ]);
        }
    }
    for (const std::vector<double>* values : {
            &directional_, &directed_, &symmetric_, &genetic_gram_,
            &same_person_, &component_kernel_diagonal_
         }) {
        int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) reduction(|:invalid)
#endif
        for (int64_t index = 0;
             index < static_cast<int64_t>(values->size()); ++index) {
            invalid |= !std::isfinite((*values)[static_cast<size_t>(index)]);
        }
        if (invalid != 0) {
            throw std::runtime_error(
                "Generalized finalized output is non-finite"
            );
        }
    }
    std::vector<double>().swap(weighted_fixed_);
    if (!retain_contextual_sources_) {
        contextual_sources_.clear();
    }
    if (!publish_component_kernel_diagonal_) {
        std::vector<double>().swap(component_kernel_diagonal_);
    }
}

nb::dict GeneralizedGxELDScoreDirectContext::publish_result() {
    if (state_ != State::finalized) {
        throw std::runtime_error(
            "Generalized native outputs cannot be published before finalization"
        );
    }
    nb::dict shapes;
    shapes["base_sources"] = nb::make_tuple(
        annotations_count_, rows_, probes_
    );
    shapes["contextual_sources"] = nb::make_tuple(
        annotations_count_, q_, rows_, probes_
    );
    shapes["component_kernel_diagonal"] = nb::make_tuple(
        components_, rows_
    );
    shapes["directional_ldscores"] = nb::make_tuple(
        variants_, pairs_, components_
    );

    nb::dict ledger;
    ledger["planned_reference_genotype_passes"] = 2;
    ledger["observed_reference_genotype_passes"] = observed_passes_;
    ledger["planned_retained_variant_visits"] = 2LL * variants_;
    ledger["observed_retained_variant_visits"] = observed_variant_visits_;
    ledger["observed_block_reads"] = observed_block_reads_;
    ledger["missing_genotype_calls"] = missing_calls_;
    ledger["duplicate_variant_visits"] = 0;
    ledger["duplicate_retained_variant_visits"] = 0;
    const int64_t decoded_blocks_per_pass =
        (static_cast<int64_t>(variants_) + variant_block_width_ - 1)
        / variant_block_width_;
    ledger["pass1_decoded_blocks"] = decoded_blocks_per_pass;
    ledger["pass2_decoded_blocks"] = decoded_blocks_per_pass;
    ledger["retry_count"] = retry_count_;
    ledger["repair_count"] = repaired_columns_;
    ledger["fallback_count"] = fallback_count_;
    ledger["checksum_recomputed_columns"] = checksum_recomputed_columns_;
    ledger["roundoff_only_columns"] = roundoff_only_columns_;
    ledger["integrity_audit_count"] = integrity_audit_count_;
    ledger["protected_integrity_audit_count"] =
        protected_integrity_audit_count_;
    ledger["independent_integrity_audit_count"] =
        independent_integrity_audit_count_;
    ledger["integrity_failures"] = integrity_failures_;
    ledger["integrity_failure_count"] = integrity_failures_;
    ledger["state"] = "finalized";

    const uint64_t base_source_output_bytes = checked_mul(
        static_cast<uint64_t>(
            retain_base_sources_ ? base_sources_.size() : 0U
        ),
        8U, "generalized published base-source bytes"
    );
    const uint64_t contextual_source_output_bytes = checked_mul(
        static_cast<uint64_t>(contextual_sources_.size()), 8U,
        "generalized published contextual-source bytes"
    );
    const uint64_t component_diagonal_output_bytes = checked_mul(
        static_cast<uint64_t>(component_kernel_diagonal_.size()), 8U,
        "generalized published component-diagonal bytes"
    );
    const uint64_t directional_output_bytes = checked_mul(
        static_cast<uint64_t>(directional_.size()), 8U,
        "generalized published directional bytes"
    );
    const uint64_t genotype_scale_output_bytes = checked_mul(
        static_cast<uint64_t>(
            affine_mean_.size() + affine_inverse_scale_.size()
        ),
        8U, "generalized published genotype-scale bytes"
    );
    const uint64_t output_bytes = checked_mul(
        static_cast<uint64_t>(
            contextual_sources_.size() + same_person_.size()
            + component_kernel_diagonal_.size()
            + directional_.size() + directed_.size() + symmetric_.size()
            + genetic_gram_.size()
            + annotation_masses_.size()
            + affine_mean_.size() + affine_inverse_scale_.size()
            + (retain_base_sources_ ? base_sources_.size() : 0U)
        ),
        static_cast<uint64_t>(sizeof(double)),
        "generalized published output bytes"
    );
    nb::dict result;
    const auto publication_wall = std::chrono::steady_clock::now();
    const double publication_cpu = process_cpu_seconds_now();
    result["shapes"] = std::move(shapes);
    if (retain_base_sources_) {
        result["base_sources"] = copy_base_sources_for_publication();
        base_sources_.clear();
    } else {
        result["base_sources"] = nb::none();
    }
    if (retain_contextual_sources_) {
        result["contextual_sources"] =
            copy_contextual_sources_for_publication();
        contextual_sources_.clear();
    } else {
        result["contextual_sources"] = nb::none();
    }
    result["component_kernel_diagonal"] =
        publish_component_kernel_diagonal_
            ? nb::cast(move_matrix(
                component_kernel_diagonal_,
                static_cast<size_t>(components_),
                static_cast<size_t>(rows_)
            ))
            : nb::none();
    result["same_person"] = move_matrix(
        same_person_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["directional_ldscores"] = move_vector(directional_);
    result["directed_numerator"] = move_matrix(
        directed_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["symmetric_numerator"] = move_matrix(
        symmetric_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["genetic_gram"] = move_matrix(
        genetic_gram_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["annotation_masses"] = move_vector(annotation_masses_);
    // The downstream trait pass must use the exact affine scale sealed by
    // this reference run.  Publish these small per-variant vectors with the
    // ephemeral native result so callers do not need an extra genotype pass.
    // They are deliberately not part of the compact reference artifact.
    result["affine_mean"] = move_vector(affine_mean_);
    result["affine_inverse_scale"] =
        move_vector(affine_inverse_scale_);
    result["residual_rank"] = residual_rank_;
    result["genotype_scale_policy"] = "variant_affine_v1";
    result["allele_orientation"] = "bim_a1_counted_v1";
    result["allele_coding"] =
        "plink_bed_snp_major_diploid_hardcall_v1";
    result["centering_source"] = "provided_v1";
    result["centering_formula"] =
        "provided_variant_affine_mean_v1";
    result["scaling_formula"] =
        "dosage_minus_mean_times_inverse_scale_v1";
    result["missing_imputation"] = "sealed_mean_v1";
    result["ploidy_policy"] = "diploid_v1";
    publication_wall_seconds_ += generalized_elapsed_seconds(publication_wall);
    publication_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - publication_cpu
    );
    phase_finalize_wall_seconds_ += publication_wall_seconds_;
    phase_finalize_cpu_seconds_ += publication_cpu_seconds_;

    nb::dict phase_wall;
    phase_wall["pass1"] = phase_pass1_wall_seconds_;
    phase_wall["barrier"] = phase_barrier_wall_seconds_;
    phase_wall["pass2"] = phase_pass2_wall_seconds_;
    phase_wall["finalize"] = phase_finalize_wall_seconds_;
    nb::dict phase_cpu;
    phase_cpu["pass1"] = phase_pass1_cpu_seconds_;
    phase_cpu["barrier"] = phase_barrier_cpu_seconds_;
    phase_cpu["pass2"] = phase_pass2_cpu_seconds_;
    phase_cpu["finalize"] = phase_finalize_cpu_seconds_;
    nb::dict subphase_wall;
    subphase_wall["decode"] = decode_wall_seconds_;
    subphase_wall["rhs_formation"] = rhs_wall_seconds_;
    subphase_wall["rhs_precompute"] = rhs_precompute_wall_seconds_;
    subphase_wall["source_nn"] = source_nn_wall_seconds_;
    subphase_wall["projection"] = projection_wall_seconds_;
    subphase_wall["same_person"] = same_person_wall_seconds_;
    subphase_wall["component_kernel_diagonal"] =
        component_diagonal_wall_seconds_;
    subphase_wall["target_tn"] = target_tn_wall_seconds_;
    subphase_wall["row_product_reduction"] = reduction_wall_seconds_;
    subphase_wall["directed_aggregation"] =
        directed_aggregation_wall_seconds_;
    subphase_wall["independent_audits"] = independent_audit_wall_seconds_;
    subphase_wall["source_independent_audits"] =
        source_independent_audit_wall_seconds_;
    subphase_wall["target_independent_audits"] =
        target_independent_audit_wall_seconds_;
    subphase_wall["publication"] = publication_wall_seconds_;
    nb::dict subphase_cpu;
    subphase_cpu["decode"] = decode_cpu_seconds_;
    subphase_cpu["rhs_formation"] = rhs_cpu_seconds_;
    subphase_cpu["rhs_precompute"] = rhs_precompute_cpu_seconds_;
    subphase_cpu["source_nn"] = source_nn_cpu_seconds_;
    subphase_cpu["projection"] = projection_cpu_seconds_;
    subphase_cpu["same_person"] = same_person_cpu_seconds_;
    subphase_cpu["component_kernel_diagonal"] =
        component_diagonal_cpu_seconds_;
    subphase_cpu["target_tn"] = target_tn_cpu_seconds_;
    subphase_cpu["row_product_reduction"] = reduction_cpu_seconds_;
    subphase_cpu["directed_aggregation"] =
        directed_aggregation_cpu_seconds_;
    subphase_cpu["independent_audits"] = independent_audit_cpu_seconds_;
    subphase_cpu["source_independent_audits"] =
        source_independent_audit_cpu_seconds_;
    subphase_cpu["target_independent_audits"] =
        target_independent_audit_cpu_seconds_;
    subphase_cpu["publication"] = publication_cpu_seconds_;

    nb::list gemm_dimensions;
    for (const GeneralizedGxEGemmShapeRecord& record : gemm_shapes_) {
        nb::dict item;
        item["operation"] = record.operation;
        item["transpose_left"] = record.transpose_left;
        item["rows"] = record.rows;
        item["columns"] = record.columns;
        item["reduction"] = record.reduction;
        item["calls"] = record.calls;
        const double logical_flops = 2.0
            * static_cast<double>(record.rows)
            * static_cast<double>(record.columns)
            * static_cast<double>(record.reduction)
            * static_cast<double>(record.calls);
        item["logical_flops"] = logical_flops;
        item["wall_seconds"] = record.wall_seconds;
        item["process_cpu_seconds"] = record.process_cpu_seconds;
        item["effective_gflops"] = record.wall_seconds > 0.0
            ? logical_flops / (record.wall_seconds * 1.0e9)
            : 0.0;
        gemm_dimensions.append(std::move(item));
    }

    nb::list numa_outputs;
    for (const auto& evidence : gemm_output_numa_evidence_) {
        numa_outputs.append(native_gemm_output_numa_to_dict(evidence));
    }
    const NativeNumaContractRequest numa_request =
        native_numa_contract_request();
    nb::dict numa_evidence;
    numa_evidence["contract_required"] = numa_request.required;
    numa_evidence["selected_nodes"] = cpu_ids_to_list(numa_request.nodes);
    numa_evidence["gemm_outputs"] = std::move(numa_outputs);
    numa_evidence["gemm_output_count"] = gemm_output_numa_call_count_;
    numa_evidence["sampled_gemm_output_count"] =
        gemm_output_numa_evidence_.size();
    numa_evidence["sampling_policy"] = "first_64_calls_v1";
    numa_evidence["placement_repair_requested"] = false;

    const std::vector<int> configured_cpus = configured_openmp_cpu_ids();
    const CpuAffinityEvidence current_affinity =
        capture_cpu_affinity_evidence();
    nb::dict affinity;
    affinity["configured_cpu_ids"] = cpu_ids_to_list(configured_cpus);
    affinity["placement_contract_configured"] = !configured_cpus.empty();
    affinity["current_cpu"] = current_affinity.current_cpu;
    affinity["current_thread_cpu_count"] = current_affinity.cpu_count;
    affinity["current_thread_cpu_list"] = current_affinity.cpu_list;

    const uint64_t decoded_bytes = checked_mul(
        static_cast<uint64_t>(observed_variant_visits_),
        static_cast<uint64_t>(reader_->bytes_per_snp_),
        "generalized logical decoded bytes"
    );
    nb::dict resident_bytes;
    resident_bytes["base_sources"] = base_source_output_bytes;
    resident_bytes["contextual_sources"] = contextual_source_output_bytes;
    resident_bytes["component_kernel_diagonal"] =
        component_diagonal_output_bytes;
    resident_bytes["directional_panel"] = directional_output_bytes;
    resident_bytes["genotype_scale_identity"] =
        genotype_scale_output_bytes;
    resident_bytes["maximum_decoded_block"] = checked_mul(
        checked_mul(
            static_cast<uint64_t>(rows_),
            static_cast<uint64_t>(std::min(variants_, variant_block_width_)),
            "generalized resident decoded block"
        ),
        8U, "generalized resident decoded block bytes"
    );
    resident_bytes["maximum_rhs"] = checked_mul(
        checked_mul(
            checked_mul(
                static_cast<uint64_t>(rows_),
                static_cast<uint64_t>(probe_tile_width_ * q_ * q_),
                "generalized resident RHS"
            ),
            static_cast<uint64_t>(target_annotation_batch_width_),
            "generalized resident RHS"
        ),
        8U, "generalized resident RHS bytes"
    );
    resident_bytes["precomputed_rhs"] = rhs_precomputed_bytes_;
    resident_bytes["maximum_source_output"] = checked_mul(
        checked_mul(
            checked_mul(
                static_cast<uint64_t>(rows_),
                static_cast<uint64_t>(source_probe_tile_width_),
                "generalized resident source output"
            ),
            static_cast<uint64_t>(source_annotation_batch_width_),
            "generalized resident source output"
        ),
        8U, "generalized resident source output bytes"
    );
    resident_bytes["published_output"] = output_bytes;

    nb::dict telemetry;
    telemetry["schema"] =
        "summit.generalized_gxe.native_execution_telemetry.v2";
    telemetry["backend"] = dense_blas_hybrid_
        ? "descriptor_dense_protected_nn_tn"
        : "descriptor_packed_mailman";
    telemetry["descriptor_owner"] = "gxeldcore.DirectContext";
    telemetry["decoder"] = "DirectContext.decode_block_into";
    telemetry["common_genotype_scale"] =
        "mean_imputed_single_affine_ddof_fp64_v1";
    telemetry["packed_backend_used"] = !dense_blas_hybrid_;
    telemetry["source_probe_tile_width"] = source_probe_tile_width_;
    telemetry["source_annotation_batch_width"] =
        source_annotation_batch_width_;
    telemetry["source_annotation_batching"] =
        singleton_annotation_rows_ && dense_blas_hybrid_
            ? "single_nonzero_annotation_compacted_genotype_v1"
            : "contiguous_rhs_columns_shared_genotype_v1";
    telemetry["internal_source_layout"] =
        "annotation_coordinate_probe_sample_v1";
    telemetry["target_probe_tile_width"] = probe_tile_width_;
    telemetry["target_annotation_batch_width"] =
        target_annotation_batch_width_;
    telemetry["target_annotation_batching"] =
        "probe_tile_major_contiguous_rhs_v1";
    telemetry["rhs_precomputed"] = precompute_rhs_;
    telemetry["rhs_precomputed_columns"] = rhs_precomputed_columns_;
    telemetry["rhs_precomputed_bytes"] = rhs_precomputed_bytes_;
    telemetry["component_pair_batch_width"] =
        component_pair_batch_width_;
    telemetry["component_residualization"] =
        "stacked_context_fixed_rank_gemm_v1";
    telemetry["component_annotation_reduction"] =
        singleton_annotation_rows_
            ? "single_nonzero_annotation_fused_pair_reduction_ordered_v1"
            : "dense_annotation_gemm_v1";
    telemetry["row_product_reduction"] =
        "cell_major_parallel_contiguous_variant_v1";
    telemetry["directed_aggregation"] =
        "protected_row_major_tn_per_variant_block_v1";
    telemetry["publication"] = "zero_copy_vector_ownership_transfer_v1";
    telemetry["source_nn_calls"] = source_nn_calls_;
    telemetry["target_tn_calls"] = target_tn_calls_;
    telemetry["decode_threads"] = decode_threads_;
    telemetry["blas_threads"] = threads_;
    telemetry["phase_wall_seconds"] = std::move(phase_wall);
    telemetry["phase_cpu_seconds"] = std::move(phase_cpu);
    telemetry["subphase_wall_seconds"] = std::move(subphase_wall);
    telemetry["subphase_cpu_seconds"] = std::move(subphase_cpu);
    telemetry["gemm_dimensions"] = std::move(gemm_dimensions);
    telemetry["logical_genotype_bytes_read"] = decoded_bytes;
    telemetry["peak_rss_bytes"] = generalized_process_peak_rss_bytes();
    telemetry["output_bytes"] = output_bytes;
    telemetry["resident_bytes"] = std::move(resident_bytes);
    telemetry["affinity"] = std::move(affinity);
    telemetry["numa_evidence"] = std::move(numa_evidence);
    telemetry["checksum_recomputed_columns"] =
        checksum_recomputed_columns_;
    telemetry["roundoff_only_columns"] = roundoff_only_columns_;
    telemetry["integrity_audit_count"] = integrity_audit_count_;
    telemetry["protected_integrity_audit_count"] =
        protected_integrity_audit_count_;
    telemetry["independent_integrity_audit_count"] =
        independent_integrity_audit_count_;
    telemetry["integrity_failure_count"] = integrity_failures_;
    telemetry["qualification_fault_injected"] =
        qualification_fault_injected_;
    telemetry["full_serial_output_witness_calls"] = 0;
    telemetry["algebraic_checksum_policy"] =
        "mature_threshold_checks_plus_two_projection_periodic_audits_v2";
    telemetry["tile_induced_descriptor_rereads"] = 0;
    telemetry["maximum_projection_leakage"] =
        maximum_projection_leakage_;
    telemetry["maximum_relative_projection_leakage"] =
        maximum_relative_projection_leakage_;
    telemetry["same_person_presymmetry_error"] =
        same_person_presymmetry_error_;
    telemetry["same_person_method"] =
        "exact_component_kernel_diagonal_v1";
    telemetry["component_kernel_diagonal_published"] =
        publish_component_kernel_diagonal_;
    telemetry["contextual_sources_published"] =
        retain_contextual_sources_;
    telemetry["presymmetry_absolute_error"] =
        presymmetry_absolute_error_;
    telemetry["presymmetry_relative_error"] =
        presymmetry_relative_error_;
    telemetry["pass_ledger"] = ledger;

    result["ledger"] = ledger;
    result["telemetry"] = std::move(telemetry);
    return result;
}

void GeneralizedGxELDScoreDirectContext::clear_publishable_outputs() noexcept {
    precomputed_rhs_.reset();
    std::vector<double>().swap(sqrt_annotation_values_);
    base_sources_.clear();
    contextual_sources_.clear();
    std::vector<double>().swap(weighted_fixed_);
    std::vector<double>().swap(component_kernel_diagonal_);
    std::vector<double>().swap(same_person_);
    std::vector<double>().swap(directional_);
    std::vector<double>().swap(directed_);
    std::vector<double>().swap(symmetric_);
    std::vector<double>().swap(genetic_gram_);
    std::vector<double>().swap(affine_mean_);
    std::vector<double>().swap(affine_inverse_scale_);
    gemm_output_numa_call_count_ = 0;
    gemm_output_numa_evidence_.clear();
}
