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
        int probe_tile_width,
        int same_person_sample_tile_width,
        uint64_t max_workspace_bytes,
        int decode_threads,
        int requested_threads,
        bool retain_base_sources,
        bool dense_blas_hybrid,
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
          probe_tile_width_(probe_tile_width),
          same_person_sample_tile_width_(same_person_sample_tile_width),
          max_workspace_bytes_(max_workspace_bytes),
          threads_(requested_threads),
          decode_threads_(decode_threads),
          retain_base_sources_(retain_base_sources),
          dense_blas_hybrid_(dense_blas_hybrid),
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
            probe_tile_width_ <= 0 || probe_tile_width_ > probes_ ||
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
        reader_->check_files_unchanged();
    }

    GeneralizedGxELDScoreDirectContext(
        const GeneralizedGxELDScoreDirectContext&) = delete;
    GeneralizedGxELDScoreDirectContext& operator=(
        const GeneralizedGxELDScoreDirectContext&) = delete;

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
        result["target_probe_tile_width"] = probe_tile_width_;
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
        result["backend"] = dense_blas_hybrid_
            ? "descriptor_dense_protected_nn_tn"
            : "descriptor_packed_mailman";
        result["genotype_scale"] =
            "mean_imputed_single_affine_ddof_fp64_v1";
        return result;
    }

    nb::dict run() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (state_ != State::constructed) {
            throw std::runtime_error(
                "Generalized descriptor-native context is single-use"
            );
        }
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
                    seal_sources_and_same_person();
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
            return publish_result();
        } catch (...) {
            state_ = State::failed;
            clear_publishable_outputs();
            throw;
        }
    }

private:
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

    static nb_numpy_vec1<double> copy_vector(
        const std::vector<double>& source
    ) {
        double* destination = nullptr;
        auto result = make_owned_numpy_vec1<double>(
            source.size(), &destination
        );
        if (!source.empty()) {
            std::memcpy(
                destination, source.data(),
                checked_mul(source.size(), sizeof(double), "published vector")
            );
        }
        return result;
    }

    static nb_numpy_mat2c<double> copy_matrix(
        const std::vector<double>& source, size_t rows, size_t columns
    ) {
        if (source.size() != checked_mul(rows, columns, "published matrix")) {
            throw std::runtime_error("Published generalized matrix is mis-sized");
        }
        double* destination = nullptr;
        auto result = make_owned_numpy_mat2c<double>(
            rows, columns, &destination
        );
        if (!source.empty()) {
            std::memcpy(
                destination, source.data(),
                checked_mul(source.size(), sizeof(double), "published matrix")
            );
        }
        return result;
    }

    size_t base_index(int annotation, int row, int probe) const noexcept {
        return (static_cast<size_t>(annotation) * static_cast<size_t>(rows_)
                + static_cast<size_t>(row)) * static_cast<size_t>(probes_)
            + static_cast<size_t>(probe);
    }

    size_t source_index(
        int annotation, int coordinate, int row, int probe
    ) const noexcept {
        return (((static_cast<size_t>(annotation) * static_cast<size_t>(q_)
                  + static_cast<size_t>(coordinate))
                 * static_cast<size_t>(rows_)
                 + static_cast<size_t>(row))
                * static_cast<size_t>(probes_))
            + static_cast<size_t>(probe);
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

    void validate_inputs(
        nb_mat2c_ro<int64_t> pair_table,
        nb_mat2c_ro<int64_t> component_table,
        nb_vec1_ro<int64_t> product_offsets,
        nb_mat2c_ro<int64_t> product_terms
    );
    void initialize_outputs();
    void decode_block(int start, int stop, std::vector<double>& genotype);
    void read_packed_block(int start, int stop, MailmanPackedBlock& packed);
    void packed_source_product(
        const MailmanPackedBlock& packed,
        const std::vector<double>& rhs,
        int columns,
        std::vector<double>& output
    ) const;
    void packed_target_product(
        const MailmanPackedBlock& packed,
        const std::vector<double>& rhs,
        int columns,
        std::vector<double>& output
    ) const;
    void execute_pass1();
    void seal_sources_and_same_person();
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
    int probe_tile_width_ = 0;
    int same_person_sample_tile_width_ = 0;
    uint64_t max_workspace_bytes_ = 0;
    int threads_ = 1;
    int decode_threads_ = 1;
    bool retain_base_sources_ = false;
    bool dense_blas_hybrid_ = true;
    std::string qualification_fault_phase_ = "none";
    int qualification_fault_row_ = -1;
    int qualification_fault_column_ = -1;
    double qualification_fault_delta_ = 0.0;
    bool qualification_fault_injected_ = false;
    State state_ = State::constructed;
    mutable std::mutex mutex_;

    std::vector<double> basis_;
    std::vector<double> fixed_;
    std::vector<double> annotation_values_;
    std::vector<double> sqrt_annotation_values_;
    std::vector<double> annotation_masses_;
    std::vector<int64_t> pair_table_;
    std::vector<int64_t> component_table_;
    std::vector<int64_t> product_offsets_;
    std::vector<int64_t> product_terms_;
    std::vector<double> base_sources_;
    std::vector<double> contextual_sources_;
    std::vector<double> same_person_;
    std::vector<double> directional_;
    std::vector<double> directed_;
    std::vector<double> symmetric_;
    std::vector<double> genetic_gram_;
    std::vector<double> affine_mean_;
    std::vector<double> affine_inverse_scale_;

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
    double source_nn_wall_seconds_ = 0.0;
    double source_nn_cpu_seconds_ = 0.0;
    double projection_wall_seconds_ = 0.0;
    double projection_cpu_seconds_ = 0.0;
    double same_person_wall_seconds_ = 0.0;
    double same_person_cpu_seconds_ = 0.0;
    double target_tn_wall_seconds_ = 0.0;
    double target_tn_cpu_seconds_ = 0.0;
    double reduction_wall_seconds_ = 0.0;
    double reduction_cpu_seconds_ = 0.0;
    double independent_audit_wall_seconds_ = 0.0;
    double independent_audit_cpu_seconds_ = 0.0;
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
    sqrt_annotation_values_.resize(annotation_values_.size());
    std::transform(
        annotation_values_.begin(), annotation_values_.end(),
        sqrt_annotation_values_.begin(),
        [](double value) { return std::sqrt(value); }
    );
    base_sources_.assign(base_elements, 0.0);
    contextual_sources_.assign(source_elements, 0.0);
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
    int start, int stop, std::vector<double>& genotype
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
    const std::vector<double>& rhs,
    int columns,
    std::vector<double>& output
) const {
    const int width = packed.L;
    if (packed.N != rows_ || columns <= 0 ||
        rhs.size() != checked_mul(
            static_cast<size_t>(rows_), static_cast<size_t>(columns),
            "generalized packed target RHS"
        )) {
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
            const double* column = rhs.data()
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
    std::vector<double> genotype;
    MailmanPackedBlock packed;
    std::vector<double> probes;
    std::vector<double> rhs;
    std::vector<double> contribution;
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
    rhs.reserve(probes.capacity());
    contribution.reserve(checked_mul(
        static_cast<size_t>(rows_),
        static_cast<size_t>(source_probe_tile_width_),
        "generalized pass-1 output arena"
    ));
    for (int start = 0; start < variants_; start += variant_block_width_) {
        const int stop = std::min(variants_, start + variant_block_width_);
        const int width = stop - start;
        if (dense_blas_hybrid_) {
            decode_block(start, stop, genotype);
        } else {
            read_packed_block(start, stop, packed);
        }
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
            rhs.resize(probes.size());
            if (!dense_blas_hybrid_) {
                contribution.resize(checked_mul(
                    static_cast<size_t>(rows_),
                    static_cast<size_t>(probe_width),
                    "generalized source contribution"
                ));
            }
            for (int annotation = 0; annotation < annotations_count_;
                 ++annotation) {
                const auto rhs_wall = std::chrono::steady_clock::now();
                const double rhs_cpu = process_cpu_seconds_now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
                for (int probe = 0; probe < probe_width; ++probe) {
                    for (int variant = 0; variant < width; ++variant) {
                        rhs[
                            static_cast<size_t>(probe)
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
                        static_cast<size_t>(probe_width),
                        "generalized source logical output"
                    );
                    const size_t logical_bytes = checked_mul(
                        logical_elements, sizeof(double),
                        "generalized source logical output bytes"
                    );
                    const size_t capacity_bytes = checked_mul(
                        checked_mul(
                            static_cast<size_t>(rows_),
                            static_cast<size_t>(source_probe_tile_width_),
                            "generalized source output capacity"
                        ),
                        sizeof(double),
                        "generalized source output capacity bytes"
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
                    contribution_data = dense_contribution->data();
                    output_scope.register_output(
                        contribution_data, logical_bytes, CblasColMajor
                    );
                    GemmIntegrityResolution resolution;
                    const auto product_wall = std::chrono::steady_clock::now();
                    const double product_cpu = process_cpu_seconds_now();
                    try {
                        (void)dgemm_nn_partitioned_rows(
                            rows_, probe_width, width,
                            genotype.data(), rows_, rhs.data(), width,
                            contribution_data, rows_, threads_, 1.0, 0.0,
                            &resolution
                        );
                    } catch (const RetryableGemmInputMutation&) {
                        ++retry_count_;
                        dgemm_nn_tiled(
                            rows_, probe_width, width,
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
                        rows_, probe_width, width,
                        product_wall_seconds, product_cpu_seconds
                    );
                    record_integrity_resolution(resolution);
                    dense_contribution->verify_after_repair();
                    const auto audit_wall = std::chrono::steady_clock::now();
                    const double audit_cpu = process_cpu_seconds_now();
                    audit_nn_product(
                        genotype.data(), rows_, width, rhs.data(),
                        probe_width, contribution_data,
                        static_cast<int>(source_nn_calls_)
                    );
                    independent_audit_wall_seconds_ +=
                        generalized_elapsed_seconds(audit_wall);
                    independent_audit_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - audit_cpu
                    );
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
                        packed, rhs, probe_width, contribution
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
                        rows_, probe_width, width,
                        product_wall_seconds, product_cpu_seconds
                    );
                }
                ++source_nn_calls_;
                for (int row = 0; row < rows_; ++row) {
                    for (int probe = 0; probe < probe_width; ++probe) {
                        base_sources_[base_index(
                            annotation, row, probe_start + probe
                        )] += contribution_data[
                            static_cast<size_t>(probe)
                                * static_cast<size_t>(rows_)
                            + static_cast<size_t>(row)
                        ];
                    }
                }
            }
        }
    }
    reader_->check_files_unchanged();
    if (observed_variant_visits_ != variants_) {
        throw std::runtime_error(
            "Generalized pass 1 did not visit every variant exactly once"
        );
    }
}

void GeneralizedGxELDScoreDirectContext::seal_sources_and_same_person() {
    if (state_ != State::pass1_running || observed_passes_ != 1 ||
        observed_variant_visits_ != variants_) {
        throw std::runtime_error(
            "Generalized pass-1 barrier is not ready to seal"
        );
    }
    std::vector<double> weighted(checked_mul(
        static_cast<size_t>(rows_), static_cast<size_t>(probes_),
        "generalized contextual weighted source"
    ));
    std::vector<double> coefficients(checked_mul(
        static_cast<size_t>(fixed_rank_), static_cast<size_t>(probes_),
        "generalized projection coefficients"
    ));
    std::vector<double> leakage(coefficients.size());
    const auto projection_wall = std::chrono::steady_clock::now();
    const double projection_cpu = process_cpu_seconds_now();
    for (int annotation = 0; annotation < annotations_count_; ++annotation) {
        for (int coordinate = 0; coordinate < q_; ++coordinate) {
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
            long double projected_squared = 0.0L;
            long double leakage_squared = 0.0L;
            for (int probe = 0; probe < probes_; ++probe) {
                for (int row = 0; row < rows_; ++row) {
                    const double value = weighted[
                        static_cast<size_t>(probe)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(row)
                    ];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(
                            "Generalized contextual source is non-finite"
                        );
                    }
                    projected_squared +=
                        static_cast<long double>(value) * value;
                    contextual_sources_[source_index(
                        annotation, coordinate, row, probe
                    )] = value;
                }
            }
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

    const auto same_person_wall = std::chrono::steady_clock::now();
    const double same_person_cpu = process_cpu_seconds_now();
    std::vector<double> sample_accumulator(checked_mul(
        static_cast<size_t>(components_), static_cast<size_t>(rows_),
        "generalized same-person sample accumulator"
    ), 0.0);
    std::vector<double> same_probe(checked_mul(
        static_cast<size_t>(components_), static_cast<size_t>(components_),
        "generalized same-probe accumulator"
    ), 0.0);
    std::vector<double> left_values;
    std::vector<double> right_values;
    for (int probe_start = 0; probe_start < probes_;
         probe_start += source_probe_tile_width_) {
        const int probe_stop = std::min(
            probes_, probe_start + source_probe_tile_width_
        );
        const int probe_width = probe_stop - probe_start;
        for (int sample_start = 0; sample_start < rows_;
             sample_start += same_person_sample_tile_width_) {
            const int sample_stop = std::min(
                rows_, sample_start + same_person_sample_tile_width_
            );
            const int sample_width = sample_stop - sample_start;
            const size_t tile_elements = checked_mul(
                static_cast<size_t>(sample_width),
                static_cast<size_t>(probe_width),
                "generalized same-person tile"
            );
            left_values.resize(tile_elements);
            right_values.resize(tile_elements);
            for (int left_component = 0; left_component < components_;
                 ++left_component) {
                const int left_annotation = static_cast<int>(
                    component_table_[
                        2U * static_cast<size_t>(left_component)
                    ]
                );
                const int left_pair = static_cast<int>(
                    component_table_[
                        2U * static_cast<size_t>(left_component) + 1U
                    ]
                );
                const int left_q = pair_q(left_pair);
                const int left_r = pair_r(left_pair);
                const double left_scale = pair_factor(left_pair) /
                    annotation_masses_[
                        static_cast<size_t>(left_annotation)
                    ];
                for (int sample = 0; sample < sample_width; ++sample) {
                    double row_sum = 0.0;
                    for (int probe = 0; probe < probe_width; ++probe) {
                        const size_t tile_index =
                            static_cast<size_t>(sample)
                                * static_cast<size_t>(probe_width)
                            + static_cast<size_t>(probe);
                        const int global_sample = sample_start + sample;
                        const int global_probe = probe_start + probe;
                        const double value = left_scale
                            * contextual_sources_[source_index(
                                left_annotation, left_q, global_sample,
                                global_probe
                            )]
                            * contextual_sources_[source_index(
                                left_annotation, left_r, global_sample,
                                global_probe
                            )];
                        left_values[tile_index] = value;
                        row_sum += value;
                    }
                    sample_accumulator[
                        static_cast<size_t>(left_component)
                            * static_cast<size_t>(rows_)
                        + static_cast<size_t>(sample_start + sample)
                    ] += row_sum;
                }
                for (int right_component = left_component;
                     right_component < components_; ++right_component) {
                    const double* right = left_values.data();
                    if (right_component != left_component) {
                        const int right_annotation = static_cast<int>(
                            component_table_[
                                2U * static_cast<size_t>(right_component)
                            ]
                        );
                        const int right_pair = static_cast<int>(
                            component_table_[
                                2U * static_cast<size_t>(right_component) + 1U
                            ]
                        );
                        const int right_q = pair_q(right_pair);
                        const int right_r = pair_r(right_pair);
                        const double right_scale = pair_factor(right_pair) /
                            annotation_masses_[
                                static_cast<size_t>(right_annotation)
                            ];
                        for (int sample = 0; sample < sample_width; ++sample) {
                            for (int probe = 0; probe < probe_width; ++probe) {
                                const int global_sample = sample_start + sample;
                                const int global_probe = probe_start + probe;
                                right_values[
                                    static_cast<size_t>(sample)
                                        * static_cast<size_t>(probe_width)
                                    + static_cast<size_t>(probe)
                                ] = right_scale
                                    * contextual_sources_[source_index(
                                        right_annotation, right_q,
                                        global_sample, global_probe
                                    )]
                                    * contextual_sources_[source_index(
                                        right_annotation, right_r,
                                        global_sample, global_probe
                                    )];
                            }
                        }
                        right = right_values.data();
                    }
                    long double dot = 0.0L;
                    for (size_t index = 0; index < tile_elements; ++index) {
                        dot += static_cast<long double>(left_values[index])
                            * static_cast<long double>(right[index]);
                    }
                    const double value = static_cast<double>(dot);
                    same_probe[
                        static_cast<size_t>(left_component)
                            * static_cast<size_t>(components_)
                        + static_cast<size_t>(right_component)
                    ] += value;
                    if (right_component != left_component) {
                        same_probe[
                            static_cast<size_t>(right_component)
                                * static_cast<size_t>(components_)
                            + static_cast<size_t>(left_component)
                        ] += value;
                    }
                }
            }
        }
    }
    const double denominator = static_cast<double>(
        probes_ * (probes_ - 1)
    );
    for (int left = 0; left < components_; ++left) {
        for (int right = 0; right < components_; ++right) {
            long double dot = 0.0L;
            for (int row = 0; row < rows_; ++row) {
                dot += static_cast<long double>(sample_accumulator[
                    static_cast<size_t>(left) * static_cast<size_t>(rows_)
                    + static_cast<size_t>(row)
                ]) * static_cast<long double>(sample_accumulator[
                    static_cast<size_t>(right) * static_cast<size_t>(rows_)
                    + static_cast<size_t>(row)
                ]);
            }
            same_person_[
                static_cast<size_t>(left)
                    * static_cast<size_t>(components_)
                + static_cast<size_t>(right)
            ] = (static_cast<double>(dot) - same_probe[
                static_cast<size_t>(left)
                    * static_cast<size_t>(components_)
                + static_cast<size_t>(right)
            ]) / denominator;
        }
    }
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
    same_person_wall_seconds_ +=
        generalized_elapsed_seconds(same_person_wall);
    same_person_cpu_seconds_ += std::max(
        0.0, process_cpu_seconds_now() - same_person_cpu
    );
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
        std::vector<double>().swap(base_sources_);
    }
    state_ = State::sources_sealed;
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
    std::vector<double> genotype;
    MailmanPackedBlock packed;
    std::vector<double> rhs;
    std::vector<double> cross;
    std::vector<double> lrow;
    std::vector<double> lsum;
    std::unique_ptr<NativeGemmOutputAllocation> dense_cross;
    const int maximum_width = std::min(variants_, variant_block_width_);
    const int maximum_rhs_columns = probe_tile_width_ * family_count;
    genotype.reserve(checked_mul(
        static_cast<size_t>(rows_), static_cast<size_t>(maximum_width),
        "generalized pass-2 genotype arena"
    ));
    rhs.reserve(checked_mul(
        static_cast<size_t>(rows_),
        static_cast<size_t>(maximum_rhs_columns),
        "generalized pass-2 RHS arena"
    ));
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
            static_cast<size_t>(maximum_width),
            static_cast<size_t>(pairs_),
            "generalized pass-2 reduction arena"
        ),
        static_cast<size_t>(pairs_),
        "generalized pass-2 reduction arena"
    ));
    for (int start = 0; start < variants_; start += variant_block_width_) {
        const int stop = std::min(variants_, start + variant_block_width_);
        const int width = stop - start;
        if (dense_blas_hybrid_) {
            decode_block(start, stop, genotype);
        } else {
            read_packed_block(start, stop, packed);
        }
        lrow.assign(checked_mul(
            checked_mul(
                static_cast<size_t>(width), static_cast<size_t>(pairs_),
                "generalized target row block"
            ),
            static_cast<size_t>(components_),
            "generalized target row block"
        ), 0.0);
        for (int annotation = 0; annotation < annotations_count_;
             ++annotation) {
            lsum.assign(checked_mul(
                checked_mul(
                    static_cast<size_t>(width),
                    static_cast<size_t>(pairs_),
                    "generalized target pair products"
                ),
                static_cast<size_t>(pairs_),
                "generalized target pair products"
            ), 0.0);
            for (int probe_start = 0; probe_start < probes_;
                 probe_start += probe_tile_width_) {
                const int probe_stop = std::min(
                    probes_, probe_start + probe_tile_width_
                );
                const int probe_width = probe_stop - probe_start;
                const int rhs_columns = probe_width * family_count;
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
                    const int local_probe = column / family_count;
                    const int family = column % family_count;
                    const int target_coordinate = family / q_;
                    const int source_coordinate = family % q_;
                    const int global_probe = probe_start + local_probe;
                    for (int row = 0; row < rows_; ++row) {
                        rhs[
                            static_cast<size_t>(column)
                                * static_cast<size_t>(rows_)
                            + static_cast<size_t>(row)
                        ] = basis_[
                            static_cast<size_t>(target_coordinate)
                                * static_cast<size_t>(rows_)
                            + static_cast<size_t>(row)
                        ] * contextual_sources_[source_index(
                            annotation, source_coordinate, row, global_probe
                        )];
                    }
                }
                rhs_wall_seconds_ += generalized_elapsed_seconds(rhs_wall);
                rhs_cpu_seconds_ += std::max(
                    0.0, process_cpu_seconds_now() - rhs_cpu
                );
                double* cross_data = cross.data();
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
                            rhs.data(), rows_, cross_data, width, threads_,
                            1.0, 0.0, nullptr, false, &resolution
                        );
                    } catch (const RetryableGemmInputMutation&) {
                        ++retry_count_;
                        decode_block(start, stop, genotype);
                        dgemm_tn_tiled(
                            width, rhs_columns, rows_, genotype.data(), rows_,
                            rhs.data(), rows_, cross_data, width, threads_
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
                        genotype.data(), width, rows_, rhs.data(),
                        rhs_columns, cross_data,
                        static_cast<int>(target_tn_calls_)
                    );
                    independent_audit_wall_seconds_ +=
                        generalized_elapsed_seconds(audit_wall);
                    independent_audit_cpu_seconds_ += std::max(
                        0.0, process_cpu_seconds_now() - audit_cpu
                    );
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
                        packed, rhs, rhs_columns, cross
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
                const auto reduction_wall = std::chrono::steady_clock::now();
                const double reduction_cpu = process_cpu_seconds_now();
                const double inverse_rank = 1.0 /
                    static_cast<double>(residual_rank_);
                const size_t cross_elements = checked_mul(
                    static_cast<size_t>(width),
                    static_cast<size_t>(rhs_columns),
                    "generalized target cross scaling"
                );
                for (size_t index = 0; index < cross_elements; ++index) {
                    cross_data[index] *= inverse_rank;
                }
                for (int target_pair = 0; target_pair < pairs_;
                     ++target_pair) {
                    for (int source_pair = 0; source_pair < pairs_;
                         ++source_pair) {
                        const int cell = target_pair * pairs_ + source_pair;
                        const int64_t term_start = product_offsets_[
                            static_cast<size_t>(cell)
                        ];
                        const int64_t term_stop = product_offsets_[
                            static_cast<size_t>(cell + 1)
                        ];
                        for (int variant = 0; variant < width; ++variant) {
                            double total = 0.0;
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
                                for (int probe = 0; probe < probe_width;
                                     ++probe) {
                                    const int first_column =
                                        probe * family_count
                                        + first_target * q_ + first_source;
                                    const int second_column =
                                        probe * family_count
                                        + second_target * q_ + second_source;
                                    total += cross_data[
                                        static_cast<size_t>(first_column)
                                            * static_cast<size_t>(width)
                                        + static_cast<size_t>(variant)
                                    ] * cross_data[
                                        static_cast<size_t>(second_column)
                                            * static_cast<size_t>(width)
                                        + static_cast<size_t>(variant)
                                    ];
                                }
                            }
                            lsum[
                                (static_cast<size_t>(variant)
                                    * static_cast<size_t>(pairs_)
                                 + static_cast<size_t>(target_pair))
                                    * static_cast<size_t>(pairs_)
                                + static_cast<size_t>(source_pair)
                            ] += total;
                        }
                    }
                }
                reduction_wall_seconds_ +=
                    generalized_elapsed_seconds(reduction_wall);
                reduction_cpu_seconds_ += std::max(
                    0.0, process_cpu_seconds_now() - reduction_cpu
                );
            }
            const auto block_reduction_wall =
                std::chrono::steady_clock::now();
            const double block_reduction_cpu = process_cpu_seconds_now();
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
                            (static_cast<size_t>(variant)
                                * static_cast<size_t>(pairs_)
                             + static_cast<size_t>(target_pair))
                                * static_cast<size_t>(pairs_)
                            + static_cast<size_t>(source_pair)
                        ] / static_cast<double>(probes_);
                    }
                }
            }
            reduction_wall_seconds_ +=
                generalized_elapsed_seconds(block_reduction_wall);
            reduction_cpu_seconds_ += std::max(
                0.0, process_cpu_seconds_now() - block_reduction_cpu
            );
        }

        const auto output_reduction_wall = std::chrono::steady_clock::now();
        const double output_reduction_cpu = process_cpu_seconds_now();
        for (int variant = 0; variant < width; ++variant) {
            const int global_variant = start + variant;
            for (int target_pair = 0; target_pair < pairs_; ++target_pair) {
                for (int component = 0; component < components_; ++component) {
                    const double value = lrow[
                        (static_cast<size_t>(variant)
                            * static_cast<size_t>(pairs_)
                         + static_cast<size_t>(target_pair))
                            * static_cast<size_t>(components_)
                        + static_cast<size_t>(component)
                    ];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(
                            "Generalized directional target row is non-finite"
                        );
                    }
                    directional_[directional_index(
                        global_variant, target_pair, component
                    )] = value;
                }
            }
            for (int annotation = 0; annotation < annotations_count_;
                 ++annotation) {
                const double weight = annotation_values_[
                    static_cast<size_t>(global_variant)
                        * static_cast<size_t>(annotations_count_)
                    + static_cast<size_t>(annotation)
                ];
                for (int target_pair = 0; target_pair < pairs_;
                     ++target_pair) {
                    const int left_component = annotation * pairs_
                        + target_pair;
                    for (int right_component = 0;
                         right_component < components_; ++right_component) {
                        const double contribution = weight * lrow[
                            (static_cast<size_t>(variant)
                                * static_cast<size_t>(pairs_)
                             + static_cast<size_t>(target_pair))
                                * static_cast<size_t>(components_)
                            + static_cast<size_t>(right_component)
                        ];
                        directed_[
                            static_cast<size_t>(left_component)
                                * static_cast<size_t>(components_)
                            + static_cast<size_t>(right_component)
                        ] += contribution;
                    }
                }
            }
        }
        reduction_wall_seconds_ +=
            generalized_elapsed_seconds(output_reduction_wall);
        reduction_cpu_seconds_ += std::max(
            0.0, process_cpu_seconds_now() - output_reduction_cpu
        );
    }
    reader_->check_files_unchanged();
    if (observed_variant_visits_ != 2LL * variants_) {
        throw std::runtime_error(
            "Generalized pass 2 did not visit every variant exactly once"
        );
    }
}

void GeneralizedGxELDScoreDirectContext::finalize_scientific_outputs() {
    if (state_ != State::pass2_running || observed_passes_ != 2 ||
        observed_variant_visits_ != 2LL * variants_) {
        throw std::runtime_error(
            "Generalized scientific finalization lacks two complete passes"
        );
    }
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
            &same_person_
         }) {
        for (double value : *values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Generalized finalized output is non-finite"
                );
            }
        }
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

    nb::dict result;
    const auto publication_wall = std::chrono::steady_clock::now();
    const double publication_cpu = process_cpu_seconds_now();
    result["shapes"] = std::move(shapes);
    result["base_sources"] = retain_base_sources_
        ? nb::cast(copy_vector(base_sources_))
        : nb::none();
    result["contextual_sources"] = copy_vector(contextual_sources_);
    result["same_person"] = copy_matrix(
        same_person_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["directional_ldscores"] = copy_vector(directional_);
    result["directed_numerator"] = copy_matrix(
        directed_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["symmetric_numerator"] = copy_matrix(
        symmetric_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["genetic_gram"] = copy_matrix(
        genetic_gram_, static_cast<size_t>(components_),
        static_cast<size_t>(components_)
    );
    result["annotation_masses"] = copy_vector(annotation_masses_);
    // The downstream trait pass must use the exact affine scale sealed by
    // this reference run.  Publish these small per-variant vectors with the
    // ephemeral native result so callers do not need an extra genotype pass.
    // They are deliberately not part of the compact reference artifact.
    result["affine_mean"] = copy_vector(affine_mean_);
    result["affine_inverse_scale"] =
        copy_vector(affine_inverse_scale_);
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
    subphase_wall["source_nn"] = source_nn_wall_seconds_;
    subphase_wall["projection"] = projection_wall_seconds_;
    subphase_wall["same_person"] = same_person_wall_seconds_;
    subphase_wall["target_tn"] = target_tn_wall_seconds_;
    subphase_wall["row_product_reduction"] = reduction_wall_seconds_;
    subphase_wall["independent_audits"] = independent_audit_wall_seconds_;
    subphase_wall["publication"] = publication_wall_seconds_;
    nb::dict subphase_cpu;
    subphase_cpu["decode"] = decode_cpu_seconds_;
    subphase_cpu["rhs_formation"] = rhs_cpu_seconds_;
    subphase_cpu["source_nn"] = source_nn_cpu_seconds_;
    subphase_cpu["projection"] = projection_cpu_seconds_;
    subphase_cpu["same_person"] = same_person_cpu_seconds_;
    subphase_cpu["target_tn"] = target_tn_cpu_seconds_;
    subphase_cpu["row_product_reduction"] = reduction_cpu_seconds_;
    subphase_cpu["independent_audits"] = independent_audit_cpu_seconds_;
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
    const uint64_t output_bytes = checked_mul(
        static_cast<uint64_t>(
            contextual_sources_.size() + same_person_.size()
            + directional_.size() + directed_.size() + symmetric_.size()
            + genetic_gram_.size()
            + annotation_masses_.size()
            + affine_mean_.size() + affine_inverse_scale_.size()
            + (retain_base_sources_ ? base_sources_.size() : 0U)
        ),
        static_cast<uint64_t>(sizeof(double)),
        "generalized published output bytes"
    );
    nb::dict resident_bytes;
    resident_bytes["base_sources"] = checked_mul(
        static_cast<uint64_t>(base_sources_.size()), 8U,
        "generalized resident base bytes"
    );
    resident_bytes["contextual_sources"] = checked_mul(
        static_cast<uint64_t>(contextual_sources_.size()), 8U,
        "generalized resident contextual bytes"
    );
    resident_bytes["directional_panel"] = checked_mul(
        static_cast<uint64_t>(directional_.size()), 8U,
        "generalized resident directional bytes"
    );
    resident_bytes["genotype_scale_identity"] = checked_mul(
        static_cast<uint64_t>(
            affine_mean_.size() + affine_inverse_scale_.size()
        ),
        8U, "generalized resident genotype scale bytes"
    );
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
            static_cast<uint64_t>(rows_),
            static_cast<uint64_t>(probe_tile_width_ * q_ * q_),
            "generalized resident RHS"
        ),
        8U, "generalized resident RHS bytes"
    );
    resident_bytes["maximum_source_output"] = checked_mul(
        checked_mul(
            static_cast<uint64_t>(rows_),
            static_cast<uint64_t>(source_probe_tile_width_),
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
    telemetry["target_probe_tile_width"] = probe_tile_width_;
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
    std::vector<double>().swap(sqrt_annotation_values_);
    std::vector<double>().swap(base_sources_);
    std::vector<double>().swap(contextual_sources_);
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
