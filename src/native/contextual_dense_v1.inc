// Contextual native API V1: isolated dense differential executor.
//
// This file is included immediately after gxeldcore.cpp's anonymous namespace.
// Same-translation-unit lookup permits reuse of the established protected GEMM
// dispatchers without exporting any legacy scientific implementation. It
// intentionally accepts only already-scaled dense matrices and publishes
// compact sufficient statistics.

namespace summit::context_v1 {

enum class ContextualDenseSemanticV1 : std::size_t {
    sample_probe_projection_tn = 0,
    sample_probe_projection_nn,
    source_tn,
    full_target_nn,
    action_projection_tn,
    action_projection_nn,
    action_gram_tn,
    group_target_nn,
    group_cross_gram_tn,
    direct_grouped_tn,
    same_person_target_nn,
    same_person_projection_tn,
    same_person_projection_nn,
    same_person_gram_tn,
    trait_score_tn,
    trait_feature_projection_tn,
    trait_feature_projection_nn,
    count
};

constexpr std::size_t kContextualDenseSemanticCountV1 =
    static_cast<std::size_t>(ContextualDenseSemanticV1::count);

constexpr std::array<const char*, kContextualDenseSemanticCountV1>
    kContextualDenseSemanticNamesV1 = {{
        "sample_probe_projection_tn",
        "sample_probe_projection_nn",
        "source_tn",
        "full_target_nn",
        "action_projection_tn",
        "action_projection_nn",
        "action_gram_tn",
        "group_target_nn",
        "group_cross_gram_tn",
        "direct_grouped_tn",
        "same_person_target_nn",
        "same_person_projection_tn",
        "same_person_projection_nn",
        "same_person_gram_tn",
        "trait_score_tn",
        "trait_feature_projection_tn",
        "trait_feature_projection_nn"
    }};

using contextual_numpy_arr3_f64_v1 =
    nb::ndarray<double, nb::numpy, nb::ndim<3>, nb::c_contig,
                nb::device::cpu>;

template <typename T>
nb_numpy_vec1<T> contextual_copy_vec_v1(const std::vector<T>& values) {
    T* out = nullptr;
    auto result = make_owned_numpy_vec1<T>(values.size(), &out);
    if (!values.empty())
        std::copy(values.begin(), values.end(), out);
    return result;
}

nb_numpy_mat2c<double> contextual_copy_mat2_v1(
    const std::vector<double>& values, std::size_t rows, std::size_t columns
) {
    if (values.size() != rows * columns)
        throw std::runtime_error("Contextual dense publication shape mismatch");
    double* out = nullptr;
    auto result = make_owned_numpy_mat2c<double>(rows, columns, &out);
    if (!values.empty())
        std::copy(values.begin(), values.end(), out);
    return result;
}

contextual_numpy_arr3_f64_v1 contextual_copy_arr3_v1(
    const std::vector<double>& values,
    std::size_t first, std::size_t second, std::size_t third
) {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
        throw std::runtime_error("Contextual dense publication shape overflow");
    const std::size_t first_second = first * second;
    if (first_second != 0 &&
        third > std::numeric_limits<std::size_t>::max() / first_second)
        throw std::runtime_error("Contextual dense publication shape overflow");
    const std::size_t count = first_second * third;
    if (values.size() != count)
        throw std::runtime_error("Contextual dense publication shape mismatch");
    void* allocation = nullptr;
    if (count > 0 &&
        posix_memalign(&allocation, 64, count * sizeof(double)) != 0)
        throw std::bad_alloc();
    auto* out = static_cast<double*>(allocation);
    if (count > 0)
        std::copy(values.begin(), values.end(), out);
    nb::capsule owner(allocation, [](void* pointer) noexcept {
        std::free(pointer);
    });
    return contextual_numpy_arr3_f64_v1(
        out, { first, second, third }, owner
    );
}

std::uint64_t contextual_checked_u64_v1(
    __uint128_t value, const char* label
) {
    if (value > std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error(
            std::string("ContextualBlockExecutorV1 checked overflow: ") + label
        );
    return static_cast<std::uint64_t>(value);
}

std::uint64_t contextual_add_u64_v1(
    std::uint64_t left, std::uint64_t right, const char* label
) {
    return contextual_checked_u64_v1(
        static_cast<__uint128_t>(left) + right, label
    );
}

std::uint64_t contextual_mul_u64_v1(
    std::uint64_t left, std::uint64_t right, const char* label
) {
    return contextual_checked_u64_v1(
        static_cast<__uint128_t>(left) * right, label
    );
}

std::uint64_t contextual_ceil_div_u64_v1(
    std::uint64_t value, std::uint64_t divisor, const char* label
) {
    if (divisor == 0)
        throw std::runtime_error(
            std::string("ContextualBlockExecutorV1 validation: zero ") + label
        );
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

struct ContextualDensePreflightNumbersV1 {
    std::uint64_t required_workspace_bytes = 0;
    std::uint64_t retry_reserve_bytes = 0;
    std::uint64_t witness_reserve_bytes = 0;
    std::uint64_t fallback_reserve_bytes = 0;
    std::uint64_t telemetry_bytes = 0;
    std::uint64_t required_telemetry_capacity = 0;
    std::uint64_t total_protected_calls = 0;
    std::uint64_t worst_case_attempts = 0;
    std::uint64_t variant_blocks = 0;
    std::uint64_t sample_probe_tiles = 0;
    std::uint64_t variant_probe_tiles = 0;
    std::uint64_t action_tiles = 0;
    std::uint64_t annotation_tiles = 0;
    std::uint64_t context_tiles = 0;
    std::uint64_t maximum_protected_output_elements = 0;
    std::array<std::uint64_t, kContextualDenseSemanticCountV1> calls{};
};

ContextualDensePreflightNumbersV1 contextual_dense_preflight_numbers_v1(
    std::uint64_t n, std::uint64_t m, std::uint64_t fixed_rank,
    std::uint64_t q, std::uint64_t k, std::uint64_t groups,
    std::uint64_t sample_probes, std::uint64_t variant_probes,
    std::uint64_t residual_components,
    std::uint64_t variant_block, std::uint64_t sample_probe_tile,
    std::uint64_t variant_probe_tile, std::uint64_t action_tile,
    std::uint64_t annotation_tile, std::uint64_t context_tile,
    std::uint64_t trait_feature_tile,
    bool fault_enabled
) {
    if (n == 0 || m == 0 || fixed_rank == 0 || q == 0 || k == 0 ||
        groups == 0 || sample_probes == 0 || variant_probes < 2 ||
        residual_components == 0) {
        throw std::runtime_error(
            "ContextualBlockExecutorV1 validation: dimensions must be positive "
            "and variant probe count must be at least two"
        );
    }
    if (q > 4)
        throw std::runtime_error(
            "ContextualBlockExecutorV1 validation: Q must be in [1,4]"
        );
    if (fixed_rank >= n)
        throw std::runtime_error(
            "ContextualBlockExecutorV1 validation: fixed rank must be below N"
        );
    variant_block = variant_block == 0 ? m : std::min(variant_block, m);
    sample_probe_tile = sample_probe_tile == 0
        ? sample_probes : std::min(sample_probe_tile, sample_probes);
    variant_probe_tile = variant_probe_tile == 0
        ? variant_probes : std::min(variant_probe_tile, variant_probes);
    const std::uint64_t pairs = contextual_checked_u64_v1(
        static_cast<__uint128_t>(q) * (q + 1) / 2, "pair count"
    );
    const std::uint64_t components = contextual_mul_u64_v1(
        k, pairs, "component count"
    );
    action_tile = action_tile == 0
        ? components : std::min(action_tile, components);
    annotation_tile = annotation_tile == 0
        ? k : std::min(annotation_tile, k);
    context_tile = context_tile == 0
        ? q : std::min(context_tile, q);
    const std::uint64_t maximum_trait_features = contextual_mul_u64_v1(
        q, variant_block, "trait feature tile extent"
    );
    trait_feature_tile = trait_feature_tile == 0
        ? maximum_trait_features
        : std::min(trait_feature_tile, maximum_trait_features);

    ContextualDensePreflightNumbersV1 out;
    out.variant_blocks = contextual_ceil_div_u64_v1(
        m, variant_block, "variant block"
    );
    out.sample_probe_tiles = contextual_ceil_div_u64_v1(
        sample_probes, sample_probe_tile, "sample-probe tile"
    );
    out.variant_probe_tiles = contextual_ceil_div_u64_v1(
        variant_probes, variant_probe_tile, "variant-probe tile"
    );
    out.action_tiles = contextual_ceil_div_u64_v1(
        components, action_tile, "action tile"
    );
    out.annotation_tiles = contextual_ceil_div_u64_v1(
        k, annotation_tile, "annotation tile"
    );
    out.context_tiles = contextual_ceil_div_u64_v1(
        q, context_tile, "context tile"
    );

    auto& calls = out.calls;
    calls[0] = out.sample_probe_tiles;
    calls[1] = out.sample_probe_tiles;
    calls[2] = contextual_mul_u64_v1(
        contextual_mul_u64_v1(
            out.sample_probe_tiles, out.variant_blocks, "source call count"
        ), out.context_tiles, "source call count"
    );
    calls[3] = contextual_mul_u64_v1(
        contextual_mul_u64_v1(
            out.sample_probe_tiles, out.annotation_tiles,
            "full target call count"
        ), out.context_tiles, "full target call count"
    );
    calls[4] = contextual_mul_u64_v1(
        out.sample_probe_tiles, out.action_tiles,
        "action projection call count"
    );
    calls[5] = calls[4];
    calls[6] = 1;
    calls[7] = contextual_mul_u64_v1(
        contextual_mul_u64_v1(
            contextual_mul_u64_v1(
                groups, out.sample_probe_tiles, "group target call count"
            ), out.annotation_tiles, "group target call count"
        ), out.context_tiles, "group target call count"
    );
    calls[8] = contextual_mul_u64_v1(
        groups, out.sample_probe_tiles, "group cross-Gram call count"
    );
    calls[9] = contextual_mul_u64_v1(
        contextual_mul_u64_v1(
            contextual_mul_u64_v1(2, out.sample_probe_tiles,
                                  "direct grouped call count"),
            q, "direct grouped call count"
        ), out.action_tiles, "direct grouped call count"
    );
    calls[10] = out.variant_probe_tiles;
    calls[11] = out.variant_probe_tiles;
    calls[12] = out.variant_probe_tiles;
    calls[13] = contextual_add_u64_v1(
        out.variant_probe_tiles, 1, "same-person call count"
    );
    calls[14] = out.variant_blocks;
    const std::uint64_t full_variant_blocks = m / variant_block;
    const std::uint64_t remainder_variants = m % variant_block;
    const std::uint64_t full_trait_projection_tiles =
        contextual_ceil_div_u64_v1(
            maximum_trait_features, trait_feature_tile,
            "trait feature projection tile"
        );
    std::uint64_t trait_projection_calls = contextual_mul_u64_v1(
        full_variant_blocks, full_trait_projection_tiles,
        "trait projection call count"
    );
    if (remainder_variants != 0) {
        trait_projection_calls = contextual_add_u64_v1(
            trait_projection_calls,
            contextual_ceil_div_u64_v1(
                contextual_mul_u64_v1(q, remainder_variants,
                                      "trait remainder features"),
                trait_feature_tile, "trait remainder projection tile"
            ), "trait projection call count"
        );
    }
    calls[15] = trait_projection_calls;
    calls[16] = calls[15];
    for (std::uint64_t value : calls)
        out.total_protected_calls = contextual_add_u64_v1(
            out.total_protected_calls, value, "total protected calls"
        );
    out.worst_case_attempts = contextual_mul_u64_v1(
        out.total_protected_calls, 3, "worst-case protected attempts"
    );

    const std::uint64_t n_q_bt = contextual_mul_u64_v1(
        contextual_mul_u64_v1(n, q, "sample projection panel"),
        sample_probe_tile, "sample projection panel"
    );
    const std::uint64_t source_panel = contextual_mul_u64_v1(
        contextual_mul_u64_v1(m, q, "source panel"),
        sample_probes, "source panel"
    );
    const std::uint64_t raw_actions = contextual_mul_u64_v1(
        contextual_mul_u64_v1(n, components, "raw action panel"),
        sample_probes, "raw action panel"
    );
    const std::uint64_t compact_c2 = contextual_mul_u64_v1(
        components, components, "compact component square"
    );
    const std::uint64_t compact_group_c2 = contextual_mul_u64_v1(
        groups, compact_c2, "compact grouped component square"
    );
    const std::uint64_t same_persistent = contextual_mul_u64_v1(
        n, components, "same-person persistent state"
    );
    const std::uint64_t trait_group = contextual_mul_u64_v1(
        contextual_mul_u64_v1(groups, components, "trait grouped state"),
        contextual_add_u64_v1(residual_components, 2,
                              "trait grouped state"),
        "trait grouped state"
    );

    std::array<std::uint64_t, 12> maxima = {{
        contextual_mul_u64_v1(
            variant_block,
            contextual_mul_u64_v1(q, sample_probe_tile,
                                  "source protected output"),
            "source protected output"),
        contextual_mul_u64_v1(
            n,
            contextual_mul_u64_v1(
                contextual_mul_u64_v1(annotation_tile, context_tile,
                                      "target protected output"),
                sample_probe_tile, "target protected output"),
            "target protected output"),
        contextual_mul_u64_v1(
            n, contextual_mul_u64_v1(action_tile, sample_probe_tile,
                                     "action protected output"),
            "action protected output"),
        compact_c2,
        contextual_mul_u64_v1(
            m, contextual_mul_u64_v1(action_tile, sample_probe_tile,
                                     "direct protected output"),
            "direct protected output"),
        contextual_mul_u64_v1(
            n, contextual_mul_u64_v1(k, variant_probe_tile,
                                     "same-person target output"),
            "same-person target output"),
        contextual_mul_u64_v1(
            n,
            contextual_mul_u64_v1(
                contextual_mul_u64_v1(k, q,
                                      "same-person projection output"),
                variant_probe_tile, "same-person projection output"),
            "same-person projection output"),
        contextual_mul_u64_v1(
            fixed_rank,
            contextual_mul_u64_v1(action_tile, sample_probe_tile,
                                  "projection coefficient output"),
            "projection coefficient output"),
        contextual_mul_u64_v1(variant_block, q,
                              "trait score protected output"),
        contextual_mul_u64_v1(
            n, trait_feature_tile,
            "trait feature protected output"),
        n_q_bt,
        raw_actions
    }};
    out.maximum_protected_output_elements =
        *std::max_element(maxima.begin(), maxima.end());
    const std::uint64_t retry_elements = contextual_add_u64_v1(
        out.maximum_protected_output_elements, 16,
        "integrity retry canary reserve"
    );
    out.retry_reserve_bytes = contextual_mul_u64_v1(
        retry_elements, sizeof(double), "integrity retry bytes"
    );
    out.witness_reserve_bytes = out.retry_reserve_bytes;
    out.fallback_reserve_bytes = out.retry_reserve_bytes;

    std::uint64_t science_elements = 0;
    for (std::uint64_t value : {
            source_panel, raw_actions,
            contextual_mul_u64_v1(4, compact_c2,
                                  "reference compact outputs"),
            contextual_mul_u64_v1(3, compact_group_c2,
                                  "grouped reference outputs"),
            same_persistent, trait_group, n_q_bt
        }) {
        science_elements = contextual_add_u64_v1(
            science_elements, value, "scientific workspace elements"
        );
    }
    const std::uint64_t science_bytes = contextual_mul_u64_v1(
        science_elements, sizeof(double), "scientific workspace bytes"
    );
    // Each protected call can emit at most repair + retry-mismatch + trusted
    // fallback.  Eight phase transitions plus the explicit scratch-release
    // event are fixed; a targeted run adds one injection event.
    out.required_telemetry_capacity = contextual_add_u64_v1(
        out.worst_case_attempts, fault_enabled ? 10 : 9,
        "telemetry event capacity"
    );
    // Fixed-size native event records are conservatively admitted at 128 B.
    out.telemetry_bytes = contextual_mul_u64_v1(
        out.required_telemetry_capacity, 128, "telemetry bytes"
    );
    out.required_workspace_bytes = science_bytes;
    out.required_workspace_bytes = contextual_add_u64_v1(
        out.required_workspace_bytes, out.retry_reserve_bytes,
        "workspace retry reserve"
    );
    out.required_workspace_bytes = contextual_add_u64_v1(
        out.required_workspace_bytes, out.witness_reserve_bytes,
        "workspace independent witness reserve"
    );
    out.required_workspace_bytes = contextual_add_u64_v1(
        out.required_workspace_bytes, out.fallback_reserve_bytes,
        "workspace fallback reserve"
    );
    out.required_workspace_bytes = contextual_add_u64_v1(
        out.required_workspace_bytes, out.telemetry_bytes,
        "workspace telemetry reserve"
    );
    return out;
}

nb::dict contextual_dense_preflight_dict_v1(
    const ContextualDensePreflightNumbersV1& estimate
) {
    nb::dict result;
    result["checked_arithmetic_complete"] = true;
    result["required_workspace_bytes"] = estimate.required_workspace_bytes;
    result["retry_reserve_bytes"] = estimate.retry_reserve_bytes;
    result["witness_reserve_bytes"] = estimate.witness_reserve_bytes;
    result["fallback_reserve_bytes"] = estimate.fallback_reserve_bytes;
    result["telemetry_bytes"] = estimate.telemetry_bytes;
    result["required_telemetry_capacity"] =
        estimate.required_telemetry_capacity;
    result["total_protected_calls"] = estimate.total_protected_calls;
    result["worst_case_protected_attempts"] = estimate.worst_case_attempts;
    result["maximum_protected_output_elements"] =
        estimate.maximum_protected_output_elements;
    result["variant_blocks"] = estimate.variant_blocks;
    result["sample_probe_tiles"] = estimate.sample_probe_tiles;
    result["variant_probe_tiles"] = estimate.variant_probe_tiles;
    result["action_tiles"] = estimate.action_tiles;
    result["annotation_tiles"] = estimate.annotation_tiles;
    result["context_tiles"] = estimate.context_tiles;
    nb::dict calls;
    for (std::size_t index = 0; index < kContextualDenseSemanticCountV1;
         ++index) {
        calls[kContextualDenseSemanticNamesV1[index]] = estimate.calls[index];
    }
    result["semantic_call_ledger"] = calls;
    return result;
}

struct ContextualDenseEventV1 {
    const char* event_class;
    const char* phase;
    const char* operation;
    const char* resolution;
};

class ContextualBlockExecutorV1 final {
public:
    ContextualBlockExecutorV1(
        nb_mat2f_ro<double> genotype,
        nb_mat2f_ro<double> fixed_basis,
        nb_mat2f_ro<double> context_basis,
        nb_mat2f_ro<double> annotation_weights,
        nb_vec1_ro<std::int64_t> group_index,
        nb_mat2f_ro<double> sample_probes,
        nb_mat2f_ro<double> variant_probes,
        nb_vec1_ro<double> phenotype_raw,
        nb_mat2f_ro<double> residual_basis,
        std::vector<std::string> annotation_names,
        std::vector<std::string> group_names,
        std::string annotation_mode,
        std::uint64_t variant_block,
        std::uint64_t sample_probe_tile,
        std::uint64_t variant_probe_tile,
        std::uint64_t action_tile,
        std::uint64_t annotation_tile,
        std::uint64_t context_tile,
        std::uint64_t group_tile,
        std::uint64_t trait_feature_tile,
        int threads,
        std::uint64_t workspace_cap_bytes,
        std::uint64_t telemetry_capacity,
        std::string fault_operation,
        std::string fault_mode,
        std::uint64_t fault_occurrence
    ) :
        n_(genotype.shape(0)), m_(genotype.shape(1)),
        rank_(fixed_basis.shape(1)), q_(context_basis.shape(1)),
        k_(annotation_weights.shape(1)), groups_(group_names.size()),
        bt_(sample_probes.shape(1)), bd_(variant_probes.shape(1)),
        h_(residual_basis.shape(1)),
        annotation_names_(std::move(annotation_names)),
        group_names_(std::move(group_names)),
        annotation_mode_(std::move(annotation_mode)),
        threads_(threads), fault_operation_(std::move(fault_operation)),
        fault_mode_(std::move(fault_mode)), fault_occurrence_(fault_occurrence),
        workspace_cap_bytes_(workspace_cap_bytes),
        telemetry_capacity_(telemetry_capacity)
    {
        validate_shapes(
            fixed_basis, context_basis, annotation_weights, group_index,
            sample_probes, variant_probes, phenotype_raw, residual_basis
        );
        validate_policy();
        build_maps();
        variant_block_ = resolve_tile(variant_block, m_, "variant_block");
        sample_probe_tile_ = resolve_tile(
            sample_probe_tile, bt_, "sample_probe_tile"
        );
        variant_probe_tile_ = resolve_tile(
            variant_probe_tile, bd_, "variant_probe_tile"
        );
        action_tile_ = resolve_tile(action_tile, components_, "action_tile");
        annotation_tile_ = resolve_tile(
            annotation_tile, k_, "annotation_tile"
        );
        context_tile_ = resolve_tile(context_tile, q_, "context_tile");
        group_tile_ = resolve_tile(group_tile, groups_, "group_tile");
        validate_blas_and_layout_dimensions();
        const std::size_t admitted_trait_features = checked_product_size(
            {q_, variant_block_}, "trait feature tile extent"
        );
        trait_feature_tile_ = resolve_tile(
            trait_feature_tile, admitted_trait_features, "trait_feature_tile"
        );

        estimate_ = contextual_dense_preflight_numbers_v1(
            n_, m_, rank_, q_, k_, groups_, bt_, bd_, h_,
            variant_block_, sample_probe_tile_, variant_probe_tile_,
            action_tile_, annotation_tile_, context_tile_, trait_feature_tile_,
            !fault_operation_.empty() && fault_mode_ != "none"
        );
        if (estimate_.required_workspace_bytes > workspace_cap_bytes_) {
            throw std::runtime_error(
                "ContextualBlockExecutorV1 admission: workspace cap is below "
                "required_workspace_bytes"
            );
        }
        if (estimate_.required_telemetry_capacity > telemetry_capacity_) {
            throw std::runtime_error(
                "ContextualBlockExecutorV1 admission: telemetry capacity is "
                "below required_telemetry_capacity"
            );
        }

        copy_array(genotype.data(), checked_product_size({n_, m_}, "genotype"), genotype_);
        copy_array(fixed_basis.data(), checked_product_size({n_, rank_}, "fixed basis"), fixed_basis_);
        copy_array(context_basis.data(), checked_product_size({n_, q_}, "context basis"), context_basis_);
        copy_array(annotation_weights.data(), checked_product_size({m_, k_}, "annotation weights"), annotation_weights_);
        copy_array(group_index.data(), m_, group_index_);
        copy_array(sample_probes.data(), checked_product_size({n_, bt_}, "sample probes"), sample_probes_);
        copy_array(variant_probes.data(), checked_product_size({m_, bd_}, "variant probes"), variant_probes_);
        copy_array(phenotype_raw.data(), n_, phenotype_raw_);
        copy_array(residual_basis.data(), checked_product_size({n_, h_}, "residual basis"), residual_basis_);

        validate_owned_values();
        seal_annotation_representation();
        compute_masses();
        input_fingerprint_ = fingerprint_inputs();
        const std::size_t protected_elements = checked_size(
            estimate_.maximum_protected_output_elements + 16,
            "protected scratch elements"
        );
        retry_buffer_.assign(protected_elements, kCanaryV1);
        witness_buffer_.assign(protected_elements, kCanaryV1);
        fallback_buffer_.assign(protected_elements, kCanaryV1);
        events_.reserve(checked_size(
            estimate_.required_telemetry_capacity, "telemetry capacity"
        ));
        state_ = "admitted";
    }

    ContextualBlockExecutorV1(const ContextualBlockExecutorV1&) = delete;
    ContextualBlockExecutorV1& operator=(const ContextualBlockExecutorV1&) =
        delete;

    nb::dict preflight() const {
        nb::dict result = contextual_dense_preflight_dict_v1(estimate_);
        result["admitted"] = true;
        result["workspace_cap_bytes"] = workspace_cap_bytes_;
        result["telemetry_capacity"] = telemetry_capacity_;
        result["variant_block"] = variant_block_;
        result["sample_probe_tile"] = sample_probe_tile_;
        result["variant_probe_tile"] = variant_probe_tile_;
        result["action_tile"] = action_tile_;
        result["annotation_tile"] = annotation_tile_;
        result["context_tile"] = context_tile_;
        result["group_tile"] = group_tile_;
        result["trait_feature_tile"] = trait_feature_tile_;
        return result;
    }

    nb::dict info() const {
        nb::dict result;
        result["native_api_version"] = 1;
        result["backend"] = "pre_scaled_dense_differential_v1";
        result["genotype_scale_policy"] = "pre_scaled_dense_v1";
        result["numeric_policy"] = "fp64_v1";
        result["state"] = state_;
        result["lifecycle"] = state_;
        result["n"] = n_;
        result["m"] = m_;
        result["fixed_rank"] = rank_;
        result["q"] = q_;
        result["k"] = k_;
        result["pair_count"] = pairs_;
        result["component_count"] = components_;
        result["group_count"] = groups_;
        result["sample_probe_count"] = bt_;
        result["variant_probe_count"] = bd_;
        result["residual_component_count"] = h_;
        result["annotation_mode"] = annotation_mode_;
        result["annotation_names"] = annotation_names_;
        result["group_names"] = group_names_;
        result["fault_operation"] = fault_operation_;
        result["fault_mode"] = fault_mode_;
        result["fault_occurrence"] = fault_occurrence_;
        result["admission"] = preflight();
        return result;
    }

    nb::dict run() {
        if (state_ != "admitted")
            throw std::runtime_error(
                "ContextualBlockExecutorV1 lifecycle: run is one-shot"
            );
        state_ = "running";
        try {
            {
                nb::gil_scoped_release release;
                initialize_outputs();
                set_phase("source");
                compute_sample_reference();
                set_phase("group");
                compute_group_restricted();
                compute_group_direct();
                set_phase("same_person");
                compute_same_person();
                set_phase("trait");
                compute_trait();
                set_phase("residual_moments");
                compute_residual_moments();
                set_phase("finalization");
                finalize_and_validate();
                set_phase("scratch_release");
                release_execution_scratch();
                set_phase("publication");
            }
            state_ = "published";
            nb::dict result = publish();
            return result;
        } catch (...) {
            state_ = "failed";
            throw;
        }
    }

private:
    static constexpr double kCanaryV1 = -9.876543210987654e207;

    template <typename T>
    static void copy_array(const T* input, std::size_t count,
                           std::vector<T>& output) {
        output.assign(input, input + count);
    }

    static std::size_t checked_size(std::uint64_t value, const char* label) {
        if (value > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error(
                std::string("ContextualBlockExecutorV1 checked overflow: ") +
                label
            );
        return static_cast<std::size_t>(value);
    }

    static std::size_t checked_product_size(
        std::initializer_list<std::size_t> factors, const char* label
    ) {
        __uint128_t value = 1;
        for (std::size_t factor : factors)
            value *= static_cast<__uint128_t>(factor);
        if (value > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error(
                std::string("ContextualBlockExecutorV1 checked overflow: ") +
                label
            );
        return static_cast<std::size_t>(value);
    }

    void validate_blas_and_layout_dimensions() const {
        for (const auto& entry : std::array<std::pair<std::size_t, const char*>, 9>{{
                 {n_, "N"}, {m_, "M"}, {rank_, "fixed rank"}, {q_, "Q"},
                 {k_, "K"}, {groups_, "group count"}, {bt_, "sample probes"},
                 {bd_, "variant probes"}, {components_, "component count"}
             }}) {
            if (entry.first > static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
                throw std::runtime_error(
                    std::string("ContextualBlockExecutorV1 checked overflow: ") +
                    entry.second + " exceeds protected BLAS integer range"
                );
        }
        (void)checked_product_size({n_, m_}, "genotype layout");
        (void)checked_product_size({n_, bt_}, "sample-probe layout");
        (void)checked_product_size({m_, bd_}, "variant-probe layout");
        const std::size_t flattened_sample_rows = checked_product_size(
            {n_, bt_}, "flattened sample-action reduction"
        );
        if (flattened_sample_rows > static_cast<std::size_t>(
                std::numeric_limits<int>::max()))
            throw std::runtime_error(
                "ContextualBlockExecutorV1 checked overflow: flattened "
                "sample-action reduction exceeds protected BLAS integer range"
            );
        auto require_blas_product = [&](std::initializer_list<std::size_t> factors,
                                        const char* label) {
            const std::size_t value = checked_product_size(factors, label);
            if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
                throw std::runtime_error(
                    std::string("ContextualBlockExecutorV1 checked overflow: ") +
                    label + " exceeds protected BLAS integer range"
                );
        };
        require_blas_product({q_, sample_probe_tile_},
                             "source protected columns");
        require_blas_product({k_, q_, sample_probe_tile_},
                             "target protected columns");
        require_blas_product({action_tile_, sample_probe_tile_},
                             "action protected columns");
        require_blas_product({n_, sample_probe_tile_},
                             "group cross-Gram reduction");
        require_blas_product({k_, variant_probe_tile_},
                             "same-person target columns");
        require_blas_product({k_, q_, variant_probe_tile_},
                             "same-person projection columns");
        require_blas_product({n_, variant_probe_tile_},
                             "same-person Gram reduction");
        require_blas_product({q_, variant_block_},
                             "trait feature columns");
    }

    static std::size_t resolve_tile(
        std::uint64_t requested, std::size_t full, const char* label
    ) {
        if (full == 0)
            throw std::runtime_error(
                std::string("ContextualBlockExecutorV1 validation: empty ") +
                label
            );
        if (requested == 0)
            return full;
        return checked_size(std::min<std::uint64_t>(requested, full), label);
    }

    void validate_shapes(
        const nb_mat2f_ro<double>& fixed_basis,
        const nb_mat2f_ro<double>& context_basis,
        const nb_mat2f_ro<double>& annotation_weights,
        const nb_vec1_ro<std::int64_t>& group_index,
        const nb_mat2f_ro<double>& sample_probes,
        const nb_mat2f_ro<double>& variant_probes,
        const nb_vec1_ro<double>& phenotype_raw,
        const nb_mat2f_ro<double>& residual_basis
    ) const {
        if (n_ == 0 || m_ == 0 || rank_ == 0 || q_ == 0 || k_ == 0 ||
            groups_ == 0 || bt_ == 0 || bd_ < 2 || h_ == 0)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: dimensions must be "
                "positive and variant probe count must be at least two"
            );
        if (q_ > 4)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: Q must be in [1,4]"
            );
        if (rank_ >= n_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: fixed rank must be below N"
            );
        if (fixed_basis.shape(0) != n_ || context_basis.shape(0) != n_ ||
            annotation_weights.shape(0) != m_ || group_index.shape(0) != m_ ||
            sample_probes.shape(0) != n_ || variant_probes.shape(0) != m_ ||
            phenotype_raw.shape(0) != n_ || residual_basis.shape(0) != n_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: incompatible input shapes"
            );
        if (annotation_names_.size() != k_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: annotation name count "
                "does not match K"
            );
    }

    void validate_policy() {
        if (threads_ <= 0)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: threads must be positive"
            );
        if (annotation_mode_ != "strict_disjoint_binary_v1" &&
            annotation_mode_ != "generic_nonnegative_weights_v1")
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: unsupported annotation mode"
            );
        validate_names(annotation_names_, "annotation");
        validate_names(group_names_, "group");
        if (fault_occurrence_ == 0)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: fault_occurrence must be positive"
            );
        static const std::array<const char*, 10> modes = {{
            "none", "one_shot", "repeated", "force_fallback",
            "repair_corruption", "fallback_corruption", "nan", "canary",
            "inf", "operand_mutation"
        }};
        if (std::find_if(modes.begin(), modes.end(), [&](const char* value) {
                return fault_mode_ == value;
            }) == modes.end())
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: unsupported fault_mode"
            );
        if (fault_mode_ == "none") {
            if (!fault_operation_.empty())
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 validation: fault operation "
                    "requires a non-none mode"
                );
            return;
        }
        auto found = std::find_if(
            kContextualDenseSemanticNamesV1.begin(),
            kContextualDenseSemanticNamesV1.end(),
            [&](const char* value) { return fault_operation_ == value; }
        );
        if (found == kContextualDenseSemanticNamesV1.end())
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: unknown fault_operation"
            );
        fault_index_ = static_cast<std::size_t>(
            found - kContextualDenseSemanticNamesV1.begin()
        );
    }

    static void validate_names(
        const std::vector<std::string>& names, const char* kind
    ) {
        std::vector<std::string> sorted = names;
        for (const auto& name : sorted) {
            if (name.empty())
                throw std::runtime_error(
                    std::string("ContextualBlockExecutorV1 validation: empty ") +
                    kind + " name"
                );
        }
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::runtime_error(
                std::string("ContextualBlockExecutorV1 validation: duplicate ") +
                kind + " name"
            );
    }

    void build_maps() {
        for (std::size_t value = 0; value < q_; ++value) {
            pair_q_.push_back(static_cast<std::int64_t>(value));
            pair_r_.push_back(static_cast<std::int64_t>(value));
            pair_eta_.push_back(1);
        }
        for (std::size_t left = 0; left < q_; ++left) {
            for (std::size_t right = left + 1; right < q_; ++right) {
                pair_q_.push_back(static_cast<std::int64_t>(left));
                pair_r_.push_back(static_cast<std::int64_t>(right));
                pair_eta_.push_back(2);
            }
        }
        pairs_ = pair_q_.size();
        if (k_ > std::numeric_limits<std::size_t>::max() / pairs_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 checked overflow: component count"
            );
        components_ = k_ * pairs_;
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            for (std::size_t pair = 0; pair < pairs_; ++pair) {
                component_annotation_.push_back(
                    static_cast<std::int64_t>(annotation)
                );
                component_pair_.push_back(static_cast<std::int64_t>(pair));
            }
        }
    }

    template <typename T>
    static bool all_finite(const std::vector<T>& values) {
        for (T value : values) {
            if (!std::isfinite(static_cast<double>(value)))
                return false;
        }
        return true;
    }

    void validate_owned_values() const {
        if (!all_finite(genotype_) || !all_finite(fixed_basis_) ||
            !all_finite(context_basis_) || !all_finite(annotation_weights_) ||
            !all_finite(sample_probes_) || !all_finite(variant_probes_) ||
            !all_finite(phenotype_raw_) || !all_finite(residual_basis_))
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: all numeric inputs must be finite"
            );
        for (std::size_t column = 0; column < rank_; ++column) {
            for (std::size_t other = 0; other < rank_; ++other) {
                double dot = 0.0;
                for (std::size_t row = 0; row < n_; ++row)
                    dot += fixed_basis_[row + n_ * column] *
                           fixed_basis_[row + n_ * other];
                const double target = column == other ? 1.0 : 0.0;
                if (std::abs(dot - target) > 1.0e-10)
                    throw std::runtime_error(
                        "ContextualBlockExecutorV1 validation: fixed_basis "
                        "columns must be orthonormal"
                    );
            }
        }
        std::vector<double> phenotype_coefficients(rank_, 0.0);
        for (std::size_t basis = 0; basis < rank_; ++basis)
            for (std::size_t row = 0; row < n_; ++row)
                phenotype_coefficients[basis] +=
                    fixed_basis_[row + n_ * basis] * phenotype_raw_[row];
        double projected_phenotype_norm = 0.0;
        for (std::size_t row = 0; row < n_; ++row) {
            double value = phenotype_raw_[row];
            for (std::size_t basis = 0; basis < rank_; ++basis)
                value -= fixed_basis_[row + n_ * basis] *
                         phenotype_coefficients[basis];
            projected_phenotype_norm += value * value;
        }
        if (!(projected_phenotype_norm > 0.0))
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: phenotype has zero "
                "projected norm"
            );
        for (double value : annotation_weights_) {
            if (value < 0.0)
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 validation: annotation weights "
                    "must be nonnegative"
                );
        }
        if (annotation_mode_ == "strict_disjoint_binary_v1") {
            for (std::size_t variant = 0; variant < m_; ++variant) {
                std::size_t memberships = 0;
                for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                    const double value = annotation_weights_[
                        variant + m_ * annotation
                    ];
                    if (value == 1.0)
                        ++memberships;
                    else if (value != 0.0)
                        throw std::runtime_error(
                            "ContextualBlockExecutorV1 validation: strict "
                            "annotations must be binary"
                        );
                }
                if (memberships != 1)
                    throw std::runtime_error(
                        "ContextualBlockExecutorV1 validation: strict annotations "
                        "require exactly one membership per variant"
                    );
            }
        }
        std::vector<std::size_t> counts(groups_, 0);
        for (std::int64_t value : group_index_) {
            if (value < 0 || static_cast<std::size_t>(value) >= groups_)
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 validation: group index is out of range"
                );
            ++counts[static_cast<std::size_t>(value)];
        }
        if (std::find(counts.begin(), counts.end(), 0) != counts.end())
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: every canonical group "
                "must contain a variant"
            );
        auto validate_rademacher = [](const std::vector<double>& values,
                                      const char* kind) {
            for (double value : values) {
                if (value != -1.0 && value != 1.0)
                    throw std::runtime_error(
                        std::string("ContextualBlockExecutorV1 validation: ") +
                        kind + " probes must be explicit Rademacher values"
                    );
            }
        };
        validate_rademacher(sample_probes_, "sample");
        validate_rademacher(variant_probes_, "variant");
    }

    void compute_masses() {
        annotation_masses_.assign(k_, 0.0);
        group_annotation_masses_.assign(groups_ * k_, 0.0);
        group_variant_counts_.assign(groups_, 0);
        for (std::size_t variant = 0; variant < m_; ++variant) {
            const std::size_t group = static_cast<std::size_t>(
                group_index_[variant]
            );
            ++group_variant_counts_[group];
            for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                const double value = annotation_value(variant, annotation);
                annotation_masses_[annotation] += value;
                group_annotation_masses_[group * k_ + annotation] += value;
            }
        }
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            if (!(annotation_masses_[annotation] > 0.0))
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 validation: annotation masses "
                    "must be positive"
                );
            for (std::size_t group = 0; group < groups_; ++group) {
                if (!(annotation_masses_[annotation] -
                      group_annotation_masses_[group * k_ + annotation] > 0.0))
                    throw std::runtime_error(
                        "ContextualBlockExecutorV1 validation: every one-group "
                        "deletion must retain positive annotation mass"
                    );
            }
        }
    }

    void seal_annotation_representation() {
        if (annotation_mode_ != "strict_disjoint_binary_v1")
            return;
        strict_annotation_ids_.assign(m_, 0);
        for (std::size_t variant = 0; variant < m_; ++variant) {
            for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                if (annotation_weights_[variant + m_ * annotation] == 1.0) {
                    strict_annotation_ids_[variant] = annotation;
                    break;
                }
            }
        }
    }

    double annotation_value(
        std::size_t variant, std::size_t annotation
    ) const noexcept {
        if (!strict_annotation_ids_.empty())
            return strict_annotation_ids_[variant] == annotation ? 1.0 : 0.0;
        return annotation_weights_[variant + m_ * annotation];
    }

    static std::uint64_t hash_bytes(
        std::uint64_t hash, const void* pointer, std::size_t bytes
    ) {
        const auto* data = static_cast<const unsigned char*>(pointer);
        for (std::size_t index = 0; index < bytes; ++index) {
            hash ^= data[index];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    template <typename T>
    static std::uint64_t hash_vector(
        std::uint64_t hash, const std::vector<T>& values
    ) {
        const std::uint64_t count = values.size();
        hash = hash_bytes(hash, &count, sizeof(count));
        return values.empty()
            ? hash
            : hash_bytes(hash, values.data(), values.size() * sizeof(T));
    }

    std::uint64_t fingerprint_inputs() const {
        std::uint64_t hash = UINT64_C(1469598103934665603);
        hash = hash_vector(hash, genotype_);
        hash = hash_vector(hash, fixed_basis_);
        hash = hash_vector(hash, context_basis_);
        hash = hash_vector(hash, annotation_weights_);
        hash = hash_vector(hash, group_index_);
        hash = hash_vector(hash, sample_probes_);
        hash = hash_vector(hash, variant_probes_);
        hash = hash_vector(hash, phenotype_raw_);
        return hash_vector(hash, residual_basis_);
    }

    void check_inputs_unchanged() const {
        if (fingerprint_inputs() != input_fingerprint_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: sealed input mutation detected"
            );
    }

    void record_event(
        const char* event_class, const char* operation,
        const char* resolution
    ) {
        if (events_.size() >= telemetry_capacity_ ||
            events_.size() >= estimate_.required_telemetry_capacity)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: telemetry overflow"
            );
        events_.push_back(ContextualDenseEventV1{
            event_class, phase_, operation, resolution
        });
    }

    void set_phase(const char* phase) {
        check_inputs_unchanged();
        phase_ = phase;
        record_event("phase_transition", "", "ok");
    }

    bool should_inject(ContextualDenseSemanticV1 operation) {
        const std::size_t index = static_cast<std::size_t>(operation);
        if (fault_mode_ == "none" || index != fault_index_)
            return false;
        ++fault_seen_occurrences_;
        if (fault_seen_occurrences_ != fault_occurrence_)
            return false;
        ++injection_count_;
        fault_consumed_ = true;
        return true;
    }

    static bool finite_panel(
        const double* values, std::size_t count
    ) {
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(values[index]))
                return false;
        }
        return true;
    }

    static bool panels_agree(
        const double* left, const double* right, std::size_t count
    ) {
        for (std::size_t index = 0; index < count; ++index) {
            const double scale = 1.0 + std::max(
                std::abs(left[index]), std::abs(right[index])
            );
            if (!std::isfinite(left[index]) ||
                std::abs(left[index] - right[index]) > 2.0e-12 * scale)
                return false;
        }
        return true;
    }

    void verify_canaries(std::size_t count) const {
        const std::size_t end = std::min(
            retry_buffer_.size(), count + static_cast<std::size_t>(16)
        );
        for (std::size_t index = count; index < end; ++index) {
            if (retry_buffer_[index] != kCanaryV1 ||
                witness_buffer_[index] != kCanaryV1 ||
                fallback_buffer_[index] != kCanaryV1)
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 integrity: scratch canary corruption"
                );
        }
    }

    void protected_nn(
        ContextualDenseSemanticV1 operation,
        int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output, int output_stride
    ) {
        protected_gemm(
            false, operation, rows, columns, reduction,
            left, left_stride, right, right_stride, output, output_stride
        );
    }

    void protected_tn(
        ContextualDenseSemanticV1 operation,
        int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output, int output_stride
    ) {
        protected_gemm(
            true, operation, rows, columns, reduction,
            left, left_stride, right, right_stride, output, output_stride
        );
    }

    void protected_gemm(
        bool transpose_left, ContextualDenseSemanticV1 operation,
        int rows, int columns, int reduction,
        const double* left, int left_stride,
        const double* right, int right_stride,
        double* output, int output_stride
    ) {
        if (rows <= 0 || columns <= 0 || reduction <= 0)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: empty protected GEMM"
            );
        const std::size_t count = static_cast<std::size_t>(rows) * columns;
        if (count + 16 > retry_buffer_.size())
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: protected scratch admission mismatch"
            );
        const auto op_index = static_cast<std::size_t>(operation);
        ++observed_calls_[op_index];
        const char* op_name = kContextualDenseSemanticNamesV1[op_index];
        check_inputs_unchanged();

        std::fill(retry_buffer_.begin(), retry_buffer_.begin() + count + 16,
                  kCanaryV1);
        std::fill(witness_buffer_.begin(), witness_buffer_.begin() + count + 16,
                  kCanaryV1);
        std::fill(fallback_buffer_.begin(), fallback_buffer_.begin() + count + 16,
                  kCanaryV1);
        GemmIntegrityResolution native_resolution{};
        if (transpose_left) {
            (void)dgemm_tn_partitioned_rows(
                rows, columns, reduction,
                left, left_stride, right, right_stride,
                output, output_stride, threads_, 1.0, 0.0,
                nullptr, false, &native_resolution
            );
            dgemm_tn_tiled(
                rows, columns, reduction,
                left, left_stride, right, right_stride,
                witness_buffer_.data(), rows, threads_
            );
        } else {
            (void)dgemm_nn_partitioned_rows(
                rows, columns, reduction,
                left, left_stride, right, right_stride,
                output, output_stride, threads_, 1.0, 0.0,
                &native_resolution
            );
            dgemm_nn_tiled(
                rows, columns, reduction,
                left, left_stride, right, right_stride,
                witness_buffer_.data(), rows, threads_
            );
        }

        const bool inject = should_inject(operation);
        if (inject && fault_mode_ == "operand_mutation")
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: injected sealed operand mutation"
            );
        if (inject && count > 0) {
            if (fault_mode_ == "nan")
                output[0] = std::numeric_limits<double>::quiet_NaN();
            else if (fault_mode_ == "inf")
                output[0] = std::numeric_limits<double>::infinity();
            else if (fault_mode_ == "canary")
                output[0] = kCanaryV1;
            else if (fault_mode_ != "force_fallback" &&
                     fault_mode_ != "fallback_corruption")
                output[0] += 1024.0 + std::abs(output[0]);
            record_event("fault_injection", op_name, fault_mode_.c_str());
        }

        const bool forced_fallback = inject &&
            (fault_mode_ == "force_fallback" ||
             fault_mode_ == "fallback_corruption");
        const bool primary_valid = !forced_fallback &&
            finite_panel(output, count) &&
            panels_agree(output, witness_buffer_.data(), count);
        if (primary_valid) {
            record_event("protected_call", op_name, "verified_primary");
            verify_canaries(count);
            return;
        }

        ++repair_count_;
        record_event("repair", op_name, "primary_mismatch");
        if (fault_mode_ != "force_fallback" &&
            fault_mode_ != "fallback_corruption") {
            ++retry_count_;
            if (transpose_left) {
                (void)dgemm_tn_partitioned_rows(
                    rows, columns, reduction,
                    left, left_stride, right, right_stride,
                    retry_buffer_.data(), rows, threads_, 1.0, 0.0,
                    nullptr, false, &native_resolution
                );
            } else {
                (void)dgemm_nn_partitioned_rows(
                    rows, columns, reduction,
                    left, left_stride, right, right_stride,
                    retry_buffer_.data(), rows, threads_, 1.0, 0.0,
                    &native_resolution
                );
            }
            if (inject && (fault_mode_ == "repeated" ||
                           fault_mode_ == "repair_corruption" ||
                           fault_mode_ == "canary"))
                retry_buffer_[0] += 2048.0 + std::abs(retry_buffer_[0]);
            if (finite_panel(retry_buffer_.data(), count) &&
                panels_agree(retry_buffer_.data(), witness_buffer_.data(), count)) {
                for (int column = 0; column < columns; ++column)
                    std::copy_n(
                        retry_buffer_.data() + static_cast<std::size_t>(rows) * column,
                        rows, output + static_cast<std::size_t>(output_stride) * column
                    );
                record_event("retry", op_name, "verified_retry");
                verify_canaries(count);
                return;
            }
            record_event("retry", op_name, "retry_mismatch");
            if (fault_mode_ == "repair_corruption")
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 integrity: injected repair corruption"
                );
        }

        ++fallback_count_;
        std::copy_n(witness_buffer_.data(), count, fallback_buffer_.data());
        if (inject && fault_mode_ == "fallback_corruption")
            fallback_buffer_[0] = std::numeric_limits<double>::quiet_NaN();
        if (!finite_panel(fallback_buffer_.data(), count) ||
            !panels_agree(fallback_buffer_.data(), witness_buffer_.data(), count))
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: trusted fallback corruption"
            );
        for (int column = 0; column < columns; ++column)
            std::copy_n(
                fallback_buffer_.data() + static_cast<std::size_t>(rows) * column,
                rows, output + static_cast<std::size_t>(output_stride) * column
            );
        record_event("trusted_fallback", op_name, "verified_fallback");
        verify_canaries(count);
    }

    void project_panel(
        const std::vector<double>& input, std::size_t columns,
        ContextualDenseSemanticV1 tn_operation,
        ContextualDenseSemanticV1 nn_operation,
        std::vector<double>& output
    ) {
        output = input;
        std::vector<double> coefficients(rank_ * columns);
        std::vector<double> correction(n_ * columns);
        protected_tn(
            tn_operation, static_cast<int>(rank_), static_cast<int>(columns),
            static_cast<int>(n_), fixed_basis_.data(), static_cast<int>(n_),
            input.data(), static_cast<int>(n_), coefficients.data(),
            static_cast<int>(rank_)
        );
        protected_nn(
            nn_operation, static_cast<int>(n_), static_cast<int>(columns),
            static_cast<int>(rank_), fixed_basis_.data(), static_cast<int>(n_),
            coefficients.data(), static_cast<int>(rank_), correction.data(),
            static_cast<int>(n_)
        );
        for (std::size_t index = 0; index < output.size(); ++index)
            output[index] -= correction[index];
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t basis = 0; basis < rank_; ++basis) {
                double dot = 0.0;
                for (std::size_t row = 0; row < n_; ++row)
                    dot += fixed_basis_[row + n_ * basis] *
                           output[row + n_ * column];
                maximum_leakage_ = std::max(maximum_leakage_, std::abs(dot));
            }
        }
    }

    void initialize_outputs() {
        source_scores_.assign(m_ * q_ * bt_, 0.0);
        raw_actions_.assign(n_ * components_ * bt_, 0.0);
        gram_.assign(components_ * components_, 0.0);
        raw_gram_numerator_.assign(components_ * components_, 0.0);
        same_person_.assign(components_ * components_, 0.0);
        group_restricted_.assign(groups_ * components_ * components_, 0.0);
        group_direct_action_.assign(groups_ * components_ * components_, 0.0);
        group_direct_genotype_.assign(groups_ * components_ * components_, 0.0);
        genetic_rhs_.assign(components_, 0.0);
        genetic_traces_.assign(components_, 0.0);
        genetic_residual_.assign(components_ * h_, 0.0);
        residual_rhs_.assign(h_, 0.0);
        residual_traces_.assign(h_, 0.0);
        residual_gram_.assign(h_ * h_, 0.0);
        group_rhs_.assign(groups_ * components_, 0.0);
        group_trace_.assign(groups_ * components_, 0.0);
        group_genetic_residual_.assign(groups_ * components_ * h_, 0.0);
    }

    void compute_sample_reference() {
        const std::size_t variant_blocks =
            (m_ + variant_block_ - 1) / variant_block_;
        observed_variant_blocks_ = variant_blocks;
        for (std::size_t b0 = 0; b0 < bt_; b0 += sample_probe_tile_) {
            const std::size_t probes = std::min(sample_probe_tile_, bt_ - b0);
            observed_sample_probe_coverage_ += probes;
            std::vector<double> z_panel(n_ * probes);
            for (std::size_t probe = 0; probe < probes; ++probe)
                std::copy_n(
                    sample_probes_.data() + n_ * (b0 + probe), n_,
                    z_panel.data() + n_ * probe
                );
            std::vector<double> projected_z;
            project_panel(
                z_panel, probes,
                ContextualDenseSemanticV1::sample_probe_projection_tn,
                ContextualDenseSemanticV1::sample_probe_projection_nn,
                projected_z
            );
            std::vector<double> source_right(n_ * q_ * probes);
            for (std::size_t context = 0; context < q_; ++context) {
                for (std::size_t probe = 0; probe < probes; ++probe) {
                    double* column = source_right.data() +
                        n_ * (context * probes + probe);
                    for (std::size_t row = 0; row < n_; ++row)
                        column[row] = context_basis_[row + n_ * context] *
                                      projected_z[row + n_ * probe];
                }
            }
            for (std::size_t v0 = 0; v0 < m_; v0 += variant_block_) {
                const std::size_t variants = std::min(variant_block_, m_ - v0);
                for (std::size_t t0 = 0; t0 < q_; t0 += context_tile_) {
                    const std::size_t contexts = std::min(context_tile_, q_ - t0);
                    std::vector<double> local(variants * contexts * probes);
                    protected_tn(
                        ContextualDenseSemanticV1::source_tn,
                        static_cast<int>(variants),
                        static_cast<int>(contexts * probes), static_cast<int>(n_),
                        genotype_.data() + n_ * v0, static_cast<int>(n_),
                        source_right.data() + n_ * t0 * probes,
                        static_cast<int>(n_), local.data(),
                        static_cast<int>(variants)
                    );
                    for (std::size_t local_context = 0;
                         local_context < contexts; ++local_context) {
                        const std::size_t context = t0 + local_context;
                        for (std::size_t probe = 0; probe < probes; ++probe) {
                            for (std::size_t local_variant = 0;
                                 local_variant < variants; ++local_variant) {
                                source_scores_[
                                    (v0 + local_variant) +
                                    m_ * (context * bt_ + b0 + probe)
                                ] = local[
                                    local_variant + variants *
                                        (local_context * probes + probe)
                                ];
                            }
                        }
                    }
                }
            }

            std::vector<double> targets(n_ * k_ * q_ * probes);
            for (std::size_t a0 = 0; a0 < k_; a0 += annotation_tile_) {
                const std::size_t annotations = std::min(annotation_tile_, k_ - a0);
                for (std::size_t t0 = 0; t0 < q_; t0 += context_tile_) {
                    const std::size_t contexts = std::min(context_tile_, q_ - t0);
                    std::vector<double> weighted_source(
                        m_ * annotations * contexts * probes
                    );
                    for (std::size_t local_annotation = 0;
                         local_annotation < annotations; ++local_annotation) {
                        const std::size_t annotation = a0 + local_annotation;
                        for (std::size_t local_context = 0;
                             local_context < contexts; ++local_context) {
                            const std::size_t context = t0 + local_context;
                            for (std::size_t probe = 0; probe < probes; ++probe) {
                                double* column = weighted_source.data() + m_ *
                                    ((local_annotation * contexts + local_context) *
                                     probes + probe);
                                const double* source = source_scores_.data() + m_ *
                                    (context * bt_ + b0 + probe);
                                for (std::size_t variant = 0; variant < m_; ++variant)
                                    column[variant] =
                                        annotation_value(variant, annotation) *
                                        source[variant];
                            }
                        }
                    }
                    std::vector<double> local_targets(
                        n_ * annotations * contexts * probes
                    );
                    protected_nn(
                        ContextualDenseSemanticV1::full_target_nn,
                        static_cast<int>(n_),
                        static_cast<int>(annotations * contexts * probes),
                        static_cast<int>(m_), genotype_.data(),
                        static_cast<int>(n_), weighted_source.data(),
                        static_cast<int>(m_), local_targets.data(),
                        static_cast<int>(n_)
                    );
                    for (std::size_t local_annotation = 0;
                         local_annotation < annotations; ++local_annotation) {
                        for (std::size_t local_context = 0;
                             local_context < contexts; ++local_context) {
                            for (std::size_t probe = 0; probe < probes; ++probe)
                                std::copy_n(
                                    local_targets.data() + n_ *
                                        ((local_annotation * contexts + local_context) *
                                         probes + probe), n_,
                                    targets.data() + n_ *
                                        (((a0 + local_annotation) * q_ +
                                          t0 + local_context) * probes + probe)
                                );
                        }
                    }
                }
            }
            std::vector<double> unprojected(n_ * components_ * probes);
            for (std::size_t component = 0; component < components_; ++component) {
                const std::size_t annotation = static_cast<std::size_t>(
                    component_annotation_[component]
                );
                const std::size_t pair = static_cast<std::size_t>(
                    component_pair_[component]
                );
                const std::size_t left = static_cast<std::size_t>(pair_q_[pair]);
                const std::size_t right = static_cast<std::size_t>(pair_r_[pair]);
                for (std::size_t probe = 0; probe < probes; ++probe) {
                    double* destination = unprojected.data() + n_ *
                        (component * probes + probe);
                    const double* right_target = targets.data() + n_ *
                        ((annotation * q_ + right) * probes + probe);
                    const double* left_target = targets.data() + n_ *
                        ((annotation * q_ + left) * probes + probe);
                    for (std::size_t row = 0; row < n_; ++row) {
                        destination[row] = context_basis_[row + n_ * left] *
                                           right_target[row];
                        if (left != right)
                            destination[row] += context_basis_[row + n_ * right] *
                                                left_target[row];
                    }
                }
            }
            for (std::size_t c0 = 0; c0 < components_; c0 += action_tile_) {
                const std::size_t count = std::min(action_tile_, components_ - c0);
                std::vector<double> input(n_ * count * probes);
                for (std::size_t local = 0; local < count; ++local) {
                    for (std::size_t probe = 0; probe < probes; ++probe)
                        std::copy_n(
                            unprojected.data() + n_ *
                                ((c0 + local) * probes + probe), n_,
                            input.data() + n_ * (local * probes + probe)
                        );
                }
                std::vector<double> projected;
                project_panel(
                    input, count * probes,
                    ContextualDenseSemanticV1::action_projection_tn,
                    ContextualDenseSemanticV1::action_projection_nn,
                    projected
                );
                for (std::size_t local = 0; local < count; ++local) {
                    for (std::size_t probe = 0; probe < probes; ++probe)
                        std::copy_n(
                            projected.data() + n_ * (local * probes + probe), n_,
                            raw_actions_.data() + n_ *
                                ((c0 + local) * bt_ + b0 + probe)
                        );
                }
            }
        }

        std::vector<double> flat(n_ * bt_ * components_);
        for (std::size_t component = 0; component < components_; ++component) {
            for (std::size_t probe = 0; probe < bt_; ++probe)
                std::copy_n(
                    raw_actions_.data() + n_ * (component * bt_ + probe), n_,
                    flat.data() + n_ * (probe + bt_ * component)
                );
        }
        std::vector<double> gram_col(components_ * components_);
        protected_tn(
            ContextualDenseSemanticV1::action_gram_tn,
            static_cast<int>(components_), static_cast<int>(components_),
            static_cast<int>(n_ * bt_), flat.data(),
            static_cast<int>(n_ * bt_), flat.data(),
            static_cast<int>(n_ * bt_), gram_col.data(),
            static_cast<int>(components_)
        );
        for (std::size_t left = 0; left < components_; ++left) {
            const std::size_t left_annotation = static_cast<std::size_t>(
                component_annotation_[left]
            );
            for (std::size_t right = 0; right < components_; ++right) {
                const std::size_t right_annotation = static_cast<std::size_t>(
                    component_annotation_[right]
                );
                const double raw = gram_col[left + components_ * right] /
                                   static_cast<double>(bt_);
                raw_gram_numerator_[left * components_ + right] = raw;
                gram_[left * components_ + right] = raw /
                    (annotation_masses_[left_annotation] *
                     annotation_masses_[right_annotation]);
            }
        }
        gram_pre_symmetry_ = symmetry_error(gram_, components_);
        symmetrize(gram_, components_);
        symmetrize(raw_gram_numerator_, components_);
    }

    void compute_group_restricted() {
        for (std::size_t g0 = 0; g0 < groups_; g0 += group_tile_) {
            const std::size_t group_count = std::min(group_tile_, groups_ - g0);
            for (std::size_t local_group = 0; local_group < group_count;
                 ++local_group) {
                const std::size_t group = g0 + local_group;
                for (std::size_t b0 = 0; b0 < bt_; b0 += sample_probe_tile_) {
                    const std::size_t probes = std::min(sample_probe_tile_, bt_ - b0);
                    std::vector<double> targets(n_ * k_ * q_ * probes);
                    for (std::size_t a0 = 0; a0 < k_; a0 += annotation_tile_) {
                        const std::size_t annotations = std::min(
                            annotation_tile_, k_ - a0
                        );
                        for (std::size_t t0 = 0; t0 < q_; t0 += context_tile_) {
                            const std::size_t contexts = std::min(
                                context_tile_, q_ - t0
                            );
                            std::vector<double> weighted(
                                m_ * annotations * contexts * probes, 0.0
                            );
                            for (std::size_t local_annotation = 0;
                                 local_annotation < annotations; ++local_annotation) {
                                const std::size_t annotation = a0 + local_annotation;
                                for (std::size_t local_context = 0;
                                     local_context < contexts; ++local_context) {
                                    const std::size_t context = t0 + local_context;
                                    for (std::size_t probe = 0; probe < probes; ++probe) {
                                        double* column = weighted.data() + m_ *
                                            ((local_annotation * contexts +
                                              local_context) * probes + probe);
                                        const double* source = source_scores_.data() + m_ *
                                            (context * bt_ + b0 + probe);
                                        for (std::size_t variant = 0;
                                             variant < m_; ++variant) {
                                            if (static_cast<std::size_t>(
                                                    group_index_[variant]) == group)
                                                column[variant] =
                                                    annotation_value(
                                                        variant, annotation
                                                    ) * source[variant];
                                        }
                                    }
                                }
                            }
                            std::vector<double> local_targets(
                                n_ * annotations * contexts * probes
                            );
                            protected_nn(
                                ContextualDenseSemanticV1::group_target_nn,
                                static_cast<int>(n_),
                                static_cast<int>(annotations * contexts * probes),
                                static_cast<int>(m_), genotype_.data(),
                                static_cast<int>(n_), weighted.data(),
                                static_cast<int>(m_), local_targets.data(),
                                static_cast<int>(n_)
                            );
                            for (std::size_t local_annotation = 0;
                                 local_annotation < annotations; ++local_annotation) {
                                for (std::size_t local_context = 0;
                                     local_context < contexts; ++local_context) {
                                    for (std::size_t probe = 0; probe < probes; ++probe)
                                        std::copy_n(
                                            local_targets.data() + n_ *
                                                ((local_annotation * contexts +
                                                  local_context) * probes + probe), n_,
                                            targets.data() + n_ *
                                                (((a0 + local_annotation) * q_ +
                                                  t0 + local_context) * probes + probe)
                                        );
                                }
                            }
                        }
                    }
                    std::vector<double> xflat(n_ * probes * components_);
                    std::vector<double> wflat(n_ * probes * components_);
                    for (std::size_t component = 0; component < components_; ++component) {
                        const std::size_t annotation = static_cast<std::size_t>(
                            component_annotation_[component]
                        );
                        const std::size_t pair = static_cast<std::size_t>(
                            component_pair_[component]
                        );
                        const std::size_t left = static_cast<std::size_t>(pair_q_[pair]);
                        const std::size_t right = static_cast<std::size_t>(pair_r_[pair]);
                        for (std::size_t probe = 0; probe < probes; ++probe) {
                            double* x = xflat.data() + n_ *
                                (probe + probes * component);
                            const double* rt = targets.data() + n_ *
                                ((annotation * q_ + right) * probes + probe);
                            const double* lt = targets.data() + n_ *
                                ((annotation * q_ + left) * probes + probe);
                            for (std::size_t row = 0; row < n_; ++row) {
                                x[row] = context_basis_[row + n_ * left] * rt[row];
                                if (left != right)
                                    x[row] += context_basis_[row + n_ * right] * lt[row];
                            }
                            std::copy_n(
                                raw_actions_.data() + n_ *
                                    (component * bt_ + b0 + probe), n_,
                                wflat.data() + n_ * (probe + probes * component)
                            );
                        }
                    }
                    std::vector<double> cross(components_ * components_);
                    protected_tn(
                        ContextualDenseSemanticV1::group_cross_gram_tn,
                        static_cast<int>(components_),
                        static_cast<int>(components_),
                        static_cast<int>(n_ * probes), xflat.data(),
                        static_cast<int>(n_ * probes), wflat.data(),
                        static_cast<int>(n_ * probes), cross.data(),
                        static_cast<int>(components_)
                    );
                    for (std::size_t left = 0; left < components_; ++left) {
                        for (std::size_t right = 0; right < components_; ++right) {
                            group_restricted_[
                                (group * components_ + left) * components_ + right
                            ] += cross[left + components_ * right] /
                                 static_cast<double>(bt_);
                        }
                    }
                }
                double* group_matrix = group_restricted_.data() +
                    group * components_ * components_;
                symmetrize_pointer(group_matrix, components_);
            }
        }
    }

    void compute_group_direct() {
        std::vector<double> directional_action(
            groups_ * components_ * components_, 0.0
        );
        std::vector<double> directional_genotype(
            groups_ * components_ * components_, 0.0
        );
        std::vector<double> scaled_genotype(n_ * m_);
        for (std::size_t b0 = 0; b0 < bt_; b0 += sample_probe_tile_) {
            const std::size_t probes = std::min(sample_probe_tile_, bt_ - b0);
            for (std::size_t context = 0; context < q_; ++context) {
                for (std::size_t c0 = 0; c0 < components_; c0 += action_tile_) {
                    const std::size_t targets = std::min(action_tile_, components_ - c0);
                    std::vector<double> action(n_ * targets * probes);
                    std::vector<double> action_scaled(n_ * targets * probes);
                    for (std::size_t target = 0; target < targets; ++target) {
                        for (std::size_t probe = 0; probe < probes; ++probe) {
                            const double* raw = raw_actions_.data() + n_ *
                                ((c0 + target) * bt_ + b0 + probe);
                            double* plain = action.data() + n_ *
                                (target * probes + probe);
                            double* scaled = action_scaled.data() + n_ *
                                (target * probes + probe);
                            for (std::size_t row = 0; row < n_; ++row) {
                                plain[row] = raw[row];
                                scaled[row] = context_basis_[row + n_ * context] *
                                              raw[row];
                            }
                        }
                    }
                    for (std::size_t variant = 0; variant < m_; ++variant) {
                        for (std::size_t row = 0; row < n_; ++row)
                            scaled_genotype[row + n_ * variant] =
                                context_basis_[row + n_ * context] *
                                genotype_[row + n_ * variant];
                    }
                    std::vector<double> contraction_action(m_ * targets * probes);
                    std::vector<double> contraction_genotype(m_ * targets * probes);
                    protected_tn(
                        ContextualDenseSemanticV1::direct_grouped_tn,
                        static_cast<int>(m_), static_cast<int>(targets * probes),
                        static_cast<int>(n_), genotype_.data(),
                        static_cast<int>(n_), action_scaled.data(),
                        static_cast<int>(n_), contraction_action.data(),
                        static_cast<int>(m_)
                    );
                    protected_tn(
                        ContextualDenseSemanticV1::direct_grouped_tn,
                        static_cast<int>(m_), static_cast<int>(targets * probes),
                        static_cast<int>(n_), scaled_genotype.data(),
                        static_cast<int>(n_), action.data(),
                        static_cast<int>(n_), contraction_genotype.data(),
                        static_cast<int>(m_)
                    );
                    for (std::size_t index = 0; index < contraction_action.size(); ++index)
                        direct_placement_max_abs_ = std::max(
                            direct_placement_max_abs_,
                            std::abs(contraction_action[index] -
                                     contraction_genotype[index])
                        );
                    for (std::size_t source_component = 0;
                         source_component < components_; ++source_component) {
                        const std::size_t annotation = static_cast<std::size_t>(
                            component_annotation_[source_component]
                        );
                        const std::size_t pair = static_cast<std::size_t>(
                            component_pair_[source_component]
                        );
                        const std::size_t left = static_cast<std::size_t>(pair_q_[pair]);
                        const std::size_t right = static_cast<std::size_t>(pair_r_[pair]);
                        if (context != left && context != right)
                            continue;
                        for (std::size_t target = 0; target < targets; ++target) {
                            const std::size_t target_component = c0 + target;
                            for (std::size_t probe = 0; probe < probes; ++probe) {
                                const double* contraction_a = contraction_action.data() +
                                    m_ * (target * probes + probe);
                                const double* contraction_g = contraction_genotype.data() +
                                    m_ * (target * probes + probe);
                                const double* source = nullptr;
                                if (context == left)
                                    source = source_scores_.data() + m_ *
                                        (right * bt_ + b0 + probe);
                                else if (left != right && context == right)
                                    source = source_scores_.data() + m_ *
                                        (left * bt_ + b0 + probe);
                                if (source == nullptr)
                                    continue;
                                for (std::size_t variant = 0; variant < m_; ++variant) {
                                    const std::size_t group = static_cast<std::size_t>(
                                        group_index_[variant]
                                    );
                                    const double coefficient = annotation_value(
                                        variant, annotation
                                    ) * source[variant];
                                    const std::size_t index =
                                        (group * components_ + source_component) *
                                        components_ + target_component;
                                    directional_action[index] +=
                                        coefficient * contraction_a[variant];
                                    directional_genotype[index] +=
                                        coefficient * contraction_g[variant];
                                }
                            }
                        }
                    }
                }
            }
        }
        const double denominator = 2.0 * static_cast<double>(bt_);
        for (std::size_t group = 0; group < groups_; ++group) {
            for (std::size_t left = 0; left < components_; ++left) {
                for (std::size_t right = 0; right < components_; ++right) {
                    const std::size_t index =
                        (group * components_ + left) * components_ + right;
                    const std::size_t transpose =
                        (group * components_ + right) * components_ + left;
                    group_direct_action_[index] =
                        (directional_action[index] + directional_action[transpose]) /
                        denominator;
                    group_direct_genotype_[index] =
                        (directional_genotype[index] + directional_genotype[transpose]) /
                        denominator;
                }
            }
        }
    }

    void compute_same_person() {
        std::vector<double> sums(n_ * components_, 0.0);
        std::vector<double> within(components_ * components_, 0.0);
        for (std::size_t v0 = 0; v0 < bd_; v0 += variant_probe_tile_) {
            const std::size_t probes = std::min(variant_probe_tile_, bd_ - v0);
            observed_variant_probe_coverage_ += probes;
            std::vector<double> right(m_ * k_ * probes);
            for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                for (std::size_t probe = 0; probe < probes; ++probe) {
                    double* column = right.data() + m_ *
                        (annotation * probes + probe);
                    for (std::size_t variant = 0; variant < m_; ++variant) {
                        const double weight = annotation_value(
                            variant, annotation
                        );
                        const double square_root_weight =
                            strict_annotation_ids_.empty()
                                ? std::sqrt(weight) : weight;
                        column[variant] = square_root_weight *
                            variant_probes_[variant + m_ * (v0 + probe)];
                    }
                }
            }
            std::vector<double> base(n_ * k_ * probes);
            protected_nn(
                ContextualDenseSemanticV1::same_person_target_nn,
                static_cast<int>(n_), static_cast<int>(k_ * probes),
                static_cast<int>(m_), genotype_.data(), static_cast<int>(n_),
                right.data(), static_cast<int>(m_), base.data(),
                static_cast<int>(n_)
            );
            std::vector<double> contextual(n_ * k_ * q_ * probes);
            for (std::size_t annotation = 0; annotation < k_; ++annotation) {
                for (std::size_t context = 0; context < q_; ++context) {
                    for (std::size_t probe = 0; probe < probes; ++probe) {
                        double* destination = contextual.data() + n_ *
                            ((annotation * q_ + context) * probes + probe);
                        const double* source = base.data() + n_ *
                            (annotation * probes + probe);
                        for (std::size_t row = 0; row < n_; ++row)
                            destination[row] = context_basis_[row + n_ * context] *
                                               source[row];
                    }
                }
            }
            std::vector<double> projected;
            project_panel(
                contextual, k_ * q_ * probes,
                ContextualDenseSemanticV1::same_person_projection_tn,
                ContextualDenseSemanticV1::same_person_projection_nn,
                projected
            );
            std::vector<double> gflat(n_ * probes * components_);
            for (std::size_t component = 0; component < components_; ++component) {
                const std::size_t annotation = static_cast<std::size_t>(
                    component_annotation_[component]
                );
                const std::size_t pair = static_cast<std::size_t>(
                    component_pair_[component]
                );
                const std::size_t left = static_cast<std::size_t>(pair_q_[pair]);
                const std::size_t right_context = static_cast<std::size_t>(pair_r_[pair]);
                const double eta = static_cast<double>(pair_eta_[pair]);
                const double mass = annotation_masses_[annotation];
                for (std::size_t probe = 0; probe < probes; ++probe) {
                    const double* left_values = projected.data() + n_ *
                        ((annotation * q_ + left) * probes + probe);
                    const double* right_values = projected.data() + n_ *
                        ((annotation * q_ + right_context) * probes + probe);
                    double* destination = gflat.data() + n_ *
                        (probe + probes * component);
                    for (std::size_t row = 0; row < n_; ++row) {
                        const double value = eta * left_values[row] *
                                             right_values[row] / mass;
                        destination[row] = value;
                        sums[row + n_ * component] += value;
                    }
                }
            }
            std::vector<double> local(components_ * components_);
            protected_tn(
                ContextualDenseSemanticV1::same_person_gram_tn,
                static_cast<int>(components_), static_cast<int>(components_),
                static_cast<int>(n_ * probes), gflat.data(),
                static_cast<int>(n_ * probes), gflat.data(),
                static_cast<int>(n_ * probes), local.data(),
                static_cast<int>(components_)
            );
            for (std::size_t left = 0; left < components_; ++left) {
                for (std::size_t right_index = 0; right_index < components_;
                     ++right_index) {
                    within[left * components_ + right_index] +=
                        local[left + components_ * right_index];
                }
            }
        }
        std::vector<double> cross(components_ * components_);
        protected_tn(
            ContextualDenseSemanticV1::same_person_gram_tn,
            static_cast<int>(components_), static_cast<int>(components_),
            static_cast<int>(n_), sums.data(), static_cast<int>(n_),
            sums.data(), static_cast<int>(n_), cross.data(),
            static_cast<int>(components_)
        );
        const double denominator = static_cast<double>(bd_) *
                                   static_cast<double>(bd_ - 1);
        for (std::size_t left = 0; left < components_; ++left) {
            for (std::size_t right = 0; right < components_; ++right) {
                same_person_[left * components_ + right] =
                    (cross[left + components_ * right] -
                     within[left * components_ + right]) / denominator;
            }
        }
        same_person_pre_symmetry_ = symmetry_error(same_person_, components_);
        symmetrize(same_person_, components_);
    }

    void compute_trait() {
        // The phenotype projection is a narrow fixed-rank operation.  Keep it
        // outside the wide trait-feature semantic ledger so occurrence-one
        // fault injection always reaches a Q-by-variant feature panel.
        std::vector<double> phenotype = phenotype_raw_;
        std::vector<double> phenotype_coefficients(rank_, 0.0);
        for (std::size_t basis = 0; basis < rank_; ++basis)
            for (std::size_t row = 0; row < n_; ++row)
                phenotype_coefficients[basis] +=
                    fixed_basis_[row + n_ * basis] * phenotype[row];
        for (std::size_t row = 0; row < n_; ++row)
            for (std::size_t basis = 0; basis < rank_; ++basis)
                phenotype[row] -= fixed_basis_[row + n_ * basis] *
                                  phenotype_coefficients[basis];
        double sum_squares = 0.0;
        for (double value : phenotype)
            sum_squares += value * value;
        if (!(sum_squares > 0.0))
            throw std::runtime_error(
                "ContextualBlockExecutorV1 validation: phenotype has zero "
                "projected norm"
            );
        const double residual_rank = static_cast<double>(n_ - rank_);
        const double scale = std::sqrt(residual_rank / sum_squares);
        for (double& value : phenotype)
            value *= scale;
        normalized_phenotype_ = phenotype;

        std::vector<double> score_right(n_ * q_);
        for (std::size_t context = 0; context < q_; ++context) {
            for (std::size_t row = 0; row < n_; ++row)
                score_right[row + n_ * context] =
                    context_basis_[row + n_ * context] * phenotype[row];
        }
        observed_trait_variant_visits_ = 0;
        for (std::size_t v0 = 0; v0 < m_; v0 += variant_block_) {
            const std::size_t variants = std::min(variant_block_, m_ - v0);
            observed_trait_variant_visits_ += variants;
            std::vector<double> scores(variants * q_);
            protected_tn(
                ContextualDenseSemanticV1::trait_score_tn,
                static_cast<int>(variants), static_cast<int>(q_),
                static_cast<int>(n_), genotype_.data() + n_ * v0,
                static_cast<int>(n_), score_right.data(), static_cast<int>(n_),
                scores.data(), static_cast<int>(variants)
            );
            std::vector<double> raw_features(n_ * q_ * variants);
            for (std::size_t context = 0; context < q_; ++context) {
                for (std::size_t variant = 0; variant < variants; ++variant) {
                    double* column = raw_features.data() + n_ *
                        (context * variants + variant);
                    const double* genotype = genotype_.data() + n_ *
                        (v0 + variant);
                    for (std::size_t row = 0; row < n_; ++row)
                        column[row] = context_basis_[row + n_ * context] *
                                      genotype[row];
                }
            }
            const std::size_t feature_columns = q_ * variants;
            std::vector<double> features(n_ * feature_columns);
            for (std::size_t column0 = 0; column0 < feature_columns;
                 column0 += trait_feature_tile_) {
                const std::size_t columns = std::min(
                    trait_feature_tile_, feature_columns - column0
                );
                std::vector<double> feature_tile(n_ * columns);
                std::copy_n(
                    raw_features.data() + n_ * column0, n_ * columns,
                    feature_tile.data()
                );
                std::vector<double> projected_tile;
                project_panel(
                    feature_tile, columns,
                    ContextualDenseSemanticV1::trait_feature_projection_tn,
                    ContextualDenseSemanticV1::trait_feature_projection_nn,
                    projected_tile
                );
                std::copy_n(
                    projected_tile.data(), n_ * columns,
                    features.data() + n_ * column0
                );
            }
            for (std::size_t variant = 0; variant < variants; ++variant) {
                const std::size_t global_variant = v0 + variant;
                const std::size_t group = static_cast<std::size_t>(
                    group_index_[global_variant]
                );
                for (std::size_t component = 0; component < components_; ++component) {
                    const std::size_t annotation = static_cast<std::size_t>(
                        component_annotation_[component]
                    );
                    const std::size_t pair = static_cast<std::size_t>(
                        component_pair_[component]
                    );
                    const std::size_t left = static_cast<std::size_t>(pair_q_[pair]);
                    const std::size_t right = static_cast<std::size_t>(pair_r_[pair]);
                    const double weight = static_cast<double>(pair_eta_[pair]) *
                        annotation_value(global_variant, annotation);
                    const double rhs = weight * scores[variant + variants * left] *
                                       scores[variant + variants * right];
                    double trace = 0.0;
                    std::vector<double> residual(h_, 0.0);
                    const double* left_feature = features.data() + n_ *
                        (left * variants + variant);
                    const double* right_feature = features.data() + n_ *
                        (right * variants + variant);
                    for (std::size_t row = 0; row < n_; ++row) {
                        const double cross = left_feature[row] * right_feature[row];
                        trace += cross;
                        for (std::size_t residual_component = 0;
                             residual_component < h_; ++residual_component) {
                            residual[residual_component] += cross * residual_basis_[
                                row + n_ * residual_component
                            ];
                        }
                    }
                    const std::size_t grouped = group * components_ + component;
                    group_rhs_[grouped] += rhs;
                    group_trace_[grouped] += weight * trace;
                    for (std::size_t residual_component = 0;
                         residual_component < h_; ++residual_component) {
                        group_genetic_residual_[
                            grouped * h_ + residual_component
                        ] += weight * residual[residual_component];
                    }
                }
            }
        }
        for (std::size_t component = 0; component < components_; ++component) {
            const std::size_t annotation = static_cast<std::size_t>(
                component_annotation_[component]
            );
            for (std::size_t group = 0; group < groups_; ++group) {
                const std::size_t grouped = group * components_ + component;
                genetic_rhs_[component] += group_rhs_[grouped];
                genetic_traces_[component] += group_trace_[grouped];
                for (std::size_t residual_component = 0;
                     residual_component < h_; ++residual_component) {
                    genetic_residual_[component * h_ + residual_component] +=
                        group_genetic_residual_[grouped * h_ + residual_component];
                }
            }
            genetic_rhs_[component] /= annotation_masses_[annotation];
            genetic_traces_[component] /= annotation_masses_[annotation];
            for (std::size_t residual_component = 0;
                 residual_component < h_; ++residual_component)
                genetic_residual_[component * h_ + residual_component] /=
                    annotation_masses_[annotation];
        }
    }

    void compute_residual_moments() {
        std::vector<double> leverage(n_, 0.0);
        for (std::size_t row = 0; row < n_; ++row) {
            for (std::size_t basis = 0; basis < rank_; ++basis) {
                const double value = fixed_basis_[row + n_ * basis];
                leverage[row] += value * value;
            }
        }
        std::vector<double> small(h_ * rank_ * rank_, 0.0);
        for (std::size_t residual = 0; residual < h_; ++residual) {
            for (std::size_t left = 0; left < rank_; ++left) {
                for (std::size_t right = 0; right < rank_; ++right) {
                    double value = 0.0;
                    for (std::size_t row = 0; row < n_; ++row)
                        value += residual_basis_[row + n_ * residual] *
                                 fixed_basis_[row + n_ * left] *
                                 fixed_basis_[row + n_ * right];
                    small[(residual * rank_ + left) * rank_ + right] = value;
                }
            }
            for (std::size_t row = 0; row < n_; ++row) {
                const double d = residual_basis_[row + n_ * residual];
                residual_rhs_[residual] += d * normalized_phenotype_[row] *
                                           normalized_phenotype_[row];
                residual_traces_[residual] += d * (1.0 - leverage[row]);
            }
        }
        for (std::size_t left = 0; left < h_; ++left) {
            for (std::size_t right = 0; right < h_; ++right) {
                double value = 0.0;
                for (std::size_t row = 0; row < n_; ++row) {
                    const double product = residual_basis_[row + n_ * left] *
                                           residual_basis_[row + n_ * right];
                    value += product * (1.0 - 2.0 * leverage[row]);
                }
                for (std::size_t first = 0; first < rank_; ++first) {
                    for (std::size_t second = 0; second < rank_; ++second)
                        value += small[(left * rank_ + first) * rank_ + second] *
                                 small[(right * rank_ + second) * rank_ + first];
                }
                residual_gram_[left * h_ + right] = value;
            }
        }
        residual_gram_pre_symmetry_ = symmetry_error(residual_gram_, h_);
        symmetrize(residual_gram_, h_);
    }

    static double symmetry_error(
        const std::vector<double>& matrix, std::size_t size
    ) {
        double result = 0.0;
        for (std::size_t left = 0; left < size; ++left)
            for (std::size_t right = 0; right < size; ++right)
                result = std::max(
                    result,
                    std::abs(matrix[left * size + right] -
                             matrix[right * size + left])
                );
        return result;
    }

    static void symmetrize_pointer(double* matrix, std::size_t size) {
        for (std::size_t left = 0; left < size; ++left) {
            for (std::size_t right = left + 1; right < size; ++right) {
                const double value = 0.5 * (
                    matrix[left * size + right] +
                    matrix[right * size + left]
                );
                matrix[left * size + right] = value;
                matrix[right * size + left] = value;
            }
        }
    }

    static void symmetrize(std::vector<double>& matrix, std::size_t size) {
        symmetrize_pointer(matrix.data(), size);
    }

    static double maximum_difference(
        const std::vector<double>& left, const std::vector<double>& right
    ) {
        if (left.size() != right.size())
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: comparison shape mismatch"
            );
        double result = 0.0;
        for (std::size_t index = 0; index < left.size(); ++index)
            result = std::max(result, std::abs(left[index] - right[index]));
        return result;
    }

    void finalize_and_validate() {
        check_inputs_unchanged();
        direct_action_restricted_max_abs_ = maximum_difference(
            group_direct_action_, group_restricted_
        );
        direct_genotype_restricted_max_abs_ = maximum_difference(
            group_direct_genotype_, group_restricted_
        );
        group_reconstruction_max_abs_ = 0.0;
        trait_group_reconstruction_max_abs_ = 0.0;
        mass_reconstruction_max_abs_ = 0.0;
        for (std::size_t left = 0; left < components_; ++left) {
            for (std::size_t right = 0; right < components_; ++right) {
                double sum = 0.0;
                for (std::size_t group = 0; group < groups_; ++group)
                    sum += group_restricted_[
                        (group * components_ + left) * components_ + right
                    ];
                group_reconstruction_max_abs_ = std::max(
                    group_reconstruction_max_abs_,
                    std::abs(sum - raw_gram_numerator_[left * components_ + right])
                );
            }
        }
        std::uint64_t reconstructed_variant_count = 0;
        for (std::size_t group = 0; group < groups_; ++group)
            reconstructed_variant_count += static_cast<std::uint64_t>(
                group_variant_counts_[group]
            );
        if (reconstructed_variant_count != m_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: group variant counts "
                "do not reconstruct M"
            );
        for (std::size_t annotation = 0; annotation < k_; ++annotation) {
            double reconstructed_mass = 0.0;
            for (std::size_t group = 0; group < groups_; ++group)
                reconstructed_mass += group_annotation_masses_[
                    group * k_ + annotation
                ];
            mass_reconstruction_max_abs_ = std::max(
                mass_reconstruction_max_abs_,
                std::abs(reconstructed_mass - annotation_masses_[annotation])
            );
        }
        for (std::size_t component = 0; component < components_; ++component) {
            const std::size_t annotation = static_cast<std::size_t>(
                component_annotation_[component]
            );
            double rhs = 0.0;
            double trace = 0.0;
            for (std::size_t group = 0; group < groups_; ++group) {
                rhs += group_rhs_[group * components_ + component];
                trace += group_trace_[group * components_ + component];
            }
            trait_group_reconstruction_max_abs_ = std::max(
                trait_group_reconstruction_max_abs_,
                std::abs(rhs - annotation_masses_[annotation] *
                               genetic_rhs_[component])
            );
            trait_group_reconstruction_max_abs_ = std::max(
                trait_group_reconstruction_max_abs_,
                std::abs(trace - annotation_masses_[annotation] *
                                 genetic_traces_[component])
            );
            for (std::size_t residual = 0; residual < h_; ++residual) {
                double cross = 0.0;
                for (std::size_t group = 0; group < groups_; ++group)
                    cross += group_genetic_residual_[
                        (group * components_ + component) * h_ + residual
                    ];
                trait_group_reconstruction_max_abs_ = std::max(
                    trait_group_reconstruction_max_abs_,
                    std::abs(cross - annotation_masses_[annotation] *
                                     genetic_residual_[component * h_ + residual])
                );
            }
        }
        double scientific_scale = 1.0;
        for (double value : raw_gram_numerator_)
            scientific_scale = std::max(scientific_scale, std::abs(value));
        const double tolerance = 1.0e-9 + 1.0e-11 * scientific_scale;
        if (direct_action_restricted_max_abs_ > tolerance ||
            direct_genotype_restricted_max_abs_ > tolerance ||
            group_reconstruction_max_abs_ > tolerance ||
            trait_group_reconstruction_max_abs_ > tolerance ||
            mass_reconstruction_max_abs_ > tolerance)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: grouped algorithm or "
                "reconstruction mismatch"
            );
        if (maximum_leakage_ > 2.0e-9)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: projection leakage exceeds tolerance"
            );
        if (gram_pre_symmetry_ > tolerance ||
            same_person_pre_symmetry_ > tolerance ||
            residual_gram_pre_symmetry_ > tolerance)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: pre-publication symmetry "
                "error exceeds tolerance"
            );
        if (observed_sample_probe_coverage_ != bt_ ||
            observed_variant_probe_coverage_ != bd_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: explicit probe coverage mismatch"
            );
        if (observed_trait_variant_visits_ != m_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: trait pass was not logical one-pass"
            );
        for (std::size_t index = 0; index < kContextualDenseSemanticCountV1;
             ++index) {
            if (observed_calls_[index] != estimate_.calls[index])
                throw std::runtime_error(
                    std::string("ContextualBlockExecutorV1 integrity: semantic call ") +
                    kContextualDenseSemanticNamesV1[index] +
                    " differs from preflight"
                );
        }
        if (fault_mode_ != "none" && !fault_consumed_)
            throw std::runtime_error(
                "ContextualBlockExecutorV1 integrity: targeted fault was not consumed"
            );
        for (const auto& output : {
                &gram_, &raw_gram_numerator_, &same_person_,
                &group_restricted_, &group_direct_action_,
                &group_direct_genotype_, &genetic_rhs_, &genetic_traces_,
                &genetic_residual_, &residual_rhs_, &residual_traces_,
                &residual_gram_, &group_rhs_, &group_trace_,
                &group_genetic_residual_
            }) {
            if (!finite_panel(output->data(), output->size()))
                throw std::runtime_error(
                    "ContextualBlockExecutorV1 integrity: non-finite compact output"
                );
        }
    }

    void release_execution_scratch() {
        std::vector<double>().swap(source_scores_);
        std::vector<double>().swap(raw_actions_);
        std::vector<double>().swap(retry_buffer_);
        std::vector<double>().swap(witness_buffer_);
        std::vector<double>().swap(fallback_buffer_);
        std::vector<double>().swap(normalized_phenotype_);
        scratch_released_ = true;
        record_event("scratch_release", "", "released_before_publication");
    }

    nb::dict publish() const {
        nb::dict result;
        result["reference_n"] = n_;
        result["residual_rank"] = n_ - rank_;
        result["annotation_names"] = annotation_names_;
        result["group_names"] = group_names_;
        result["annotation_mode"] = annotation_mode_;
        result["genotype_scale_policy"] = "pre_scaled_dense_v1";
        result["numeric_policy"] = "fp64_v1";
        result["deletion_policy"] = "approximate_summary_only_v1";
        result["grouped_encoding"] = "grouped_unnormalized_dense_v1";
        result["grouped_algorithms_executed"] = std::vector<std::string>{
            "direct_grouped_tn_action_scaled_v1",
            "direct_grouped_tn_genotype_scaled_v1",
            "group_restricted_action_v1"
        };
        result["selected_grouped_algorithm"] =
            "group_restricted_action_v1";
        result["gram"] = contextual_copy_mat2_v1(
            gram_, components_, components_
        );
        result["raw_gram_numerator"] = contextual_copy_mat2_v1(
            raw_gram_numerator_, components_, components_
        );
        result["same_person"] = contextual_copy_mat2_v1(
            same_person_, components_, components_
        );
        result["group_gram_numerator"] = contextual_copy_arr3_v1(
            group_restricted_, groups_, components_, components_
        );
        result["group_gram_numerator_restricted"] = contextual_copy_arr3_v1(
            group_restricted_, groups_, components_, components_
        );
        result["group_gram_numerator_direct_action_scaled"] =
            contextual_copy_arr3_v1(
                group_direct_action_, groups_, components_, components_
            );
        result["group_gram_numerator_direct_genotype_scaled"] =
            contextual_copy_arr3_v1(
                group_direct_genotype_, groups_, components_, components_
            );
        result["annotation_masses"] = contextual_copy_vec_v1(
            annotation_masses_
        );
        result["group_annotation_masses"] = contextual_copy_mat2_v1(
            group_annotation_masses_, groups_, k_
        );
        result["group_variant_counts"] = contextual_copy_vec_v1(
            group_variant_counts_
        );
        result["genetic_rhs"] = contextual_copy_vec_v1(genetic_rhs_);
        result["genetic_traces"] = contextual_copy_vec_v1(genetic_traces_);
        result["genetic_residual"] = contextual_copy_mat2_v1(
            genetic_residual_, components_, h_
        );
        result["residual_rhs"] = contextual_copy_vec_v1(residual_rhs_);
        result["residual_traces"] = contextual_copy_vec_v1(residual_traces_);
        result["residual_gram"] = contextual_copy_mat2_v1(
            residual_gram_, h_, h_
        );
        result["group_rhs_unnormalized_num"] = contextual_copy_mat2_v1(
            group_rhs_, groups_, components_
        );
        result["group_trace_unnormalized_num"] = contextual_copy_mat2_v1(
            group_trace_, groups_, components_
        );
        result["group_genetic_residual_num"] = contextual_copy_arr3_v1(
            group_genetic_residual_, groups_, components_, h_
        );
        result["pair_q"] = contextual_copy_vec_v1(pair_q_);
        result["pair_r"] = contextual_copy_vec_v1(pair_r_);
        result["pair_eta"] = contextual_copy_vec_v1(pair_eta_);
        result["component_annotation"] = contextual_copy_vec_v1(
            component_annotation_
        );
        result["component_pair"] = contextual_copy_vec_v1(component_pair_);

        nb::dict units;
        units["gram"] = "normalized_by_source_and_target_annotation_mass_v1";
        units["raw_gram_numerator"] =
            "raw_source_target_annotation_mass_product_fixed_probe_v1";
        units["group_gram_numerator"] =
            "raw_source_target_annotation_mass_product_fixed_probe_v1";
        units["genetic_trait"] = "normalized_by_source_annotation_mass_v1";
        units["group_trait"] = "raw_source_annotation_mass_v1";
        result["units"] = units;

        nb::dict diagnostics;
        diagnostics["gram_pre_symmetry_max_abs"] = gram_pre_symmetry_;
        diagnostics["same_person_pre_symmetry_max_abs"] =
            same_person_pre_symmetry_;
        diagnostics["residual_gram_pre_symmetry_max_abs"] =
            residual_gram_pre_symmetry_;
        diagnostics["group_reconstruction_max_abs"] =
            group_reconstruction_max_abs_;
        diagnostics["trait_group_reconstruction_max_abs"] =
            trait_group_reconstruction_max_abs_;
        diagnostics["mass_reconstruction_max_abs"] =
            mass_reconstruction_max_abs_;
        diagnostics["direct_action_restricted_max_abs"] =
            direct_action_restricted_max_abs_;
        diagnostics["direct_genotype_restricted_max_abs"] =
            direct_genotype_restricted_max_abs_;
        diagnostics["direct_placement_max_abs"] = direct_placement_max_abs_;
        diagnostics["maximum_projection_leakage"] = maximum_leakage_;
        diagnostics["trait_variant_visits"] = observed_trait_variant_visits_;
        diagnostics["trait_logical_passes"] = 1;
        diagnostics["sample_probe_coverage"] = observed_sample_probe_coverage_;
        diagnostics["variant_probe_coverage"] = observed_variant_probe_coverage_;
        diagnostics["decode_passes"] = 1;
        diagnostics["descriptor_decode_passes"] = 0;
        diagnostics["group_tile_batches"] =
            (groups_ + group_tile_ - 1) / group_tile_;
        diagnostics["inputs_unchanged"] = true;
        diagnostics["semantic_call_ledger_exact"] = true;
        diagnostics["expected_calls_verified"] = true;
        diagnostics["group_reconstruction_verified"] = true;
        diagnostics["scratch_released"] = scratch_released_;
        diagnostics["scratch_released_before_publication"] = scratch_released_;
        diagnostics["compact_outputs_only"] = true;
        diagnostics["strict_disjoint_optimized_path"] =
            !strict_annotation_ids_.empty();
        nb::dict observed;
        nb::dict expected;
        for (std::size_t index = 0; index < kContextualDenseSemanticCountV1;
             ++index) {
            observed[kContextualDenseSemanticNamesV1[index]] =
                observed_calls_[index];
            expected[kContextualDenseSemanticNamesV1[index]] =
                estimate_.calls[index];
        }
        diagnostics["protected_call_counts"] = observed;
        diagnostics["expected_protected_call_counts"] = expected;
        result["diagnostics"] = diagnostics;

        nb::dict telemetry;
        telemetry["capacity"] = telemetry_capacity_;
        telemetry["required_capacity"] = estimate_.required_telemetry_capacity;
        telemetry["observed_events"] = events_.size();
        telemetry["injection_count"] = injection_count_;
        telemetry["repair_count"] = repair_count_;
        telemetry["retry_count"] = retry_count_;
        telemetry["fallback_count"] = fallback_count_;
        telemetry["complete_without_drop"] = true;
        telemetry["fault_operation"] = fault_operation_;
        telemetry["fault_mode"] = fault_mode_;
        telemetry["protected_call_counts"] = observed;
        nb::list event_list;
        for (const auto& event : events_) {
            nb::dict item;
            item["event_class"] = event.event_class;
            item["phase"] = event.phase;
            item["operation"] = event.operation;
            item["resolution"] = event.resolution;
            event_list.append(item);
        }
        telemetry["events"] = event_list;
        result["telemetry"] = telemetry;
        result["admission"] = preflight();
        result["state"] = state_;
        result["lifecycle"] = state_;
        result["backend"] = "pre_scaled_dense_differential_v1";
        result["native_api_version"] = 1;
        return result;
    }

    std::size_t n_ = 0, m_ = 0, rank_ = 0, q_ = 0, k_ = 0;
    std::size_t groups_ = 0, bt_ = 0, bd_ = 0, h_ = 0;
    std::size_t pairs_ = 0, components_ = 0;
    std::size_t variant_block_ = 0, sample_probe_tile_ = 0;
    std::size_t variant_probe_tile_ = 0, action_tile_ = 0;
    std::size_t annotation_tile_ = 0, context_tile_ = 0, group_tile_ = 0;
    std::size_t trait_feature_tile_ = 0;
    std::vector<std::string> annotation_names_, group_names_;
    std::string annotation_mode_;
    int threads_ = 1;
    std::string fault_operation_, fault_mode_;
    std::uint64_t fault_occurrence_ = 1;
    std::size_t fault_index_ = kContextualDenseSemanticCountV1;
    std::uint64_t fault_seen_occurrences_ = 0;
    bool fault_consumed_ = false;
    std::uint64_t workspace_cap_bytes_ = 0, telemetry_capacity_ = 0;
    ContextualDensePreflightNumbersV1 estimate_;
    std::string state_ = "validation_and_sealing";
    const char* phase_ = "admission";

    std::vector<double> genotype_, fixed_basis_, context_basis_;
    std::vector<double> annotation_weights_, sample_probes_, variant_probes_;
    std::vector<double> phenotype_raw_, residual_basis_;
    std::vector<std::int64_t> group_index_;
    std::vector<std::size_t> strict_annotation_ids_;
    std::uint64_t input_fingerprint_ = 0;

    std::vector<std::int64_t> pair_q_, pair_r_, pair_eta_;
    std::vector<std::int64_t> component_annotation_, component_pair_;
    std::vector<double> annotation_masses_, group_annotation_masses_;
    std::vector<std::int64_t> group_variant_counts_;

    std::vector<double> retry_buffer_, witness_buffer_, fallback_buffer_;
    std::array<std::uint64_t, kContextualDenseSemanticCountV1> observed_calls_{};
    std::vector<ContextualDenseEventV1> events_;
    std::uint64_t injection_count_ = 0, repair_count_ = 0;
    std::uint64_t retry_count_ = 0, fallback_count_ = 0;

    std::vector<double> source_scores_, raw_actions_;
    std::vector<double> gram_, raw_gram_numerator_, same_person_;
    std::vector<double> group_restricted_, group_direct_action_;
    std::vector<double> group_direct_genotype_;
    std::vector<double> genetic_rhs_, genetic_traces_, genetic_residual_;
    std::vector<double> residual_rhs_, residual_traces_, residual_gram_;
    std::vector<double> group_rhs_, group_trace_, group_genetic_residual_;
    std::vector<double> normalized_phenotype_;

    std::uint64_t observed_variant_blocks_ = 0;
    std::uint64_t observed_trait_variant_visits_ = 0;
    std::uint64_t observed_sample_probe_coverage_ = 0;
    std::uint64_t observed_variant_probe_coverage_ = 0;
    double gram_pre_symmetry_ = 0.0, same_person_pre_symmetry_ = 0.0;
    double residual_gram_pre_symmetry_ = 0.0;
    double group_reconstruction_max_abs_ = 0.0;
    double trait_group_reconstruction_max_abs_ = 0.0;
    double mass_reconstruction_max_abs_ = 0.0;
    double direct_action_restricted_max_abs_ = 0.0;
    double direct_genotype_restricted_max_abs_ = 0.0;
    double direct_placement_max_abs_ = 0.0;
    double maximum_leakage_ = 0.0;
    bool scratch_released_ = false;
};

void test_contextual_checked_layout_overflow_v1(
    std::uint64_t first, std::uint64_t second, std::uint64_t item_bytes
) {
    const std::uint64_t elements = contextual_mul_u64_v1(
        first, second, "test layout elements"
    );
    (void)contextual_mul_u64_v1(
        elements, item_bytes, "test layout bytes"
    );
}

nb::dict contextual_dense_preflight_v1(
    std::uint64_t n, std::uint64_t m, std::uint64_t fixed_rank,
    std::uint64_t q, std::uint64_t k, std::uint64_t groups,
    std::uint64_t sample_probes, std::uint64_t variant_probes,
    std::uint64_t residual_components,
    std::uint64_t variant_block, std::uint64_t sample_probe_tile,
    std::uint64_t variant_probe_tile, std::uint64_t action_tile,
    std::uint64_t annotation_tile, std::uint64_t context_tile,
    std::uint64_t trait_feature_tile,
    bool fault_enabled
) {
    return contextual_dense_preflight_dict_v1(
        contextual_dense_preflight_numbers_v1(
            n, m, fixed_rank, q, k, groups, sample_probes, variant_probes,
            residual_components, variant_block, sample_probe_tile,
            variant_probe_tile, action_tile, annotation_tile, context_tile,
            trait_feature_tile, fault_enabled
        )
    );
}

void bind_contextual_dense_v1(nb::module_& module) {
    module.def(
        "_test_contextual_checked_layout_overflow",
        &test_contextual_checked_layout_overflow_v1,
        nb::arg("first"), nb::arg("second"), nb::arg("item_bytes")
    );
    module.def(
        "contextual_dense_preflight_v1", &contextual_dense_preflight_v1,
        nb::arg("n"), nb::arg("m"), nb::arg("fixed_rank"), nb::arg("q"),
        nb::arg("k"), nb::arg("groups"), nb::arg("sample_probes"),
        nb::arg("variant_probes"), nb::arg("residual_components"),
        nb::arg("variant_block") = 0,
        nb::arg("sample_probe_tile") = 0,
        nb::arg("variant_probe_tile") = 0,
        nb::arg("action_tile") = 0,
        nb::arg("annotation_tile") = 0,
        nb::arg("context_tile") = 0,
        nb::arg("trait_feature_tile") = 0,
        nb::arg("fault_enabled") = false,
        "Checked dimension-only admission estimate for the contextual dense V1 executor."
    );
    module.def("contextual_dense_semantic_operations_v1", []() {
        std::vector<std::string> values;
        values.reserve(kContextualDenseSemanticCountV1);
        for (const char* value : kContextualDenseSemanticNamesV1)
            values.emplace_back(value);
        return values;
    });
    nb::class_<ContextualBlockExecutorV1>(module, "ContextualBlockExecutorV1")
        .def(
            nb::init<
                nb_mat2f_ro<double>, nb_mat2f_ro<double>, nb_mat2f_ro<double>,
                nb_mat2f_ro<double>, nb_vec1_ro<std::int64_t>,
                nb_mat2f_ro<double>, nb_mat2f_ro<double>, nb_vec1_ro<double>,
                nb_mat2f_ro<double>, std::vector<std::string>,
                std::vector<std::string>, std::string,
                std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                int, std::uint64_t, std::uint64_t, std::string, std::string,
                std::uint64_t
            >(),
            nb::arg("genotype"), nb::arg("fixed_basis"),
            nb::arg("context_basis"), nb::arg("annotation_weights"),
            nb::arg("group_index"), nb::arg("sample_probes"),
            nb::arg("variant_probes"), nb::arg("phenotype_raw"),
            nb::arg("residual_basis"), nb::arg("annotation_names"),
            nb::arg("group_names"),
            nb::arg("annotation_mode") = "generic_nonnegative_weights_v1",
            nb::arg("variant_block") = 0,
            nb::arg("sample_probe_tile") = 0,
            nb::arg("variant_probe_tile") = 0,
            nb::arg("action_tile") = 0,
            nb::arg("annotation_tile") = 0,
            nb::arg("context_tile") = 0,
            nb::arg("group_tile") = 0,
            nb::arg("trait_feature_tile") = 0,
            nb::arg("threads") = 1,
            nb::arg("workspace_cap_bytes") =
                std::numeric_limits<std::uint64_t>::max(),
            nb::arg("telemetry_capacity") =
                std::numeric_limits<std::uint64_t>::max(),
            nb::arg("fault_operation") = "",
            nb::arg("fault_mode") = "none",
            nb::arg("fault_occurrence") = 1,
            "Seal an owned, one-shot, pre-scaled dense contextual differential executor."
        )
        .def("info", &ContextualBlockExecutorV1::info)
        .def("preflight", &ContextualBlockExecutorV1::preflight)
        .def("run", &ContextualBlockExecutorV1::run);
}

}  // namespace summit::context_v1
