// Coordinate updates in a cached, fixed-effect-projected genotype block.
// Candidate models are independent OpenMP work units; SNPs are Gauss-Seidel.
// No genotype decoding or alternative BLAS implementation is introduced here.
// Hard-call affine statistics use exact integer category counts, avoiding
// several N-by-block FP64 temporaries during biobank-scale setup.
static void prediction_hardcall_scale(nb_mat2f_ro<int8_t> raw,
    nb_vec1_rw<double> mean, nb_vec1_rw<double> inverse, int ddof, int threads)
{
    const size_t n = raw.shape(0), b = raw.shape(1);
    if (!b || n <= static_cast<size_t>(ddof) || (ddof != 0 && ddof != 1) ||
        mean.size() != b || inverse.size() != b || threads < 1)
        throw std::runtime_error("Invalid hard-call scale dimensions or controls");
    prediction_no_alias(mean.data(), mean.size(), inverse.data(), inverse.size());
    const auto start = reinterpret_cast<uintptr_t>(raw.data());
    const auto end = start+raw.size();
    for (auto output : {mean, inverse}) {
        const auto address = reinterpret_cast<uintptr_t>(output.data());
        if (address < end && start < address+output.size()*sizeof(double))
            throw std::runtime_error("Hard-call scale output aliases raw input");
    }
    validate_configured_openmp_placement_for_entry();
    nb::gil_scoped_release release;
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
#endif
    for (size_t j = 0; j < b; ++j) {
        uint64_t counts[3] = {0,0,0};
        const int8_t* column = raw.data()+j*n;
        for (size_t i = 0; i < n; ++i) {
            const int v = column[i];
            if (v >= 0 && v <= 2) ++counts[v];
            else if (v != -127) invalid = 1;
        }
        const uint64_t observed = counts[0]+counts[1]+counts[2];
        const double mu = observed ? static_cast<double>(counts[1]+2*counts[2])/observed : 0.;
        double ss = 0.;
        for (int v = 0; v < 3; ++v) {
            const double centered = v-mu;
            ss += counts[v]*centered*centered;
        }
        mean.data()[j] = mu;
        inverse.data()[j] = ss > 0 ? std::sqrt(static_cast<double>(n-ddof)/ss) : 1.;
    }
    if (invalid) throw std::runtime_error("Invalid genotype code in hard-call scale input");
}

static void prediction_mixture_block(
    nb_mat2c_ro<double> gram, nb_mat2c_ro<double> score,
    nb_mat2c_ro<double> posterior_covariance, nb_mat2c_ro<double> log_normalizer,
    nb_mat2c_rw<double> weights, nb_mat2c_rw<double> penalty,
    nb_mat2c_rw<double> metrics, int q, int sweeps, double tolerance,
    bool independent, int threads)
{
    const size_t models = weights.shape(0), d = weights.shape(1);
    if (q < 1 || q > 32 || !d || d % q || !models || sweeps < 1 || threads < 1 ||
        !(tolerance >= 0) || !std::isfinite(tolerance))
        throw std::runtime_error("Invalid mixture block dimensions or controls");
    const size_t b = d / q;
    const size_t components = log_normalizer.shape(1)/b;
    if (gram.shape(0) != d || gram.shape(1) != d || score.shape(0) != models ||
        score.shape(1) != d || posterior_covariance.shape(0) != models ||
        components < 1 || components > 16 || posterior_covariance.shape(1) != b*components*q*q || log_normalizer.shape(0) != models ||
        log_normalizer.shape(1) != components*b || penalty.shape(0) != models ||
        penalty.shape(1) != b || metrics.shape(0) != models || metrics.shape(1) != 2)
        throw std::runtime_error("Mixture block array shapes disagree");
    for (auto input : {gram, score, posterior_covariance, log_normalizer}) {
        prediction_no_alias(input.data(), input.size(), weights.data(), weights.size());
        prediction_no_alias(input.data(), input.size(), penalty.data(), penalty.size());
        prediction_no_alias(input.data(), input.size(), metrics.data(), metrics.size());
        for (size_t i = 0; i < input.size(); ++i)
            if (!std::isfinite(input.data()[i])) throw std::runtime_error("Nonfinite mixture input");
    }
    prediction_no_alias(weights.data(), weights.size(), penalty.data(), penalty.size());
    prediction_no_alias(weights.data(), weights.size(), metrics.data(), metrics.size());
    prediction_no_alias(penalty.data(), penalty.size(), metrics.data(), metrics.size());
    for (size_t i = 0; i < weights.size(); ++i)
        if (!std::isfinite(weights.data()[i])) throw std::runtime_error("Nonfinite mixture weight");
    validate_configured_openmp_placement_for_entry();
    nb::gil_scoped_release release;
    // Coordinate updates need Gram columns. Copy their exact values once so
    // each candidate streams contiguous memory instead of striding by d.
    // Do not assume exact symmetry: callers may have rounding asymmetry.
    std::vector<double> gram_columns(independent ? 0 : d*d);
    if (!independent) {
        constexpr size_t tile = 32;
        for (size_t i = 0; i < d; i += tile)
            for (size_t j = 0; j < d; j += tile)
                for (size_t row = i; row < std::min(i+tile, d); ++row)
                    for (size_t col = j; col < std::min(j+tile, d); ++col)
                        gram_columns[col*d+row] = gram.data()[row*d+col];
    }
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
#endif
    for (size_t model = 0; model < models; ++model) {
        std::vector<double> gradient(score.data()+model*d, score.data()+(model+1)*d);
        std::vector<double> gradient_update(independent ? 0 : d);
        std::vector<double> t(q), means(components*q), delta(q), old(q), logweight(components), probability(components);
        double* beta = weights.data()+model*d;
        for (int sweep = 0; sweep < (independent ? 1 : sweeps); ++sweep) {
            double energy = 0;
            for (size_t j = 0; j < b; ++j) {
                const size_t start = j*q;
                for (int r = 0; r < q; ++r) old[r] = beta[start+r];
                for (int r = 0; r < q; ++r) {
                    t[r] = gradient[start+r];
                    for (int s = 0; s < q; ++s)
                        t[r] += gram.data()[(start+r)*d+start+s]*old[s];
                }
                for (size_t component = 0; component < components; ++component) {
                    const double* covariance = posterior_covariance.data()+
                        (model*b*components+j*components+component)*q*q;
                    logweight[component] = log_normalizer.data()[model*components*b+j*components+component];
                    for (int r = 0; r < q; ++r) {
                        double mu = 0;
                        for (int s = 0; s < q; ++s) mu += covariance[r*q+s]*t[s];
                        means[component*q+r] = mu;
                        logweight[component] += .5*t[r]*mu;
                    }
                }
                const double maximum = *std::max_element(logweight.begin(), logweight.end());
                double denominator = 0.;
                for (size_t component = 0; component < components; ++component) {
                    probability[component] = std::exp(logweight[component]-maximum);
                    denominator += probability[component];
                }
                const double logz = maximum+std::log(denominator);
                double site = -logz, change = 0;
                for (int r = 0; r < q; ++r) {
                    beta[start+r] = 0.;
                    for (size_t component = 0; component < components; ++component)
                        beta[start+r] += probability[component]*means[component*q+r]/denominator;
                    delta[r] = beta[start+r]-old[r];
                    site += t[r]*beta[start+r];
                }
                for (int r = 0; r < q; ++r) for (int s = 0; s < q; ++s) {
                    const double g = gram.data()[(start+r)*d+start+s];
                    site -= .5*beta[start+r]*g*beta[start+s];
                    change += delta[r]*g*delta[s];
                }
                penalty.data()[model*b+j] = site;
                energy += std::max(0.0, change);
                invalid |= !std::isfinite(site) || !std::isfinite(change);
                if (!independent) {
                    std::fill(gradient_update.begin(), gradient_update.end(), 0.);
                    // Preserve the r summation order and the single final
                    // subtraction for every row; vectorize across rows only.
                    for (int r = 0; r < q; ++r) {
                        const double* column = gram_columns.data()+(start+r)*d;
                        const double change_r = delta[r];
                        for (size_t row = 0; row < d; ++row)
                            gradient_update[row] += column[row]*change_r;
                    }
                    for (size_t row = 0; row < d; ++row)
                        gradient[row] -= gradient_update[row];
                }
            }
            metrics.data()[model*2] = energy;
            metrics.data()[model*2+1] = sweep+1;
            if (energy <= tolerance*tolerance) break;
        }
    }
    if (invalid) throw std::runtime_error("Nonfinite mixture coordinate update");
}
