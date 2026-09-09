// Prediction has an iterative lifecycle, independent of reference estimation.
// All BED bit decoding and guarded GEMMs remain in the existing common stack.

class PredictionBEDReader {
public:
    PredictionBEDReader(int bed, int bim, int fam, nb::object rows,
                        int threads, uint64_t workspace)
        : reader_(std::make_unique<DirectContext>(bed, bim, fam, std::move(rows),
              1, threads, workspace, threads, DescriptorOnlyDirectContextTag{}, 1)) {}

    void read(nb_vec1_ro<int64_t> variants, nb_mat2f_rw<int8_t> output) {
        auto lock = reader_->acquire_call_lock();
        reader_->ensure_open();
        reader_->check_files_unchanged();
        const size_t b = variants.shape(0);
        if (b == 0 || b > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            output.shape(0) != static_cast<size_t>(reader_->n_) || output.shape(1) != b ||
            checked_mul(output.shape(0), b, "prediction raw block") > reader_->max_workspace_bytes_)
            throw std::runtime_error("Prediction raw block shape/workspace mismatch");
        const auto input_start = reinterpret_cast<uintptr_t>(variants.data());
        const auto output_start = reinterpret_cast<uintptr_t>(output.data());
        if (input_start < output_start + output.size() &&
            output_start < input_start + variants.size()*sizeof(int64_t))
            throw std::runtime_error("Prediction raw output must not alias variant indices");
        for (size_t j = 0; j < b; ++j) {
            if (variants.data()[j] < 0 || variants.data()[j] >= reader_->m_total_ ||
                (j && variants.data()[j] <= variants.data()[j-1]))
                throw std::runtime_error("Prediction variants must be sorted unique and in range");
        }
        {
            nb::gil_scoped_release release;
#if defined(__linux__)
            // Bounded readahead on this same authenticated mapping. Noncontiguous
            // selections do not prefetch unrelated chromosome-sized ranges.
            const size_t begin = 3 + static_cast<size_t>(variants.data()[0]) * reader_->bytes_per_snp_;
            const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
            if (variants.data()[b-1] - variants.data()[0] == static_cast<int64_t>(b-1)) {
                const size_t end = begin + b * reader_->bytes_per_snp_;
                const size_t aligned = (begin / page) * page;
                (void)::madvise(reader_->bed_base_ + aligned, end - aligned, MADV_WILLNEED);
            }
#endif
            read_bed_calls_a1_memory(reader_->bed_base_, reader_->bed_size_,
                reader_->n_total_, reader_->bytes_per_snp_, reader_->rows_,
                variants.data(), static_cast<int>(b), reader_->decode_threads_, output.data());
#if defined(__linux__)
            for (size_t j = 0; j < b;) {
                size_t end = j + 1;
                while (end < b && variants.data()[end] == variants.data()[end-1] + 1) ++end;
                madvise_dontneed_consumed_range(reader_->bed_base_, reader_->bed_size_,
                    3 + static_cast<size_t>(variants.data()[j]) * reader_->bytes_per_snp_,
                    (end-j) * reader_->bytes_per_snp_);
                j = end;
            }
#endif
        }
        reader_->check_files_unchanged();
    }

    void close() { reader_->close(); }
    nb::dict info() const { return reader_->info(); }
private:
    std::unique_ptr<DirectContext> reader_;
};

static int prediction_dim(size_t value) {
    if (!value || value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Prediction kernel dimension exceeds BLAS integer range");
    return static_cast<int>(value);
}

static void prediction_no_alias(const double* input, size_t ni,
                                const double* output, size_t no) {
    const auto a = reinterpret_cast<uintptr_t>(input);
    const auto b = reinterpret_cast<uintptr_t>(output);
    if (a < b + no * sizeof(double) && b < a + ni * sizeof(double))
        throw std::runtime_error("Prediction output must not alias an input");
}

static void prediction_product(nb_mat2f_ro<double> left, nb_mat2f_ro<double> right,
                               nb_mat2f_rw<double> output, bool transpose, int threads) {
    const int m = prediction_dim(left.shape(transpose ? 1 : 0));
    const int k = prediction_dim(left.shape(transpose ? 0 : 1));
    const int n = prediction_dim(right.shape(1));
    if (right.shape(0) != static_cast<size_t>(k) || output.shape(0) != static_cast<size_t>(m) ||
        output.shape(1) != static_cast<size_t>(n))
        throw std::runtime_error("Prediction matrix product dimensions disagree");
    prediction_no_alias(left.data(), left.size(), output.data(), output.size());
    prediction_no_alias(right.data(), right.size(), output.data(), output.size());
    validate_protected_gemm_threads(threads);
    nb::gil_scoped_release release;
    if (transpose) {
        (void)dgemm_tn_partitioned_columns(m, n, k, left.data(), k, right.data(), k,
                                          output.data(), m, threads);
    } else {
        (void)dgemm_nn_partitioned_rows(m, n, k, left.data(), m, right.data(), k,
                                       output.data(), m, threads, 1.0, 0.0);
    }
}

static void prediction_covariance_block(
    nb_mat2f_ro<double> genotype, nb_mat2f_ro<double> packed,
    nb_mat2c_ro<double> priors, nb_mat2f_ro<double> phi,
    nb_mat2f_rw<double> output, double mass, int threads)
{
    const int n = prediction_dim(genotype.shape(0));
    const int b = prediction_dim(genotype.shape(1));
    const int q = prediction_dim(phi.shape(1));
    const int models = prediction_dim(priors.shape(0));
    const int columns = prediction_dim(checked_mul(static_cast<size_t>(q), models, "prediction RHS"));
    if (phi.shape(0) != static_cast<size_t>(n) || packed.shape(0) != static_cast<size_t>(n) ||
        packed.shape(1) != static_cast<size_t>(columns) ||
        priors.shape(1) != checked_mul(static_cast<size_t>(q), q, "prediction covariance") ||
        output.shape(0) != static_cast<size_t>(n) || output.shape(1) != static_cast<size_t>(models) ||
        !(mass > 0.0) || !std::isfinite(mass))
        throw std::runtime_error("Prediction covariance block dimensions/mass disagree");
    for (auto input : {genotype, packed, phi})
        prediction_no_alias(input.data(), input.size(), output.data(), output.size());
    prediction_no_alias(priors.data(), priors.size(), output.data(), output.size());
    validate_protected_gemm_threads(threads);
    const size_t small = checked_mul(static_cast<size_t>(b), columns, "prediction SNP tile");
    const size_t large = checked_mul(static_cast<size_t>(n), columns, "prediction product tile");
    nb::gil_scoped_release release;
    // Both GEMMs use beta=0 and every mixed cell is assigned. Avoid zeroing
    // multi-megabyte scratch that is immediately overwritten (NUMA first touch).
    std::unique_ptr<double[]> transposed(new double[small]);
    std::unique_ptr<double[]> mixed(new double[small]);
    std::unique_ptr<double[]> product(new double[large]);
    (void)dgemm_tn_partitioned_columns(b, columns, n, genotype.data(), n,
        packed.data(), n, transposed.get(), b, threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int model = 0; model < models; ++model) {
        for (int j = 0; j < b; ++j) {
            for (int r = 0; r < q; ++r) {
                double value = 0.0;
                for (int s = 0; s < q; ++s)
                    value += transposed[static_cast<size_t>(model*q+s)*b+j] *
                        priors.data()[static_cast<size_t>(model)*q*q+s*q+r];
                mixed[static_cast<size_t>(model*q+r)*b+j] = value / mass;
            }
        }
    }
    (void)dgemm_nn_partitioned_rows(n, columns, b, genotype.data(), n,
        mixed.get(), b, product.get(), n, threads, 1.0, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int model = 0; model < models; ++model) {
        for (int i = 0; i < n; ++i) {
            double value = 0.0;
            for (int r = 0; r < q; ++r)
                value += phi.data()[static_cast<size_t>(r)*n+i] *
                    product[static_cast<size_t>(model*q+r)*n+i];
            output.data()[static_cast<size_t>(model)*n+i] += value;
        }
    }
}

static void bind_prediction(nb::module_& module) {
    nb::class_<PredictionBEDReader>(module, "PredictionBEDReader")
        .def(nb::init<int, int, int, nb::object, int, uint64_t>(),
             nb::arg("bed_descriptor"), nb::arg("bim_descriptor"), nb::arg("fam_descriptor"),
             nb::arg("rows"), nb::arg("threads"), nb::arg("max_workspace_bytes"))
        .def("read", &PredictionBEDReader::read, nb::arg("variants").noconvert(), nb::arg("out").noconvert())
        .def("close", &PredictionBEDReader::close)
        .def("info", &PredictionBEDReader::info);
    module.def("prediction_product", &prediction_product, nb::arg("left").noconvert(),
        nb::arg("right").noconvert(), nb::arg("out").noconvert(), nb::arg("transpose"), nb::arg("threads"));
    module.def("prediction_covariance_block", &prediction_covariance_block,
        nb::arg("genotype").noconvert(), nb::arg("packed").noconvert(), nb::arg("priors").noconvert(),
        nb::arg("phi").noconvert(), nb::arg("out").noconvert(), nb::arg("mass"), nb::arg("threads"));
}
