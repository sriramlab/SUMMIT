// Bounded pair-feature preparation for the shared PCGC covariance reducer.
// Matrix reductions use the existing protected NN/TN entry points.

void pcgc_require_disjoint(const void* output, size_t output_bytes,
                          const void* input, size_t input_bytes) {
    const auto a=reinterpret_cast<uintptr_t>(output), b=reinterpret_cast<uintptr_t>(input);
    if ((a<=b && b-a<output_bytes) || (b<a && a-b<input_bytes))
        throw std::invalid_argument("PCGC output must not overlap its inputs");
}

bool pcgc_accumulate_relatedness(
    nb_mat2c_ro<double> genotype, nb_mat2f_ro<double> annotations,
    nb_mat2c_ro<int64_t> partners, nb_mat2c_rw<double> target, int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const size_t n=genotype.shape(0), b=genotype.shape(1), l=partners.shape(1), k=annotations.shape(1);
    const size_t pairs=checked_mul(n,l,"PCGC sampled genotype pairs");
    if (n<2 || !b || !l || !k || annotations.shape(0)!=b || partners.shape(0)!=n ||
        target.shape(0)!=pairs || target.shape(1)!=k || pairs>static_cast<size_t>(INT64_MAX))
        throw std::invalid_argument("PCGC sampled relatedness axes disagree");
    const size_t target_bytes=checked_mul(checked_mul(pairs,k,"PCGC relatedness size"),sizeof(double),"PCGC relatedness bytes");
    pcgc_require_disjoint(target.data(),target_bytes,genotype.data(),
        checked_mul(checked_mul(n,b,"PCGC genotype size"),sizeof(double),"PCGC genotype bytes"));
    pcgc_require_disjoint(target.data(),target_bytes,annotations.data(),
        checked_mul(checked_mul(b,k,"PCGC annotation size"),sizeof(double),"PCGC annotation bytes"));
    pcgc_require_disjoint(target.data(),target_bytes,partners.data(),
        checked_mul(pairs,sizeof(int64_t),"PCGC partner bytes"));
    std::vector<std::vector<std::pair<size_t,double>>> columns(k);
    bool partition=true;
    for (size_t variant=0;variant<b;++variant) {
        size_t count=0;
        for (size_t annotation=0;annotation<k;++annotation) {
            const double value=annotations.data()[annotation*b+variant];
            if (!std::isfinite(value) || value<0.)
                throw std::invalid_argument("PCGC annotation weights must be finite and nonnegative");
            if (value!=0.) { columns[annotation].emplace_back(variant,value); ++count; }
        }
        partition &= count<=1;
    }
    int invalid=0;
    {
        nb::gil_scoped_release release;
        // Validate before updating the caller-owned accumulator.
#pragma omp parallel for num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t row=0;row<static_cast<int64_t>(n);++row) {
            for (size_t variant=0;variant<b;++variant)
                invalid |= !std::isfinite(genotype.data()[static_cast<size_t>(row)*b+variant]);
            for (size_t draw=0;draw<l;++draw) {
                const size_t pair=static_cast<size_t>(row)*l+draw;
                const int64_t other=partners.data()[pair];
                invalid |= other<0 || static_cast<uint64_t>(other)>=n || other==row;
                for (size_t annotation=0;annotation<k;++annotation)
                    invalid |= !std::isfinite(target.data()[pair*k+annotation]);
            }
        }
        if (!invalid) {
#pragma omp parallel num_threads(requested_threads) reduction(|:invalid)
            {
                std::vector<double> products(b);
#pragma omp for schedule(static)
                for (int64_t pair=0;pair<static_cast<int64_t>(pairs);++pair) {
                    const double* left=genotype.data()+(static_cast<size_t>(pair)/l)*b;
                    const double* right=genotype.data()+static_cast<size_t>(partners.data()[pair])*b;
#pragma omp simd reduction(|:invalid)
                    for (size_t variant=0;variant<b;++variant) {
                        products[variant]=left[variant]*right[variant];
                        invalid |= !std::isfinite(products[variant]);
                    }
                    for (size_t annotation=0;annotation<k;++annotation) {
                        double sum=0.;
                        const auto& active=columns[annotation];
#pragma omp simd reduction(+:sum)
                        for (size_t j=0;j<active.size();++j) sum+=products[active[j].first]*active[j].second;
                        const size_t offset=static_cast<size_t>(pair)*k+annotation;
                        target.data()[offset]+=sum;
                        invalid |= !std::isfinite(target.data()[offset]);
                    }
                }
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC sampled relatedness contains invalid partners or non-finite products");
    return partition;
}

nb::tuple pcgc_pair_panels(
    nb_mat2c_ro<double> relatedness,
    nb_mat2c_ro<int64_t> partners,
    nb_mat2c_ro<double> features,
    nb_vec1_ro<double> response,
    nb_vec1_ro<double> probabilities,
    nb_vec1_ro<double> leverage,
    nb_mat2c_ro<int64_t> context_pairs,
    nb_mat2c_ro<int64_t> square_pairs,
    int64_t first, int64_t last, bool context_only,
    uint64_t memory_bytes, int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const size_t n = features.shape(0), q = features.shape(1);
    const size_t l = partners.shape(1), k = relatedness.shape(1);
    const size_t p = context_pairs.shape(0), s = square_pairs.shape(0);
    const size_t c = checked_mul(k, p, "PCGC pair components");
    if (n < 8 || !q || l < 2 || !k || !p || !s ||
        partners.shape(0) != n || response.shape(0) != n || leverage.shape(0) != n ||
        (probabilities.shape(0) != 0 && probabilities.shape(0) != n) ||
        relatedness.shape(0) != checked_mul(n, l, "PCGC pair count") ||
        context_pairs.shape(1) != 2 || square_pairs.shape(1) != 2 ||
        first < 0 || last <= first || static_cast<uint64_t>(last) > n) {
        throw std::invalid_argument("PCGC pair panel axes or row range disagree");
    }
    for (size_t j = 0; j < p*2; ++j) {
        if (context_pairs.data()[j] < 0 || static_cast<uint64_t>(context_pairs.data()[j]) >= q)
            throw std::invalid_argument("PCGC context pair is out of range");
    }
    for (size_t j = 0; j < s*2; ++j) {
        if (square_pairs.data()[j] < 0 ||
            static_cast<uint64_t>(square_pairs.data()[j]) >= (context_only ? p : c))
            throw std::invalid_argument("PCGC square pair is out of range");
    }
    const size_t people = static_cast<size_t>(last-first);
    const size_t rows = checked_mul(people, l, "PCGC pair panel rows");
    // Three square panels, two score panels, kernels, context products, weights,
    // and one row-mean panel. All dimensions are admitted before allocation.
    const size_t columns = checked_add(checked_mul(3, s, "PCGC square panels"),
        checked_add(checked_mul(3, c, "PCGC score panels"),
                    checked_add(p, 2, "PCGC row weights"), "PCGC panels"), "PCGC panels");
    const size_t elements = checked_add(checked_mul(rows, columns, "PCGC panels"),
                                        checked_mul(people, s, "PCGC row means"), "PCGC workspace");
    if (checked_mul(elements, sizeof(double), "PCGC workspace bytes") > memory_bytes)
        throw std::runtime_error("PCGC pair panels exceed the admitted workspace");
    double *score, *weighted_score, *square, *weighted_square, *noise_square, *mean;
    auto score_out = make_owned_numpy_mat2f<double>(rows,c,&score);
    auto weighted_score_out = make_owned_numpy_mat2f<double>(rows,c,&weighted_score);
    auto square_out = make_owned_numpy_mat2f<double>(rows,s,&square);
    auto weighted_square_out = make_owned_numpy_mat2f<double>(rows,s,&weighted_square);
    auto noise_square_out = make_owned_numpy_mat2f<double>(rows,s,&noise_square);
    auto mean_out = make_owned_numpy_mat2f<double>(people,s,&mean);
    std::vector<double> kernels(checked_mul(rows,c,"PCGC kernels"));
    std::vector<double> contexts(checked_mul(rows,p,"PCGC context products"));
    std::vector<double> inverse(rows), ypair(rows);
    int invalid = 0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t local = 0; local < static_cast<int64_t>(rows); ++local) {
            const size_t i = static_cast<size_t>(first)+static_cast<size_t>(local)/l;
            const size_t draw = static_cast<size_t>(local)%l;
            const int64_t other = partners.data()[i*l+draw];
            if (other < 0 || static_cast<uint64_t>(other) >= n || static_cast<size_t>(other) == i) {
                invalid |= 1;
                continue;
            }
            double weight = static_cast<double>(n-1);
            if (probabilities.shape(0)) {
                const double own = probabilities.data()[i], partner = probabilities.data()[other];
                if (!(own > 0. && own < 1. && partner > 0. && partner < 1.)) invalid |= 1;
                weight = (1.-own)/partner;
            }
            inverse[local] = weight;
            ypair[local] = response.data()[i]*response.data()[other];
            invalid |= !std::isfinite(weight) || !std::isfinite(ypair[local]) || !std::isfinite(leverage.data()[i]);
            for (size_t pair = 0; pair < p; ++pair) {
                const size_t u = static_cast<size_t>(context_pairs.data()[2*pair]);
                const size_t v = static_cast<size_t>(context_pairs.data()[2*pair+1]);
                double value = features.data()[i*q+u]*features.data()[static_cast<size_t>(other)*q+v];
                if (u != v) value += features.data()[i*q+v]*features.data()[static_cast<size_t>(other)*q+u];
                contexts[pair*rows+local] = value;
                for (size_t annotation = 0; annotation < k; ++annotation) {
                    const size_t column = annotation*p+pair;
                    const double kernel = value*relatedness.data()[(i*l+draw)*k+annotation];
                    kernels[column*rows+local] = kernel;
                    score[column*rows+local] = kernel*ypair[local];
                    weighted_score[column*rows+local] = score[column*rows+local]*weight;
                    invalid |= !std::isfinite(kernel) || !std::isfinite(weighted_score[column*rows+local]);
                }
            }
        }
        if (!invalid) {
            const double* source = context_only ? contexts.data() : kernels.data();
#pragma omp parallel for num_threads(requested_threads) schedule(static) reduction(|:invalid)
            for (int64_t column = 0; column < static_cast<int64_t>(s); ++column) {
                const size_t u = static_cast<size_t>(square_pairs.data()[2*column]);
                const size_t v = static_cast<size_t>(square_pairs.data()[2*column+1]);
                for (size_t person = 0; person < people; ++person) {
                    double total = 0.;
                    for (size_t draw = 0; draw < l; ++draw) {
                        const size_t row = person*l+draw, offset = static_cast<size_t>(column)*rows+row;
                        const double value = source[u*rows+row]*source[v*rows+row];
                        square[offset] = value;
                        weighted_square[offset] = value*inverse[row];
                        noise_square[offset] = weighted_square[offset]*inverse[row]*leverage.data()[static_cast<size_t>(first)+person];
                        total += weighted_square[offset];
                        invalid |= !std::isfinite(value) || !std::isfinite(weighted_square[offset]) || !std::isfinite(noise_square[offset]);
                    }
                    mean[static_cast<size_t>(column)*people+person] = total/static_cast<double>(l);
                    invalid |= !std::isfinite(mean[static_cast<size_t>(column)*people+person]);
                }
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC pair panels contain invalid partners or non-finite products");
    return nb::make_tuple(score_out,weighted_score_out,square_out,weighted_square_out,noise_square_out,mean_out);
}

// Products of r_a h_p factor into separate annotation and person monomials.
// The returned panels still go through the shared protected matrix products.
nb::tuple pcgc_factored_pair_panel(
    nb_mat2c_ro<double> relatedness, nb_mat2c_ro<int64_t> partners,
    nb_mat2c_ro<double> features, nb_vec1_ro<double> response,
    nb_vec1_ro<double> probabilities, nb_vec1_ro<double> leverage,
    nb_mat2c_ro<int64_t> annotation_powers, nb_mat2c_ro<int64_t> context_powers,
    int64_t first, int64_t last, bool noise, uint64_t memory_bytes, int threads
) {
    validate_protected_gemm_threads(threads);
    const size_t n=features.shape(0), q=features.shape(1), k=relatedness.shape(1), l=partners.shape(1);
    const size_t degree=annotation_powers.shape(1), a=annotation_powers.shape(0), f=context_powers.shape(0);
    if (n<8 || !q || !k || l<2 || !a || !f || degree<2 || degree>4 || (noise && degree!=4) ||
        context_powers.shape(1)!=degree || partners.shape(0)!=n || response.shape(0)!=n || leverage.shape(0)!=n ||
        relatedness.shape(0)!=checked_mul(n,l,"PCGC pair count") ||
        (probabilities.shape(0)!=0 && probabilities.shape(0)!=n) ||
        first<0 || last<=first || static_cast<uint64_t>(last)>n)
        throw std::invalid_argument("PCGC factored panel axes disagree");
    for (size_t j=0;j<checked_mul(a,degree,"PCGC annotation powers");++j)
        if (annotation_powers.data()[j]<0 || static_cast<uint64_t>(annotation_powers.data()[j])>=k)
            throw std::invalid_argument("PCGC annotation power is out of range");
    for (size_t j=0;j<checked_mul(f,degree,"PCGC context powers");++j)
        if (context_powers.data()[j]<0 || static_cast<uint64_t>(context_powers.data()[j])>=q)
            throw std::invalid_argument("PCGC context power is out of range");
    const size_t rows=checked_mul(static_cast<size_t>(last-first),l,"PCGC panel rows");
    const size_t ff=checked_mul(f,checked_add(f,1,"PCGC context powers"),"PCGC context products")/2;
    const size_t columns=checked_mul(ff,noise?2:1,"PCGC noise panels");
    const size_t workspace=checked_mul(rows,checked_add(checked_add(a,columns,"PCGC panels"),
        checked_add(checked_mul(2,f,"PCGC monomials"),3,"PCGC weights"),"PCGC workspace"),"PCGC panel size");
    const size_t workspace_bytes=checked_add(checked_mul(workspace,sizeof(double),"PCGC panel bytes"),
        checked_mul(ff,sizeof(std::pair<size_t,size_t>),"PCGC context index bytes"),"PCGC factored workspace bytes");
    if (rows>static_cast<size_t>(INT64_MAX) || workspace_bytes>memory_bytes)
        throw std::runtime_error("PCGC factored panels exceed the admitted workspace");
    double *left,*right;
    auto left_out=make_owned_numpy_mat2f<double>(rows,a,&left);
    auto right_out=make_owned_numpy_mat2f<double>(rows,columns,&right);
    std::vector<double> own(checked_mul(rows,f,"PCGC own monomials")),other(own.size()),weight(rows),noise_weight(rows);
    std::vector<int64_t> neighbor(rows);
    std::vector<std::pair<size_t,size_t>> context_products;
    context_products.reserve(ff);
    for (size_t u=0;u<f;++u) for (size_t v=u;v<f;++v) context_products.emplace_back(u,v);
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
        for (int64_t row=0;row<static_cast<int64_t>(rows);++row) {
            const size_t i=static_cast<size_t>(first)+static_cast<size_t>(row)/l;
            const int64_t j=partners.data()[static_cast<size_t>(first)*l+static_cast<size_t>(row)];
            neighbor[row]=j;
            if (j<0 || static_cast<uint64_t>(j)>=n || static_cast<size_t>(j)==i) { invalid|=1; continue; }
            double w=static_cast<double>(n-1);
            if (probabilities.shape(0)) {
                const double pi=probabilities.data()[i], pj=probabilities.data()[j];
                if (!(pi>0. && pi<1. && pj>0. && pj<1.)) invalid|=1;
                w=(1.-pi)/pj;
            }
            const double y=response.data()[i]*response.data()[j];
            weight[row]=w/static_cast<double>(l)*(degree==2 ? y*y : degree==3 ? y : 1.);
            noise_weight[row]=noise ? w/static_cast<double>(l)*w*leverage.data()[i] : 0.;
            invalid |= !std::isfinite(weight[row]) || !std::isfinite(noise_weight[row]);
        }
        if (!invalid) {
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
            for (int64_t col=0;col<static_cast<int64_t>(a);++col) {
                for (size_t row=0;row<rows;++row) {
                    const double* r=relatedness.data()+(static_cast<size_t>(first)*l+row)*k;
                    double value=1.;
                    for (size_t d=0;d<degree;++d) value*=r[annotation_powers.data()[static_cast<size_t>(col)*degree+d]];
                    left[static_cast<size_t>(col)*rows+row]=value;
                    invalid |= !std::isfinite(value);
                }
            }
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
            for (int64_t col=0;col<static_cast<int64_t>(f);++col) {
                for (size_t row=0;row<rows;++row) {
                    const size_t i=static_cast<size_t>(first)+row/l, j=static_cast<size_t>(neighbor[row]);
                    double x=1.,y=1.;
                    for (size_t d=0;d<degree;++d) {
                        const size_t index=static_cast<size_t>(context_powers.data()[static_cast<size_t>(col)*degree+d]);
                        x*=features.data()[i*q+index]; y*=features.data()[j*q+index];
                    }
                    own[static_cast<size_t>(col)*rows+row]=x; other[static_cast<size_t>(col)*rows+row]=y;
                    invalid |= !std::isfinite(x) || !std::isfinite(y);
                }
            }
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
            for (int64_t col=0;col<static_cast<int64_t>(columns);++col) {
                const size_t index=static_cast<size_t>(col)%ff;
                const size_t u=context_products[index].first,v=context_products[index].second;
                const double* w=static_cast<size_t>(col)<ff ? weight.data() : noise_weight.data();
                for (size_t row=0;row<rows;++row) {
                    double value=own[u*rows+row]*other[v*rows+row];
                    if (u!=v) value+=own[v*rows+row]*other[u*rows+row];
                    value*=w[row];
                    right[static_cast<size_t>(col)*rows+row]=value;
                    invalid |= !std::isfinite(value);
                }
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC factored panels contain invalid partners or non-finite products");
    return nb::make_tuple(left_out,right_out);
}

nb_numpy_mat2f<double> pcgc_pair_means(
    nb_mat2c_ro<double> relatedness, nb_mat2c_ro<int64_t> partners, nb_mat2c_ro<double> features,
    nb_vec1_ro<double> probabilities, nb_mat2c_ro<int64_t> context_pairs, nb_mat2c_ro<int64_t> square_pairs,
    int64_t first, int64_t last, uint64_t memory_bytes, int threads
) {
    validate_protected_gemm_threads(threads);
    const size_t n=features.shape(0),q=features.shape(1),k=relatedness.shape(1),l=partners.shape(1);
    const size_t p=context_pairs.shape(0),s=square_pairs.shape(0),c=checked_mul(k,p,"PCGC components");
    if (n<8 || !q || !k || l<2 || !p || !s || partners.shape(0)!=n ||
        relatedness.shape(0)!=checked_mul(n,l,"PCGC pair count") || context_pairs.shape(1)!=2 || square_pairs.shape(1)!=2 ||
        (probabilities.shape(0)!=0 && probabilities.shape(0)!=n) || first<0 || last<=first || static_cast<uint64_t>(last)>n)
        throw std::invalid_argument("PCGC pair mean axes disagree");
    for (size_t j=0;j<checked_mul(p,2,"PCGC context indices");++j)
        if (context_pairs.data()[j]<0 || static_cast<uint64_t>(context_pairs.data()[j])>=q)
            throw std::invalid_argument("PCGC context pair is out of range");
    for (size_t j=0;j<checked_mul(s,2,"PCGC square indices");++j)
        if (square_pairs.data()[j]<0 || static_cast<uint64_t>(square_pairs.data()[j])>=c)
            throw std::invalid_argument("PCGC square pair is out of range");
    const size_t people=static_cast<size_t>(last-first),rows=checked_mul(people,l,"PCGC panel rows");
    const size_t workspace=checked_add(checked_mul(rows,checked_add(c,1,"PCGC mean columns"),"PCGC mean panels"),
        checked_mul(people,s,"PCGC means"),"PCGC mean workspace");
    if (rows>static_cast<size_t>(INT64_MAX) || checked_mul(workspace,sizeof(double),"PCGC mean bytes")>memory_bytes)
        throw std::runtime_error("PCGC pair means exceed the admitted workspace");
    double* mean;
    auto result=make_owned_numpy_mat2f<double>(people,s,&mean);
    std::vector<double> kernels(checked_mul(rows,c,"PCGC kernels")),weight(rows);
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
        for (int64_t row=0;row<static_cast<int64_t>(rows);++row) {
            const size_t i=static_cast<size_t>(first)+static_cast<size_t>(row)/l;
            const int64_t j=partners.data()[static_cast<size_t>(first)*l+static_cast<size_t>(row)];
            if (j<0 || static_cast<uint64_t>(j)>=n || static_cast<size_t>(j)==i) { invalid|=1; continue; }
            double w=static_cast<double>(n-1);
            if (probabilities.shape(0)) {
                const double pi=probabilities.data()[i],pj=probabilities.data()[j];
                if (!(pi>0. && pi<1. && pj>0. && pj<1.)) invalid|=1;
                w=(1.-pi)/pj;
            }
            weight[row]=w;
            invalid |= !std::isfinite(w);
        }
        if (!invalid) {
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
            for (int64_t column=0;column<static_cast<int64_t>(c);++column) {
                const size_t a=static_cast<size_t>(column)/p,pair=static_cast<size_t>(column)%p;
                const size_t u=static_cast<size_t>(context_pairs.data()[2*pair]),v=static_cast<size_t>(context_pairs.data()[2*pair+1]);
                for (size_t row=0;row<rows;++row) {
                    const size_t offset=static_cast<size_t>(first)*l+row,i=static_cast<size_t>(first)+row/l,j=static_cast<size_t>(partners.data()[offset]);
                    double h=features.data()[i*q+u]*features.data()[j*q+v];
                    if (u!=v) h+=features.data()[i*q+v]*features.data()[j*q+u];
                    kernels[static_cast<size_t>(column)*rows+row]=relatedness.data()[offset*k+a]*h;
                    invalid |= !std::isfinite(kernels[static_cast<size_t>(column)*rows+row]);
                }
            }
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
            for (int64_t column=0;column<static_cast<int64_t>(s);++column) {
                const size_t u=static_cast<size_t>(square_pairs.data()[2*column]),v=static_cast<size_t>(square_pairs.data()[2*column+1]);
                for (size_t person=0;person<people;++person) {
                    double sum=0.;
                    for (size_t draw=0;draw<l;++draw) {
                        const size_t row=person*l+draw;
                        sum+=kernels[u*rows+row]*kernels[v*rows+row]*weight[row];
                    }
                    mean[static_cast<size_t>(column)*people+person]=sum/static_cast<double>(l);
                    invalid |= !std::isfinite(sum);
                }
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC pair means contain invalid partners or non-finite products");
    return result;
}

void pcgc_center_strata(nb_mat2f_rw<double> values, nb_vec1_ro<uint8_t> cases, int requested_threads) {
    validate_protected_gemm_threads(requested_threads);
    const size_t n = values.shape(0), columns = values.shape(1);
    if (cases.shape(0) != n) throw std::invalid_argument("PCGC stratum axes disagree");
    size_t counts[2] = {0,0};
    for (size_t i=0;i<n;++i) {
        if (cases.data()[i] > 1) throw std::invalid_argument("PCGC strata must be zero or one");
        ++counts[cases.data()[i]];
    }
    if (counts[0]<4 || counts[1]<4) throw std::invalid_argument("PCGC sampling needs at least four cases and four controls");
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t column=0;column<static_cast<int64_t>(columns);++column) {
            double sums[2]={0.,0.};
            double* data=values.data()+static_cast<size_t>(column)*n;
            for (size_t i=0;i<n;++i) { sums[cases.data()[i]]+=data[i]; invalid |= !std::isfinite(data[i]); }
            sums[0]/=static_cast<double>(counts[0]); sums[1]/=static_cast<double>(counts[1]);
            for (size_t i=0;i<n;++i) { data[i]-=sums[cases.data()[i]]; invalid |= !std::isfinite(data[i]); }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC stratum centering produced non-finite values");
}

nb_numpy_mat2f<double> pcgc_column_products(
    nb_mat2c_ro<double> values, nb_mat2c_ro<int64_t> pairs, int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const size_t n=values.shape(0), c=values.shape(1), s=pairs.shape(0);
    if (!n || !c || !s || pairs.shape(1)!=2)
        throw std::invalid_argument("PCGC product columns have invalid axes");
    for (size_t j=0;j<2*s;++j)
        if (pairs.data()[j]<0 || static_cast<uint64_t>(pairs.data()[j])>=c)
            throw std::invalid_argument("PCGC product column is out of range");
    checked_mul(checked_mul(n,s,"PCGC product columns"),sizeof(double),"PCGC product bytes");
    double* output;
    auto result=make_owned_numpy_mat2f<double>(n,s,&output);
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t column=0;column<static_cast<int64_t>(s);++column) {
            const size_t a=static_cast<size_t>(pairs.data()[2*column]), b=static_cast<size_t>(pairs.data()[2*column+1]);
            for (size_t row=0;row<n;++row) {
                const double value=values.data()[row*c+a]*values.data()[row*c+b];
                output[static_cast<size_t>(column)*n+row]=value;
                invalid |= !std::isfinite(value);
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC column products are non-finite");
    return result;
}

size_t pcgc_architecture_dimension(nb_mat2c_ro<double> omega, int q, int probes) {
    if (q<=0 || probes<=0 || !omega.shape(0) || omega.shape(1)!=checked_mul(q,q,"PCGC contexts"))
        throw std::invalid_argument("PCGC architecture covariance axes disagree");
    return checked_mul(checked_mul(omega.shape(0),static_cast<size_t>(q),"PCGC architecture axes"),
                       static_cast<size_t>(probes),"PCGC architecture probes");
}

nb_numpy_mat2f<double> pcgc_architecture_features(
    nb_mat2c_ro<double> omega, nb_mat2c_ro<double> values,
    int q, int probes, bool transpose, int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const size_t d=pcgc_architecture_dimension(omega,q,probes), t=values.shape(0);
    const size_t size=checked_mul(d,d,"PCGC architecture square");
    if (!t || values.shape(1)!=size)
        throw std::invalid_argument("PCGC architecture sketch axes disagree");
    checked_mul(checked_mul(size,t,"PCGC transformed sketch"),sizeof(double),"PCGC transformed bytes");
    double* output;
    auto result=make_owned_numpy_mat2f<double>(size,t,&output);
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for collapse(2) num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t target=0;target<static_cast<int64_t>(t);++target) {
            for (int64_t row=0;row<static_cast<int64_t>(d);++row) {
                const size_t annotation=static_cast<size_t>(row)/(q*static_cast<size_t>(probes));
                const size_t context=(static_cast<size_t>(row)/probes)%q, probe=static_cast<size_t>(row)%probes;
                for (size_t column=0;column<d;++column) {
                    double value=0.;
                    for (int r=0;r<q;++r) {
                        const size_t other=(annotation*q+static_cast<size_t>(r))*probes+probe;
                        value+=omega.data()[(annotation*q+context)*q+r]*values.data()[static_cast<size_t>(target)*size+other*d+column];
                    }
                    const size_t index=transpose ? column*d+static_cast<size_t>(row) : static_cast<size_t>(row)*d+column;
                    output[static_cast<size_t>(target)*size+index]=value;
                    invalid |= !std::isfinite(value);
                }
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC transformed architecture sketch is non-finite");
    return result;
}

nb_numpy_mat2f<double> pcgc_architecture_polynomials(
    nb_mat2c_ro<double> omega, nb_mat2c_ro<double> directions,
    nb_mat2c_ro<double> left, nb_mat2c_ro<double> right,
    int q, int probes, int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const size_t d=pcgc_architecture_dimension(omega,q,probes), t=left.shape(0);
    const size_t size=checked_mul(d,d,"PCGC architecture square");
    const size_t parameters=checked_mul(omega.shape(0),omega.shape(1),"PCGC architecture coefficients");
    if (!t || left.shape(1)!=size || right.shape(0)!=t || right.shape(1)!=size ||
        directions.shape(0)!=t || directions.shape(1)!=parameters)
        throw std::invalid_argument("PCGC architecture polynomial axes disagree");
    std::vector<double> partial(checked_mul(checked_mul(t,d,"PCGC trace rows"),3,"PCGC trace coefficients"));
    double* output;
    auto result=make_owned_numpy_mat2f<double>(t,3,&output);
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for collapse(2) num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t target=0;target<static_cast<int64_t>(t);++target) {
            for (int64_t row=0;row<static_cast<int64_t>(d);++row) {
                const size_t a=static_cast<size_t>(row)/(q*static_cast<size_t>(probes));
                const size_t u=(static_cast<size_t>(row)/probes)%q, bi=static_cast<size_t>(row)%probes;
                double sums[3]={0.,0.,0.};
                for (size_t column=0;column<d;++column) {
                    const size_t b=column/(q*static_cast<size_t>(probes)), v=(column/probes)%q, bj=column%probes;
                    double l0=0.,ld=0.,r0=0.,rd=0.;
                    for (int s=0;s<q;++s) {
                        const size_t li=(a*q+static_cast<size_t>(s))*probes+bi;
                        const size_t ri=(b*q+static_cast<size_t>(s))*probes+bj;
                        const double lv=left.data()[static_cast<size_t>(target)*size+li*d+column];
                        const double rv=right.data()[static_cast<size_t>(target)*size+ri*d+static_cast<size_t>(row)];
                        l0+=omega.data()[(a*q+u)*q+s]*lv;
                        ld+=directions.data()[static_cast<size_t>(target)*parameters+(a*q+u)*q+s]*lv;
                        r0+=omega.data()[(b*q+v)*q+s]*rv;
                        rd+=directions.data()[static_cast<size_t>(target)*parameters+(b*q+v)*q+s]*rv;
                    }
                    sums[0]+=l0*r0; sums[1]+=ld*r0+l0*rd; sums[2]+=ld*rd;
                }
                for (size_t j=0;j<3;++j) {
                    partial[(static_cast<size_t>(target)*d+static_cast<size_t>(row))*3+j]=sums[j];
                    invalid |= !std::isfinite(sums[j]);
                }
            }
        }
        for (size_t target=0;target<t;++target) {
            for (size_t j=0;j<3;++j) {
                long double sum=0.;
                for (size_t row=0;row<d;++row) sum+=partial[(target*d+row)*3+j];
                output[j*t+target]=static_cast<double>(2.L*sum/(static_cast<long double>(probes)*probes));
                invalid |= !std::isfinite(output[j*t+target]);
            }
        }
    }
    if (invalid) throw std::invalid_argument("PCGC architecture variance polynomial is non-finite");
    return result;
}

nb_numpy_mat2f<double> pcgc_sampling_direction_terms(
    nb_vec1_ro<double> theta, nb_mat2c_ro<double> rows,
    nb_mat2c_ro<double> linear, nb_mat2c_ro<double> quadratic, int requested_threads
) {
    validate_protected_gemm_threads(requested_threads);
    const size_t c=theta.shape(0), j=rows.shape(1), jj=checked_mul(j,j,"PCGC covariance equations");
    if (!c || !j || rows.shape(0)!=c || linear.shape(0)!=c || linear.shape(1)!=jj ||
        quadratic.shape(0)!=checked_mul(c,c,"PCGC covariance parameters") || quadratic.shape(1)!=jj)
        throw std::invalid_argument("PCGC covariance direction axes disagree");
    double* output;
    auto result=make_owned_numpy_mat2f<double>(c,2,&output);
    int invalid=0;
    {
        nb::gil_scoped_release release;
#pragma omp parallel for num_threads(requested_threads) schedule(static) reduction(|:invalid)
        for (int64_t target=0;target<static_cast<int64_t>(c);++target) {
            double first=0.,second=0.;
            for (size_t i=0;i<j;++i) {
                const double ri=rows.data()[static_cast<size_t>(target)*j+i];
                if (ri==0.) continue;
                for (size_t h=0;h<j;++h) {
                    const double weight=ri*rows.data()[static_cast<size_t>(target)*j+h];
                    if (weight==0.) continue;
                    const size_t offset=i*j+h;
                    double derivative=linear.data()[static_cast<size_t>(target)*jj+offset];
                    for (size_t parameter=0;parameter<c;++parameter) {
                        derivative+=theta.data()[parameter]*(quadratic.data()[(static_cast<size_t>(target)*c+parameter)*jj+offset]+
                            quadratic.data()[(parameter*c+static_cast<size_t>(target))*jj+offset]);
                    }
                    first+=weight*derivative;
                    second+=weight*quadratic.data()[(static_cast<size_t>(target)*c+static_cast<size_t>(target))*jj+offset];
                }
            }
            output[target]=first; output[c+target]=second;
            invalid |= !std::isfinite(first) || !std::isfinite(second);
        }
    }
    if (invalid) throw std::invalid_argument("PCGC covariance direction polynomial is non-finite");
    return result;
}

void bind_pcgc_moments(nb::module_& module) {
    module.def("pcgc_accumulate_relatedness", &pcgc_accumulate_relatedness,
        nb::arg("genotype"),nb::arg("annotations"),nb::arg("partners"),nb::arg("target"),nb::arg("threads"));
    module.def("pcgc_pair_panels", &pcgc_pair_panels,
        nb::arg("relatedness"),nb::arg("partners"),nb::arg("features"),nb::arg("response"),
        nb::arg("probabilities"),nb::arg("leverage"),nb::arg("context_pairs"),nb::arg("square_pairs"),
        nb::arg("first"),nb::arg("last"),nb::arg("context_only"),nb::arg("memory_bytes"),nb::arg("threads"));
    module.def("pcgc_factored_pair_panel", &pcgc_factored_pair_panel,
        nb::arg("relatedness"),nb::arg("partners"),nb::arg("features"),nb::arg("response"),
        nb::arg("probabilities"),nb::arg("leverage"),nb::arg("annotation_powers"),nb::arg("context_powers"),
        nb::arg("first"),nb::arg("last"),nb::arg("noise"),nb::arg("memory_bytes"),nb::arg("threads"));
    module.def("pcgc_pair_means", &pcgc_pair_means,
        nb::arg("relatedness"),nb::arg("partners"),nb::arg("features"),nb::arg("probabilities"),
        nb::arg("context_pairs"),nb::arg("square_pairs"),nb::arg("first"),nb::arg("last"),nb::arg("memory_bytes"),nb::arg("threads"));
    module.def("pcgc_center_strata", &pcgc_center_strata,nb::arg("values"),nb::arg("cases"),nb::arg("threads"));
    module.def("pcgc_column_products", &pcgc_column_products,nb::arg("values"),nb::arg("pairs"),nb::arg("threads"));
    module.def("pcgc_architecture_features", &pcgc_architecture_features,
        nb::arg("omega"),nb::arg("values"),nb::arg("contexts"),nb::arg("probes"),nb::arg("transpose"),nb::arg("threads"));
    module.def("pcgc_architecture_polynomials", &pcgc_architecture_polynomials,
        nb::arg("omega"),nb::arg("directions"),nb::arg("left"),nb::arg("right"),
        nb::arg("contexts"),nb::arg("probes"),nb::arg("threads"));
    module.def("pcgc_sampling_direction_terms", &pcgc_sampling_direction_terms,
        nb::arg("theta"),nb::arg("rows"),nb::arg("linear"),nb::arg("quadratic"),nb::arg("threads"));
}
