// A residual has one native owner throughout block updates and reconstruction.
// Python receives explicit snapshots for checkpointing and reporting only.
static void prediction_interaction_design(nb_mat2f_ro<double> genotype,
    nb_mat2f_ro<double> phi, nb_mat2f_rw<double> output, int threads) {
    const size_t n = genotype.shape(0), b = genotype.shape(1), q = phi.shape(1);
    if (!n || !b || !q || phi.shape(0) != n || output.shape(0) != n ||
        output.shape(1) != checked_mul(b,q,"mixture design") || threads < 1)
        throw std::runtime_error("Mixture design dimensions/threads disagree");
    prediction_no_alias(genotype.data(),genotype.size(),output.data(),output.size());
    prediction_no_alias(phi.data(),phi.size(),output.data(),output.size());
    validate_configured_openmp_placement_for_entry();
    nb::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (size_t j=0;j<b*q;++j) {
        const double* g=genotype.data()+(j/q)*n;
        const double* p=phi.data()+(j%q)*n;
        double* out=output.data()+j*n;
        for (size_t i=0;i<n;++i) out[i]=g[i]*p[i];
    }
}

class PredictionMixtureResidual {
public:
    PredictionMixtureResidual(nb_vec1_ro<double> yw, nb_mat2f_ro<double> basis,
                              int models, int threads, bool deferred_projection=false)
        : n_(prediction_dim(yw.size())), k_(models), rank_(static_cast<int>(basis.shape(1))),
          threads_(threads), deferred_(deferred_projection) {
        if (k_<1 || threads_<1 || basis.shape(0)!=static_cast<size_t>(n_) || rank_>=n_)
            throw std::runtime_error("Invalid native mixture residual dimensions");
        validate_protected_gemm_threads(threads_);
        const size_t size=checked_mul(static_cast<size_t>(n_),k_,"mixture residual");
        nb::gil_scoped_release release;
        basis_.assign(basis.data(),basis.data()+basis.size());
        yp_.assign(yw.data(),yw.data()+yw.size());
        residual_.resize(size); raw_.resize(size); projected_.resize(size);
        small_.resize(checked_mul(static_cast<size_t>(rank_),k_,"mixture projection"));
        if (rank_) {
            dgemm_tn_tiled(rank_,1,n_,basis_.data(),n_,yp_.data(),n_,small_.data(),rank_,threads_);
            dgemm_nn_tiled(n_,1,rank_,basis_.data(),n_,small_.data(),rank_,projected_.data(),n_,threads_);
            for (int i=0;i<n_;++i) yp_[i]-=projected_[i];
        }
        for (int model=0;model<k_;++model)
            std::copy(yp_.begin(),yp_.end(),residual_.begin()+static_cast<size_t>(model)*n_);
        if (deferred_ && rank_) {
            correction_.assign(small_.size(),0.);
            residual_projection_.resize(small_.size());
            basis_error_.resize(checked_mul(static_cast<size_t>(rank_),rank_,"mixture basis Gram"));
            dgemm_tn_tiled(rank_,rank_,n_,basis_.data(),n_,basis_.data(),n_,basis_error_.data(),rank_,threads_);
            for (int j=0;j<rank_;++j) basis_error_[j+static_cast<size_t>(rank_)*j]-=1.;
            refresh_projection();
        }
    }

    void score(nb_mat2f_ro<double> w, nb_mat2f_ro<double> projection,
               nb_mat2f_rw<double> output) {
        const int d=design(w,projection);
        if (output.shape(0)!=static_cast<size_t>(d) || output.shape(1)!=static_cast<size_t>(k_))
            throw std::runtime_error("Mixture score shape mismatch");
        prediction_no_alias(w.data(),w.size(),output.data(),output.size());
        prediction_no_alias(projection.data(),projection.size(),output.data(),output.size());
        auto guard=lock(); nb::gil_scoped_release release;
        dgemm_tn_tiled(d,k_,n_,w.data(),n_,residual_.data(),n_,output.data(),d,threads_);
        if (rank_) {
            if (deferred_) {
                // r = residual_ + B correction_; residual_projection_ = B' r.
                for (size_t i=0;i<small_.size();++i)
                    small_[i]=correction_[i]-residual_projection_[i];
                dgemm_tn_tiled(d,k_,rank_,projection.data(),rank_,small_.data(),rank_,output.data(),d,threads_,1.,1.);
            } else {
                dgemm_tn_tiled(rank_,k_,n_,basis_.data(),n_,residual_.data(),n_,small_.data(),rank_,threads_);
                dgemm_tn_tiled(d,k_,rank_,projection.data(),rank_,small_.data(),rank_,output.data(),d,threads_,-1.,1.);
            }
        }
    }

    void update(nb_mat2f_ro<double> w, nb_mat2f_ro<double> delta,
                nb_mat2f_ro<double> projection) {
        const int d=design(w,projection); weights(delta,d);
        auto guard=lock(); nb::gil_scoped_release release;
        dgemm_nn_tiled(n_,k_,d,w.data(),n_,delta.data(),d,residual_.data(),n_,threads_,-1.,1.);
        if (rank_) {
            dgemm_nn_tiled(rank_,k_,d,projection.data(),rank_,delta.data(),d,small_.data(),rank_,threads_);
            if (deferred_) {
                for (size_t i=0;i<small_.size();++i) correction_[i]+=small_[i];
                // B'(r-PW delta) = B'r + (B'B-I)(B'W delta).
                // Retain B'B-I instead of assuming numerically perfect orthogonality.
                dgemm_nn_tiled(rank_,k_,rank_,basis_error_.data(),rank_,small_.data(),rank_,
                    residual_projection_.data(),rank_,threads_,1.,1.);
                dirty_=true;
            } else {
                dgemm_nn_tiled(n_,k_,rank_,basis_.data(),n_,small_.data(),rank_,residual_.data(),n_,threads_,1.,1.);
            }
        }
    }

    void begin_reconstruction() {
        auto guard=lock(); nb::gil_scoped_release release;
        std::fill(raw_.begin(),raw_.end(),0.); reconstructing_=true;
    }
    void add_prediction(nb_mat2f_ro<double> w, nb_mat2f_ro<double> beta) {
        if (w.shape(0)!=static_cast<size_t>(n_)) throw std::runtime_error("Mixture design row mismatch");
        const int d=prediction_dim(w.shape(1)); weights(beta,d);
        auto guard=lock();
        if (!reconstructing_) throw std::runtime_error("Mixture reconstruction was not started");
        nb::gil_scoped_release release;
        dgemm_nn_tiled(n_,k_,d,w.data(),n_,beta.data(),d,raw_.data(),n_,threads_,1.,1.);
    }
    double finish_reconstruction() {
        auto guard=lock();
        if (!reconstructing_) throw std::runtime_error("Mixture reconstruction was not started");
        nb::gil_scoped_release release;
        synchronize_unlocked();
        if (rank_) {
            dgemm_tn_tiled(rank_,k_,n_,basis_.data(),n_,raw_.data(),n_,small_.data(),rank_,threads_);
            dgemm_nn_tiled(n_,k_,rank_,basis_.data(),n_,small_.data(),rank_,projected_.data(),n_,threads_);
        }
        long double drift=0.;
        for (size_t i=0;i<residual_.size();++i) {
            const double value=yp_[i%n_]-raw_[i]+(rank_ ? projected_[i] : 0.);
            if (!std::isfinite(value) || !std::isfinite(residual_[i]))
                throw std::runtime_error("Nonfinite native mixture residual");
            const long double error=static_cast<long double>(value)-residual_[i];
            drift+=error*error; residual_[i]=value;
        }
        reconstructing_=false;
        if (deferred_ && rank_) refresh_projection();
        return std::sqrt(static_cast<double>(drift));
    }
    void copy_residual(nb_mat2f_rw<double> out) {
        if (out.shape(0)!=static_cast<size_t>(n_) || out.shape(1)!=static_cast<size_t>(k_))
            throw std::runtime_error("Mixture snapshot shape mismatch");
        auto guard=lock(); nb::gil_scoped_release release;
        std::copy(residual_.begin(),residual_.end(),out.data());
        if (deferred_ && rank_ && dirty_)
            dgemm_nn_tiled(n_,k_,rank_,basis_.data(),n_,correction_.data(),rank_,out.data(),n_,threads_,1.,1.);
    }
    // Canonicalize at sweep/checkpoint boundaries. Resuming then reproduces the
    // same arithmetic state, without serializing an implementation-specific gauge.
    void synchronize() {
        auto guard=lock(); nb::gil_scoped_release release;
        synchronize_unlocked();
    }
    void copy_prediction(nb_mat2f_rw<double> out) { snapshot(raw_,out); }
    void copy_phenotype(nb_vec1_rw<double> out) {
        if (out.size()!=yp_.size()) throw std::runtime_error("Mixture phenotype shape mismatch");
        auto guard=lock(); nb::gil_scoped_release release;
        std::copy(yp_.begin(),yp_.end(),out.data());
    }
    void restore(nb_mat2f_ro<double> value) {
        if (value.shape(0)!=static_cast<size_t>(n_) || value.shape(1)!=static_cast<size_t>(k_))
            throw std::runtime_error("Mixture restored residual shape mismatch");
        auto guard=lock(); nb::gil_scoped_release release;
        for (size_t i=0;i<value.size();++i) {
            if (!std::isfinite(value.data()[i])) throw std::runtime_error("Nonfinite restored residual");
        }
        std::copy(value.data(),value.data()+value.size(),residual_.begin());
        if (deferred_ && rank_) {
            std::fill(correction_.begin(),correction_.end(),0.);
            dirty_=false;
            refresh_projection();
        }
    }
private:
    int n_, k_, rank_, threads_;
    bool reconstructing_=false, deferred_=false, dirty_=false;
    std::mutex mutex_;
    std::vector<double> basis_,yp_,residual_,raw_,projected_,small_;
    std::vector<double> correction_,residual_projection_,basis_error_;
    void refresh_projection() {
        dgemm_tn_tiled(rank_,k_,n_,basis_.data(),n_,residual_.data(),n_,
            residual_projection_.data(),rank_,threads_);
    }
    void synchronize_unlocked() {
        if (!(deferred_ && rank_ && dirty_)) return;
        dgemm_nn_tiled(n_,k_,rank_,basis_.data(),n_,correction_.data(),rank_,residual_.data(),n_,threads_,1.,1.);
        std::fill(correction_.begin(),correction_.end(),0.);
        dirty_=false;
        refresh_projection();
    }
    std::unique_lock<std::mutex> lock() {
        std::unique_lock<std::mutex> guard(mutex_,std::try_to_lock);
        if (!guard.owns_lock()) throw std::runtime_error("Native mixture residual already in use");
        validate_configured_openmp_placement_for_entry();
        return guard;
    }
    int design(nb_mat2f_ro<double> w, nb_mat2f_ro<double> projection) {
        const int d=prediction_dim(w.shape(1));
        if (w.shape(0)!=static_cast<size_t>(n_) || projection.shape(0)!=static_cast<size_t>(rank_) ||
            projection.shape(1)!=static_cast<size_t>(d)) throw std::runtime_error("Mixture design/projection shape mismatch");
        return d;
    }
    void weights(nb_mat2f_ro<double> value,int d) {
        if (value.shape(0)!=static_cast<size_t>(d) || value.shape(1)!=static_cast<size_t>(k_))
            throw std::runtime_error("Mixture weights shape mismatch");
    }
    void snapshot(const std::vector<double>& source,nb_mat2f_rw<double> out) {
        if (out.shape(0)!=static_cast<size_t>(n_) || out.shape(1)!=static_cast<size_t>(k_))
            throw std::runtime_error("Mixture snapshot shape mismatch");
        auto guard=lock(); nb::gil_scoped_release release;
        std::copy(source.begin(),source.end(),out.data());
    }
};
