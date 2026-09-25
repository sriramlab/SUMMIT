// Lossless packing of already-decoded hard calls, independent of file format.
// BED reading remains in the shared descriptor reader. Cache codes are 0/1/2/3,
// where 3 means missing; these are deliberately not the BED bit convention.
static void prediction_cache_no_alias(const void* a, size_t na, const void* b, size_t nb) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    if (x < y+nb && y < x+na)
        throw std::runtime_error("Hard-call cache buffers must not alias");
}

static void prediction_pack_calls(nb_mat2f_ro<int8_t> raw, nb_mat2f_rw<uint8_t> packed, int threads) {
    const size_t n=raw.shape(0), b=raw.shape(1), stride=(n+3)/4;
    if (!n || !b || packed.shape(0)!=stride || packed.shape(1)!=b || threads<1)
        throw std::runtime_error("Invalid packed hard-call cache dimensions");
    prediction_cache_no_alias(raw.data(),raw.size(),packed.data(),packed.size());
    validate_configured_openmp_placement_for_entry();
    nb::gil_scoped_release release;
    int invalid=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
#endif
    for (size_t j=0;j<b;++j) {
        for (size_t i=0;i<stride;++i) {
            uint8_t byte=0;
            for (size_t r=0;r<4;++r) {
                const int v=4*i+r<n ? raw.data()[j*n+4*i+r] : -127;
                invalid |= v!=-127 && (v<0 || v>2);
                const unsigned code=v==-127 ? 3u : static_cast<unsigned>(v)&3u;
                byte |= static_cast<uint8_t>(code<<(2*r));
            }
            packed.data()[j*stride+i]=byte;
        }
    }
    if (invalid) throw std::runtime_error("Invalid hard call in packed cache");
}

static void prediction_unpack_calls(nb_mat2f_ro<uint8_t> packed, nb_mat2f_rw<int8_t> raw, int threads) {
    const size_t n=raw.shape(0), b=raw.shape(1), stride=(n+3)/4;
    if (!n || !b || packed.shape(0)!=stride || packed.shape(1)!=b || threads<1)
        throw std::runtime_error("Invalid packed hard-call cache dimensions");
    prediction_cache_no_alias(raw.data(),raw.size(),packed.data(),packed.size());
    validate_configured_openmp_placement_for_entry();
    nb::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (size_t j=0;j<b;++j) {
        for (size_t i=0;i<n;++i) {
            const unsigned code=(packed.data()[j*stride+i/4]>>(2*(i%4)))&3u;
            raw.data()[j*n+i]=code==3 ? -127 : static_cast<int8_t>(code);
        }
    }
}
