#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

inline void nb_check_for_interrupt() {
    nb::gil_scoped_acquire gil;
    if (PyErr_CheckSignals() != 0)
        throw nb::python_error();
}

template <typename T>
using nb_vec1_ro = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using nb_vec1_rw = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using nb_mat2c_ro = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename T>
using nb_mat2c_rw = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename T>
using nb_mat2f_ro = nb::ndarray<const T, nb::ndim<2>, nb::f_contig, nb::device::cpu>;

template <typename T>
using nb_mat2f_rw = nb::ndarray<T, nb::ndim<2>, nb::f_contig, nb::device::cpu>;

using nb_any_array_ro = nb::ndarray<nb::ro, nb::device::cpu>;

template <typename T>
using nb_numpy_vec1 = nb::ndarray<T, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using nb_numpy_mat2c = nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename T>
using nb_numpy_mat2f = nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::f_contig, nb::device::cpu>;

template <typename T>
inline nb_numpy_vec1<T> make_owned_numpy_vec1(std::size_t n, T** out_ptr = nullptr, std::size_t align = 64) {
    void* p = nullptr;
    if (n > 0 && posix_memalign(&p, align, n * sizeof(T)) != 0)
        throw std::bad_alloc();
    if (out_ptr)
        *out_ptr = static_cast<T*>(p);
    nb::capsule owner(p, [](void* q) noexcept { std::free(q); });
    return nb_numpy_vec1<T>(static_cast<T*>(p), { n }, owner);
}

template <typename T>
inline nb_numpy_mat2c<T> make_owned_numpy_mat2c(std::size_t rows,
                                                std::size_t cols,
                                                T** out_ptr = nullptr,
                                                std::size_t align = 64) {
    const std::size_t n = rows * cols;
    void* p = nullptr;
    if (n > 0 && posix_memalign(&p, align, n * sizeof(T)) != 0)
        throw std::bad_alloc();
    if (out_ptr)
        *out_ptr = static_cast<T*>(p);
    nb::capsule owner(p, [](void* q) noexcept { std::free(q); });
    return nb_numpy_mat2c<T>(static_cast<T*>(p), { rows, cols }, owner);
}

template <typename T>
inline nb_numpy_mat2f<T> make_owned_numpy_mat2f(std::size_t rows,
                                                std::size_t cols,
                                                T** out_ptr = nullptr,
                                                std::size_t align = 64) {
    const std::size_t n = rows * cols;
    void* p = nullptr;
    if (n > 0 && posix_memalign(&p, align, n * sizeof(T)) != 0)
        throw std::bad_alloc();
    if (out_ptr)
        *out_ptr = static_cast<T*>(p);
    nb::capsule owner(p, [](void* q) noexcept { std::free(q); });
    return nb_numpy_mat2f<T>(static_cast<T*>(p), { rows, cols }, owner);
}

inline const std::vector<int>& parse_row_sel_nb(nb::object row_sel_obj, int64_t N_total) {
    struct Cache {
        PyObject* key = nullptr;
        int64_t N_total = -1;
        std::vector<int> rows;
    };
    static thread_local Cache C;

    PyObject* k = row_sel_obj.is_none() ? nullptr : row_sel_obj.ptr();
    if (C.key == k && C.N_total == N_total && !C.rows.empty())
        return C.rows;

    C.key = k;
    C.N_total = N_total;
    C.rows.clear();

    if (row_sel_obj.is_none()) {
        C.rows.resize((std::size_t) N_total);
        for (int64_t i = 0; i < N_total; ++i)
            C.rows[(std::size_t) i] = (int) i;
        return C.rows;
    }

    nb_any_array_ro idx = nb::cast<nb_any_array_ro>(row_sel_obj);
    if (idx.ndim() != 1)
        throw std::runtime_error("row_sel must be a 1D CPU array of int32 or int64 indices");

    const std::size_t n = idx.shape(0);
    C.rows.resize(n);

    if (idx.dtype() == nb::dtype<int32_t>()) {
        auto v = idx.view<const int32_t, nb::ndim<1>>();
        for (std::size_t i = 0; i < n; ++i)
            C.rows[i] = (int) v(i);
    } else if (idx.dtype() == nb::dtype<int64_t>()) {
        auto v = idx.view<const int64_t, nb::ndim<1>>();
        for (std::size_t i = 0; i < n; ++i)
            C.rows[i] = (int) v(i);
    } else {
        throw std::runtime_error("row_sel must have dtype int32 or int64");
    }

    return C.rows;
}

