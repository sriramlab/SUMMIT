// src/blas_compat.hpp
#pragma once

extern "C" {
#if defined(GWLDCORE_USE_MKL_CBLAS)
// ---- Intel MKL (Linux/Intel) ----
  #include <mkl_cblas.h>

#elif defined(__APPLE__) && defined(GWLDCORE_USE_ACCELERATE)
// ---- Apple Accelerate (macOS) ----
// CMake should define ACCELERATE_NEW_LAPACK to avoid deprecation warnings.
  #include <Accelerate/Accelerate.h>

#elif defined(GWLDCORE_USE_GENERIC_CBLAS)
// ---- Generic CBLAS (OpenBLAS / Netlib) ----
  #include <cblas.h>

#else
// ---- Fallback heuristics ----
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif
} // extern "C"
