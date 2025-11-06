#pragma once

extern "C" {
#ifdef GWLDCORE_USE_MKL_CBLAS
  // When MKL is the chosen vendor, prefer MKL's CBLAS header
  #include <mkl_cblas.h>
#else
  // Otherwise, use the generic CBLAS header (OpenBLAS / Netlib)
  #include <cblas.h>
#endif
}