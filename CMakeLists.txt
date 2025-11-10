cmake_minimum_required(VERSION 3.21)
project(gwldcore LANGUAGES CXX)

# ---- pybind11 / Python ----
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- BLAS selection heuristics (overridable) ----
# User can force a specific vendor with: -DBLA_VENDOR=...
# Common values: Intel10_64lp, OpenBLAS, Generic (on mac this resolves to vecLib/Accelerate)
if(NOT DEFINED BLA_VENDOR)
  if(APPLE)
    set(BLA_VENDOR Generic CACHE STRING "" FORCE)  # system BLAS -> Accelerate
  else()
    if(DEFINED ENV{MKLROOT})
      set(BLA_VENDOR Intel10_64lp CACHE STRING "" FORCE)
    elseif(EXISTS "$ENV{CONDA_PREFIX}/lib/libopenblas.so" OR
           EXISTS "/usr/lib/x86_64-linux-gnu/libopenblas.so")
      set(BLA_VENDOR OpenBLAS CACHE STRING "" FORCE)
    else()
      set(BLA_VENDOR Generic CACHE STRING "" FORCE)
    endif()
  endif()
endif()
message(STATUS "Using BLA_VENDOR=${BLA_VENDOR}")

find_package(BLAS REQUIRED)  # honors BLA_VENDOR

# ---- Extension target ----
pybind11_add_module(gwldcore src/gwldcore.cpp)
target_compile_features(gwldcore PUBLIC cxx_std_17)

# Some compilers like having this; harmless elsewhere
target_compile_options(gwldcore PRIVATE -O3 -march=native)

# ---- Tell your shim which header to use ----
if(APPLE)
  # We’re using Accelerate; prefer new LAPACK interface to avoid deprecations
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_ACCELERATE ACCELERATE_NEW_LAPACK)

  # Link either the imported BLAS target or the framework directly as fallback
  if(TARGET BLAS::BLAS)
    target_link_libraries(gwldcore PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(gwldcore PRIVATE "-framework Accelerate")
  endif()

elseif(BLA_VENDOR MATCHES "Intel")
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_MKL_CBLAS)
  if(DEFINED ENV{MKLROOT})
    target_include_directories(gwldcore PRIVATE "$ENV{MKLROOT}/include")
  endif()
  target_link_libraries(gwldcore PRIVATE BLAS::BLAS)

else()
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_GENERIC_CBLAS)
  # Try to help CBLAS header discovery when using Conda OpenBLAS
  find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
  if(CBLAS_INCLUDE_DIR)
    target_include_directories(gwldcore PRIVATE "${CBLAS_INCLUDE_DIR}")
  endif()
  target_link_libraries(gwldcore PRIVATE BLAS::BLAS)
endif()

# ---- Optional OpenMP (mac needs Homebrew libomp) ----
option(GWLDCORE_USE_OPENMP "Enable OpenMP if available" ON)
if(GWLDCORE_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_OPENMP)
    target_link_libraries(gwldcore PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# ---- Install into wheel root ----
install(TARGETS gwldcore
  LIBRARY DESTINATION .
  RUNTIME  DESTINATION .
  ARCHIVE DESTINATION .)
