cmake_minimum_required(VERSION 3.21)
project(gwldcore LANGUAGES CXX)

# ---- pybind11 / Python ----
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =====================================================================================
# BLAS vendor selection (overridable with -DBLA_VENDOR=...)
#   macOS              -> Generic (Accelerate)
#   MKL available      -> Intel10_64lp
#   otherwise (Linux)  -> OpenBLAS (explicit, prefer conda)
# =====================================================================================
if(NOT DEFINED BLA_VENDOR)
  if(APPLE)
    set(BLA_VENDOR Generic CACHE STRING "" FORCE)  # Accelerate
  else()
    if(DEFINED ENV{MKLROOT} OR EXISTS "$ENV{CONDA_PREFIX}/lib/libmkl_rt.so")
      set(BLA_VENDOR Intel10_64lp CACHE STRING "" FORCE)
    else()
      set(BLA_VENDOR OpenBLAS CACHE STRING "" FORCE)
    endif()
  endif()
endif()
message(STATUS "Using BLA_VENDOR=${BLA_VENDOR}")

# First try CMake's FindBLAS, honoring BLA_VENDOR
set(BLA_STATIC OFF CACHE BOOL "" FORCE)
find_package(BLAS)

# If we're insisting on OpenBLAS and FindBLAS failed, wire it manually (prefer conda)
if(NOT BLAS_FOUND AND BLA_VENDOR STREQUAL "OpenBLAS")
  message(STATUS "FindBLAS didn't locate OpenBLAS; trying manual OpenBLAS wiring")
  # Library
  find_library(OPENBLAS_LIB
    NAMES openblas openblas64_ libopenblas
    HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib" "/usr/lib/x86_64-linux-gnu")
  # Header
  find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS "$ENV{CONDA_PREFIX}/include" "/usr/local/include" "/usr/include")

  if(OPENBLAS_LIB)
    add_library(BLAS::BLAS UNKNOWN IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_LOCATION "${OPENBLAS_LIB}")
    set(BLAS_FOUND TRUE)
    set(BLAS_LIBRARIES "${OPENBLAS_LIB}")
    if(CBLAS_INCLUDE_DIR)
      set(OPENBLAS_INCLUDE_DIR "${CBLAS_INCLUDE_DIR}")
    endif()
    message(STATUS "Manually wired OpenBLAS: ${OPENBLAS_LIB}")
  else()
    message(FATAL_ERROR
      "OpenBLAS not found. Please install it in your env (e.g., `conda install openblas`).")
  endif()
endif()

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "BLAS not found (BLA_VENDOR=${BLA_VENDOR}). "
                      "Try setting -DBLA_VENDOR=OpenBLAS or installing MKL/OpenBLAS.")
endif()

# ---- Extension target ----
pybind11_add_module(gwldcore src/gwldcore.cpp)
target_compile_features(gwldcore PUBLIC cxx_std_17)
target_compile_options(gwldcore PRIVATE -O3 -march=native)

# ---- Tell your shim which header to use + link BLAS ----
if(APPLE)
  # Accelerate (vecLib)
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
  if(TARGET BLAS::BLAS)
    target_link_libraries(gwldcore PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(gwldcore PRIVATE "-framework Accelerate")
  endif()

elseif(BLA_VENDOR MATCHES "Intel")
  # MKL
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_MKL_CBLAS)
  if(DEFINED ENV{MKLROOT})
    target_include_directories(gwldcore PRIVATE "$ENV{MKLROOT}/include")
  endif()
  target_link_libraries(gwldcore PRIVATE BLAS::BLAS)

else()
  # OpenBLAS
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_GENERIC_CBLAS)
  if(DEFINED OPENBLAS_INCLUDE_DIR)
    target_include_directories(gwldcore PRIVATE "${OPENBLAS_INCLUDE_DIR}")
  else()
    # Help CBLAS header discovery (Conda/system)
    find_path(CBLAS_INCLUDE_DIR cblas.h
      HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
    if(CBLAS_INCLUDE_DIR)
      target_include_directories(gwldcore PRIVATE "${CBLAS_INCLUDE_DIR}")
    endif()
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
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .)
