cmake_minimum_required(VERSION 3.21)

# IMPORTANT: do NOT list CUDA here; we enable it conditionally later.
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

set(BLA_STATIC OFF CACHE BOOL "" FORCE)
find_package(BLAS)

# Manual OpenBLAS wiring if FindBLAS failed and vendor is OpenBLAS
if(NOT BLAS_FOUND AND BLA_VENDOR STREQUAL "OpenBLAS")
  message(STATUS "FindBLAS didn't locate OpenBLAS; trying manual wiring")
  find_library(OPENBLAS_LIB
    NAMES openblas openblas64_ libopenblas
    HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib" "/usr/lib/x86_64-linux-gnu")
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
    message(FATAL_ERROR "OpenBLAS not found. Please install it (e.g., `conda install openblas`).")
  endif()
endif()

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "BLAS not found (BLA_VENDOR=${BLA_VENDOR}). "
                      "Try setting -DBLA_VENDOR=OpenBLAS or installing MKL/OpenBLAS.")
endif()

# =====================================================================================
# Optional OpenMP
# =====================================================================================
option(GWLDCORE_USE_OPENMP "Enable OpenMP if available" ON)
if(GWLDCORE_USE_OPENMP)
  find_package(OpenMP)
endif()

# =====================================================================================
# Shared host-only core (genotype reader) – used by both CPU & CUDA modules
# =====================================================================================
add_library(gwldcore_core STATIC
  src/genotype.cpp
)
set_target_properties(gwldcore_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gwldcore_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(gwldcore_core PUBLIC cxx_std_17)
target_compile_options(gwldcore_core PRIVATE -O3)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gwldcore_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# =====================================================================================
# CPU Python module
# =====================================================================================
pybind11_add_module(gwldcore src/gwldcore.cpp)
target_compile_features(gwldcore PUBLIC cxx_std_17)
target_compile_options(gwldcore PRIVATE -O3 -march=native)
target_include_directories(gwldcore PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gwldcore PRIVATE gwldcore_core)

if(APPLE)
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
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
  if(DEFINED OPENBLAS_INCLUDE_DIR)
    target_include_directories(gwldcore PRIVATE "${OPENBLAS_INCLUDE_DIR}")
  else()
    find_path(CBLAS_INCLUDE_DIR cblas.h
      HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
    if(CBLAS_INCLUDE_DIR)
      target_include_directories(gwldcore PRIVATE "${CBLAS_INCLUDE_DIR}")
    endif()
  endif()
  target_link_libraries(gwldcore PRIVATE BLAS::BLAS)
endif()

if(OpenMP_CXX_FOUND)
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_OPENMP)
  target_link_libraries(gwldcore PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS gwldcore
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .)

# ======================================
# CUDA backend (optional)
# ======================================
option(GWLDCORE_BUILD_CUDA "Build CUDA backend for Phase 2" ON)

if(GWLDCORE_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # CUDA::cudart, CUDA::cublas

    # reasonable default archs if user didn't set
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89;90)
    endif()

    pybind11_add_module(gwldcore_cuda src/gwldcore_cuda.cu)
    set_target_properties(gwldcore_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON)

    target_compile_options(gwldcore_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)

    # --- BLAS for host-side projection inside CUDA TU ---
    if(APPLE)
      target_compile_definitions(gwldcore_cuda PRIVATE GWLDCORE_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
      if(TARGET BLAS::BLAS)
        target_link_libraries(gwldcore_cuda PRIVATE BLAS::BLAS)
      else()
        target_link_libraries(gwldcore_cuda PRIVATE "-framework Accelerate")
      endif()
    elseif(BLA_VENDOR MATCHES "Intel")
      target_compile_definitions(gwldcore_cuda PRIVATE GWLDCORE_USE_MKL_CBLAS)
      if(DEFINED ENV{MKLROOT})
        target_include_directories(gwldcore_cuda PRIVATE "$ENV{MKLROOT}/include")
      endif()
      target_link_libraries(gwldcore_cuda PRIVATE BLAS::BLAS)
    else()
      target_compile_definitions(gwldcore_cuda PRIVATE GWLDCORE_USE_GENERIC_CBLAS)
      if(DEFINED OPENBLAS_INCLUDE_DIR)
        target_include_directories(gwldcore_cuda PRIVATE "${OPENBLAS_INCLUDE_DIR}")
      else()
        find_path(CBLAS_INCLUDE_DIR cblas.h
          HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
        if(CBLAS_INCLUDE_DIR)
          target_include_directories(gwldcore_cuda PRIVATE "${CBLAS_INCLUDE_DIR}")
        endif()
      endif()
      target_link_libraries(gwldcore_cuda PRIVATE BLAS::BLAS)
    endif()

    target_link_libraries(gwldcore_cuda PRIVATE
      gwldcore_core
      BLAS::BLAS
      CUDA::cudart
      CUDA::cublas
      pybind11::module
    )

    # keep rpath so conda’s BLAS is found at runtime
    set_target_properties(gwldcore_cuda PROPERTIES
      BUILD_RPATH "$ORIGIN;$ENV{CONDA_PREFIX}/lib"
      INSTALL_RPATH "$ORIGIN;$ENV{CONDA_PREFIX}/lib"
    )

    install(TARGETS gwldcore_cuda
      LIBRARY DESTINATION .
      RUNTIME DESTINATION .
      ARCHIVE DESTINATION .)
  else()
    message(STATUS "CUDA compiler not found; gwldcore_cuda will not be built.")
  endif()
endif()
