cmake_minimum_required(VERSION 3.21)

project(gwldcore LANGUAGES CXX)

find_package(Git QUIET)
set(GWLDCORE_SOURCE_COMMIT "auto" CACHE STRING
    "Exact 40-hex source commit embedded in native build provenance")
set(GWLDCORE_SOURCE_TREE_SHA256 "auto" CACHE STRING
    "Exact SHA-256 of the source snapshot used for this native build")
if (GWLDCORE_SOURCE_COMMIT STREQUAL "auto")
  set(GWLDCORE_SOURCE_COMMIT "unknown")
endif()
if (GWLDCORE_SOURCE_TREE_SHA256 STREQUAL "auto")
  set(GWLDCORE_SOURCE_TREE_SHA256 "unknown")
endif()
if (Git_FOUND AND GWLDCORE_SOURCE_COMMIT STREQUAL "unknown")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE GWLDCORE_GIT_RESULT
    OUTPUT_VARIABLE GWLDCORE_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  string(LENGTH "${GWLDCORE_GIT_COMMIT}" GWLDCORE_GIT_COMMIT_LENGTH)
  if (GWLDCORE_GIT_RESULT EQUAL 0 AND GWLDCORE_GIT_COMMIT_LENGTH EQUAL 40 AND
      GWLDCORE_GIT_COMMIT MATCHES "^[0-9a-fA-F]+$")
    string(TOLOWER "${GWLDCORE_GIT_COMMIT}" GWLDCORE_SOURCE_COMMIT)
  endif()
endif()
if (Git_FOUND AND GWLDCORE_SOURCE_TREE_SHA256 STREQUAL "unknown")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE GWLDCORE_GIT_STATUS_RESULT
    OUTPUT_VARIABLE GWLDCORE_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if (GWLDCORE_GIT_STATUS_RESULT EQUAL 0 AND GWLDCORE_GIT_STATUS STREQUAL "")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse "HEAD^{tree}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE GWLDCORE_GIT_TREE_RESULT
      OUTPUT_VARIABLE GWLDCORE_GIT_TREE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    string(LENGTH "${GWLDCORE_GIT_TREE}" GWLDCORE_GIT_TREE_LENGTH)
    if (GWLDCORE_GIT_TREE_RESULT EQUAL 0 AND GWLDCORE_GIT_TREE_LENGTH EQUAL 40 AND
        GWLDCORE_GIT_TREE MATCHES "^[0-9a-fA-F]+$")
      string(TOLOWER "${GWLDCORE_GIT_TREE}" GWLDCORE_GIT_TREE)
      string(SHA256 GWLDCORE_SOURCE_TREE_SHA256 "git-tree:${GWLDCORE_GIT_TREE}")
    endif()
  endif()
endif()

if (NOT SKBUILD)
  message(WARNING
    "This CMake file is intended to be driven by scikit-build-core.\n"
    "For local development, prefer:\n"
    "  pip install nanobind scikit-build-core[pyproject]\n"
    "  pip install --no-build-isolation -ve .")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
if (NOT nanobind_ROOT)
  message(FATAL_ERROR "Failed to locate nanobind via 'python -m nanobind --cmake_dir'.")
endif()
find_package(nanobind CONFIG REQUIRED)

# =====================================================================================
# BLAS vendor selection (overridable with -DBLA_VENDOR=...)
#   macOS              -> Generic (Accelerate)
#   MKL available      -> Intel10_64lp
#   otherwise (Linux)  -> OpenBLAS (explicit, prefer conda)
# =====================================================================================
if (NOT DEFINED BLA_VENDOR)
  if (APPLE)
    set(BLA_VENDOR Generic CACHE STRING "" FORCE)  # Accelerate
  else()
    if (DEFINED ENV{MKLROOT} OR EXISTS "$ENV{CONDA_PREFIX}/lib/libmkl_rt.so")
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
if (NOT BLAS_FOUND AND BLA_VENDOR STREQUAL "OpenBLAS")
  message(STATUS "FindBLAS did not locate OpenBLAS; trying manual wiring")
  find_library(OPENBLAS_LIB
    NAMES openblas openblas64_ libopenblas
    HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib" "/usr/lib/x86_64-linux-gnu")
  find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS "$ENV{CONDA_PREFIX}/include" "/usr/local/include" "/usr/include")
  if (OPENBLAS_LIB)
    add_library(BLAS::BLAS UNKNOWN IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_LOCATION "${OPENBLAS_LIB}")
    set(BLAS_FOUND TRUE)
    set(BLAS_LIBRARIES "${OPENBLAS_LIB}")
    if (CBLAS_INCLUDE_DIR)
      set(OPENBLAS_INCLUDE_DIR "${CBLAS_INCLUDE_DIR}")
    endif()
    message(STATUS "Manually wired OpenBLAS: ${OPENBLAS_LIB}")
  else()
    message(FATAL_ERROR "OpenBLAS not found. Please install it (e.g. 'conda install openblas').")
  endif()
endif()

if (NOT BLAS_FOUND)
  message(FATAL_ERROR
    "BLAS not found (BLA_VENDOR=${BLA_VENDOR}). Try setting -DBLA_VENDOR=OpenBLAS or installing MKL/OpenBLAS.")
endif()

# =====================================================================================
# Optional OpenMP
# =====================================================================================
option(GWLDCORE_USE_OPENMP "Enable OpenMP if available" ON)
if (GWLDCORE_USE_OPENMP)
  find_package(OpenMP)
endif()

# =====================================================================================
# CPU tuning options
# =====================================================================================
option(GWLDCORE_ENABLE_NATIVE_OPT "Enable -march=native / architecture-specific local tuning" ON)

set(SUMMIT_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/summit")
set(SUMMIT_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/native")
set(SUMMIT_NATIVE_COMMON_DIR "${SUMMIT_NATIVE_DIR}/common")

# =====================================================================================
# Shared genotype / BED decoding core
# =====================================================================================
add_library(gwldcore_core STATIC
  ${SUMMIT_NATIVE_COMMON_DIR}/genotype.cpp
)
set_target_properties(gwldcore_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gwldcore_core PUBLIC ${SUMMIT_NATIVE_COMMON_DIR})
target_compile_features(gwldcore_core PUBLIC cxx_std_17)
if (MSVC)
  target_compile_options(gwldcore_core PRIVATE /O2)
else()
  target_compile_options(gwldcore_core PRIVATE -O3)
endif()
if (OpenMP_CXX_FOUND)
  target_link_libraries(gwldcore_core PUBLIC OpenMP::OpenMP_CXX)
endif()

function(gwldcore_configure_module target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_17)
  target_include_directories(${target_name} PRIVATE ${SUMMIT_NATIVE_COMMON_DIR})
  target_link_libraries(${target_name} PRIVATE gwldcore_core)

  if (MSVC)
    target_compile_options(${target_name} PRIVATE /O2)
  else()
    target_compile_options(${target_name} PRIVATE -O3)
    if (GWLDCORE_ENABLE_NATIVE_OPT)
      target_compile_options(${target_name} PRIVATE -march=native)
    endif()
  endif()

  if (APPLE)
    target_compile_definitions(${target_name} PRIVATE GWLDCORE_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    if (TARGET BLAS::BLAS)
      target_link_libraries(${target_name} PRIVATE BLAS::BLAS)
    else()
      target_link_libraries(${target_name} PRIVATE "-framework Accelerate")
    endif()
  elseif (BLA_VENDOR MATCHES "Intel")
    target_compile_definitions(${target_name} PRIVATE GWLDCORE_USE_MKL_CBLAS)
    if (DEFINED ENV{MKLROOT})
      target_include_directories(${target_name} PRIVATE "$ENV{MKLROOT}/include")
    endif()
    target_link_libraries(${target_name} PRIVATE BLAS::BLAS)
  else()
    target_compile_definitions(${target_name} PRIVATE GWLDCORE_USE_GENERIC_CBLAS)
    if (BLA_VENDOR STREQUAL "OpenBLAS")
      target_compile_definitions(${target_name} PRIVATE GWLDCORE_USE_OPENBLAS)
    endif()
    if (DEFINED OPENBLAS_INCLUDE_DIR)
      target_include_directories(${target_name} PRIVATE "${OPENBLAS_INCLUDE_DIR}")
    else()
      find_path(CBLAS_INCLUDE_DIR cblas.h
        HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
      if (CBLAS_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE "${CBLAS_INCLUDE_DIR}")
      endif()
    endif()
    target_link_libraries(${target_name} PRIVATE BLAS::BLAS)
  endif()

  if (OpenMP_CXX_FOUND)
    target_compile_definitions(${target_name} PRIVATE GWLDCORE_USE_OPENMP)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if (DEFINED ENV{CONDA_PREFIX} AND UNIX)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "$ORIGIN;$ENV{CONDA_PREFIX}/lib"
      INSTALL_RPATH "$ORIGIN;$ENV{CONDA_PREFIX}/lib")
  endif()
endfunction()

# =====================================================================================
# Python extension modules (nanobind)
# =====================================================================================
# Use NOMINSIZE because these modules execute substantial compute-heavy code directly
# and should be optimized for throughput rather than binary size.
nanobind_add_module(gwldcore NOMINSIZE ${SUMMIT_NATIVE_DIR}/gwldcore.cpp)
gwldcore_configure_module(gwldcore)

nanobind_add_module(winldcore NOMINSIZE ${SUMMIT_NATIVE_DIR}/winldcore.cpp)
gwldcore_configure_module(winldcore)

set(GWLDCORE_MODULE_TARGETS gwldcore winldcore)

# The bounded direct GxE context deliberately owns Linux file descriptors and
# an mmap.  Do not make supported Apple builds compile a source file whose
# decoder cannot be correct there; Python keeps the portable NumPy fallback.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  nanobind_add_module(gxeldcore NOMINSIZE ${SUMMIT_NATIVE_DIR}/gxeldcore.cpp)
  gwldcore_configure_module(gxeldcore)
  # Wide threaded GEMMs have produced rare, vendor-independent corruptions on
  # production NUMA hosts. Keep the lightweight checksum/repair path enabled
  # for the direct GxE backend regardless of the selected BLAS implementation.
  target_compile_definitions(gxeldcore PRIVATE GWLDCORE_GEMM_INTEGRITY)
  target_compile_definitions(gxeldcore PRIVATE
    GWLDCORE_SOURCE_COMMIT="${GWLDCORE_SOURCE_COMMIT}"
    GWLDCORE_SOURCE_TREE_SHA256="${GWLDCORE_SOURCE_TREE_SHA256}"
    GWLDCORE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    GWLDCORE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    GWLDCORE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    GWLDCORE_BLAS_VENDOR="${BLA_VENDOR}"
    GWLDCORE_NATIVE_OPT=$<BOOL:${GWLDCORE_ENABLE_NATIVE_OPT}>
    GWLDCORE_OPENMP_ENABLED=$<BOOL:${OpenMP_CXX_FOUND}>)
  list(APPEND GWLDCORE_MODULE_TARGETS gxeldcore)
endif()

install(TARGETS ${GWLDCORE_MODULE_TARGETS}
  LIBRARY DESTINATION summit
  RUNTIME DESTINATION summit
  ARCHIVE DESTINATION summit)

# Install the Python package. The source-tree compatibility launcher
# src/summit.py must not be installed because it shadows the summit package.
install(DIRECTORY "${SUMMIT_PYTHON_PACKAGE_DIR}"
  DESTINATION .
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "tmp" EXCLUDE)
