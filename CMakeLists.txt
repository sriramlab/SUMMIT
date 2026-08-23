cmake_minimum_required(VERSION 3.21)

project(gwldcore LANGUAGES CXX)

find_package(Git QUIET)
set(GWLDCORE_SOURCE_COMMIT "auto" CACHE STRING
    "Exact 40-hex source commit embedded in native build provenance")
set(GWLDCORE_SOURCE_TREE_SHA256 "auto" CACHE STRING
    "Exact SHA-256 commitment to the tracked source snapshot used for this native build")
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
      COMMAND "${GIT_EXECUTABLE}" ls-files --stage
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE GWLDCORE_GIT_FILES_RESULT
      OUTPUT_VARIABLE GWLDCORE_GIT_FILES
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if (GWLDCORE_GIT_FILES_RESULT EQUAL 0 AND NOT GWLDCORE_GIT_FILES STREQUAL "")
      # Commit directly to every tracked path, mode, byte length, and SHA-256
      # content digest.  This is deliberately not a SHA-256 wrapper around
      # Git's SHA-1 tree identifier.
      set(GWLDCORE_SOURCE_MANIFEST
        "summit-tracked-source-tree-sha256-v1\n")
      string(REPLACE "\n" ";" GWLDCORE_GIT_FILE_LINES
        "${GWLDCORE_GIT_FILES}")
      foreach(GWLDCORE_GIT_FILE_LINE IN LISTS GWLDCORE_GIT_FILE_LINES)
        if (GWLDCORE_GIT_FILE_LINE STREQUAL "")
          continue()
        endif()
        string(REGEX MATCH
          "^([0-9]+) [0-9a-fA-F]+ [0-9]+\t(.+)$"
          GWLDCORE_GIT_FILE_MATCH "${GWLDCORE_GIT_FILE_LINE}")
        if (NOT GWLDCORE_GIT_FILE_MATCH)
          message(FATAL_ERROR
            "Cannot parse tracked source entry: ${GWLDCORE_GIT_FILE_LINE}")
        endif()
        set(GWLDCORE_GIT_FILE_MODE "${CMAKE_MATCH_1}")
        set(GWLDCORE_GIT_FILE_PATH "${CMAKE_MATCH_2}")
        set(GWLDCORE_GIT_FILE_ABSOLUTE
          "${CMAKE_CURRENT_SOURCE_DIR}/${GWLDCORE_GIT_FILE_PATH}")
        if (NOT EXISTS "${GWLDCORE_GIT_FILE_ABSOLUTE}")
          message(FATAL_ERROR
            "Tracked source path is absent: ${GWLDCORE_GIT_FILE_PATH}")
        endif()
        file(SIZE "${GWLDCORE_GIT_FILE_ABSOLUTE}"
          GWLDCORE_GIT_FILE_SIZE)
        file(SHA256 "${GWLDCORE_GIT_FILE_ABSOLUTE}"
          GWLDCORE_GIT_FILE_SHA256)
        string(LENGTH "${GWLDCORE_GIT_FILE_PATH}"
          GWLDCORE_GIT_FILE_PATH_LENGTH)
        string(APPEND GWLDCORE_SOURCE_MANIFEST
          "${GWLDCORE_GIT_FILE_MODE}:${GWLDCORE_GIT_FILE_PATH_LENGTH}:"
          "${GWLDCORE_GIT_FILE_PATH}:${GWLDCORE_GIT_FILE_SIZE}:"
          "${GWLDCORE_GIT_FILE_SHA256}\n")
      endforeach()
      string(SHA256 GWLDCORE_SOURCE_TREE_SHA256
        "${GWLDCORE_SOURCE_MANIFEST}")
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

# The Linux GxE backend can embed one private vendor BLAS instance. Its symbols
# and mutable thread-pool state are then isolated from NumPy's BLAS, allowing
# the native executor to keep one fixed thread contract for the process.
# The accepted GxE backend is deliberately opt-in and requires explicit,
# immutable upstream-BLIS provenance.  A normal build uses the process BLAS;
# private OpenBLAS remains available only as an explicit compatibility option.
option(GXELDCORE_USE_PRIVATE_BLIS
  "Statically isolate an explicitly supplied upstream BLIS inside the Linux GxE extension"
  OFF)
set(GXELDCORE_PRIVATE_BLIS_ARCHIVE "" CACHE FILEPATH
  "PIC static upstream BLIS archive for an isolated Linux GxE runtime")
set(GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR "" CACHE PATH
  "Include root containing blis.h, cblas.h, and blis/{blis,cblas}.h")
set(GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT "" CACHE STRING
  "Exact 40-hex upstream BLIS source commit")
set(GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256 "" CACHE STRING
  "Exact SHA-256 identity of the upstream BLIS source tree")
set(GXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY "" CACHE STRING
  "Upstream BLIS configuration family (for example, zen)")

set(GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE "" CACHE FILEPATH
  "PIC static OpenBLAS archive for an isolated Linux GxE runtime")
set(_GXELDCORE_PRIVATE_OPENBLAS_DEFAULT OFF)
if (NOT GXELDCORE_USE_PRIVATE_BLIS AND
    CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BLA_VENDOR STREQUAL "OpenBLAS")
  if (NOT GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE)
    find_file(_GXELDCORE_DISCOVERED_OPENBLAS_ARCHIVE
      NAMES libopenblasp-r0.3.34.a libopenblas.a
      HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu")
    if (NOT _GXELDCORE_DISCOVERED_OPENBLAS_ARCHIVE AND DEFINED ENV{CONDA_PREFIX})
      file(GLOB _GXELDCORE_VERSIONED_OPENBLAS_ARCHIVES
        LIST_DIRECTORIES FALSE "$ENV{CONDA_PREFIX}/lib/libopenblasp-r*.a")
      list(SORT _GXELDCORE_VERSIONED_OPENBLAS_ARCHIVES COMPARE NATURAL ORDER DESCENDING)
      list(LENGTH _GXELDCORE_VERSIONED_OPENBLAS_ARCHIVES _GXELDCORE_ARCHIVE_COUNT)
      if (_GXELDCORE_ARCHIVE_COUNT GREATER 0)
        list(GET _GXELDCORE_VERSIONED_OPENBLAS_ARCHIVES 0
          _GXELDCORE_DISCOVERED_OPENBLAS_ARCHIVE)
      endif()
    endif()
    if (_GXELDCORE_DISCOVERED_OPENBLAS_ARCHIVE)
      set(GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE
        "${_GXELDCORE_DISCOVERED_OPENBLAS_ARCHIVE}" CACHE FILEPATH
        "PIC static OpenBLAS archive for an isolated Linux GxE runtime" FORCE)
    endif()
  endif()
endif()
option(GXELDCORE_USE_PRIVATE_OPENBLAS
  "Statically isolate OpenBLAS inside the Linux GxE extension"
  ${_GXELDCORE_PRIVATE_OPENBLAS_DEFAULT})
if (GXELDCORE_USE_PRIVATE_OPENBLAS AND GXELDCORE_USE_PRIVATE_BLIS)
  message(FATAL_ERROR
    "GXELDCORE_USE_PRIVATE_OPENBLAS and GXELDCORE_USE_PRIVATE_BLIS are mutually exclusive")
endif()
set(_GXELDCORE_GEMM_INTEGRITY_DEFAULT ON)
option(GXELDCORE_GEMM_INTEGRITY
  "Enable GxE immutable-input, NUMA-output, and optional checksum safeguards"
  ${_GXELDCORE_GEMM_INTEGRITY_DEFAULT})
if (GXELDCORE_USE_PRIVATE_BLIS)
  set(_GXELDCORE_GEMM_CHECKSUM_DEFAULT OFF)
else()
  set(_GXELDCORE_GEMM_CHECKSUM_DEFAULT ON)
endif()
option(GXELDCORE_GEMM_CHECKSUM
  "Enable redundant GEMM checksum recomputation and deterministic repair"
  ${_GXELDCORE_GEMM_CHECKSUM_DEFAULT})
if (GXELDCORE_GEMM_CHECKSUM AND NOT GXELDCORE_GEMM_INTEGRITY)
  message(FATAL_ERROR
    "GXELDCORE_GEMM_CHECKSUM requires GXELDCORE_GEMM_INTEGRITY=ON")
endif()
if (GXELDCORE_USE_PRIVATE_OPENBLAS AND NOT GXELDCORE_GEMM_INTEGRITY)
  message(FATAL_ERROR
    "Private OpenBLAS is a compatibility backend and requires GXELDCORE_GEMM_INTEGRITY=ON")
endif()
if (GXELDCORE_USE_PRIVATE_OPENBLAS AND NOT GXELDCORE_GEMM_CHECKSUM)
  message(FATAL_ERROR
    "Private OpenBLAS requires GXELDCORE_GEMM_CHECKSUM=ON")
endif()
set(GXELDCORE_PRIVATE_OPENBLAS_SHA256 "none")
set(GXELDCORE_PRIVATE_OPENBLAS_THREADING_LAYER "shared_or_unknown")
set(GXELDCORE_RECOMMENDED_OMP_WAIT_POLICY "PASSIVE")
set(GXELDCORE_PRIVATE_BLAS_ENABLED OFF)
set(GXELDCORE_PRIVATE_BLAS_BACKEND "none")
set(GXELDCORE_PRIVATE_BLAS_ARCHIVE "")
set(GXELDCORE_PRIVATE_BLAS_SHA256 "none")
set(GXELDCORE_PRIVATE_BLAS_SOURCE_COMMIT "none")
set(GXELDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256 "none")
set(GXELDCORE_PRIVATE_BLAS_CONFIG_FAMILY "none")
set(GXELDCORE_PRIVATE_BLAS_HEADER_SHA256 "none")
set(GXELDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256 "none")
set(GXELDCORE_BLAS_VENDOR "${BLA_VENDOR}")
if (GXELDCORE_USE_PRIVATE_OPENBLAS)
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT BLA_VENDOR STREQUAL "OpenBLAS")
    message(FATAL_ERROR
      "GXELDCORE_USE_PRIVATE_OPENBLAS requires Linux and BLA_VENDOR=OpenBLAS")
  endif()
  if (NOT GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE)
    find_file(GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE
      NAMES libopenblasp-r0.3.34.a libopenblas.a
      HINTS "$ENV{CONDA_PREFIX}/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu")
  endif()
  if (NOT GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE OR
      NOT EXISTS "${GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE}")
    message(FATAL_ERROR
      "A PIC static OpenBLAS archive is required for private GxE linkage")
  endif()
  find_package(Threads REQUIRED)
  file(SHA256 "${GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE}"
    GXELDCORE_PRIVATE_OPENBLAS_SHA256)
  # The CLI must select its OpenMP wait policy before importing NumPy or this
  # extension. Detect the private archive's threading layer without loading it.
  # Both threading layers must sleep while Python performs streamed feature
  # reductions. ACTIVE makes an OpenMP BLAS team consume every assigned core at
  # its barrier between calls, starving the single calling thread for hours on
  # production-sized multi-environment runs. Runtime isolation and serialized
  # entry provide safety; busy waiting does not.
  find_program(GXELDCORE_GREP_EXECUTABLE grep REQUIRED)
  execute_process(
    COMMAND "${CMAKE_NM}" -u "${GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE}"
    COMMAND "${GXELDCORE_GREP_EXECUTABLE}" -m 1 -E
            "[[:space:]](GOMP_|omp_get_|omp_set_)"
    RESULT_VARIABLE GXELDCORE_PRIVATE_OPENBLAS_OPENMP_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)
  if (GXELDCORE_PRIVATE_OPENBLAS_OPENMP_RESULT EQUAL 0)
    set(GXELDCORE_PRIVATE_OPENBLAS_THREADING_LAYER "openmp")
  else()
    set(GXELDCORE_PRIVATE_OPENBLAS_THREADING_LAYER "pthreads")
  endif()
  message(STATUS
    "Using private GxE OpenBLAS archive: ${GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE}")
  message(STATUS
    "Private GxE OpenBLAS threading layer: ${GXELDCORE_PRIVATE_OPENBLAS_THREADING_LAYER}")
  set(GXELDCORE_PRIVATE_BLAS_ENABLED ON)
  set(GXELDCORE_PRIVATE_BLAS_BACKEND "openblas")
  set(GXELDCORE_PRIVATE_BLAS_ARCHIVE
    "${GXELDCORE_PRIVATE_OPENBLAS_ARCHIVE}")
  set(GXELDCORE_PRIVATE_BLAS_SHA256
    "${GXELDCORE_PRIVATE_OPENBLAS_SHA256}")
endif()
if (GXELDCORE_USE_PRIVATE_BLIS)
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GXELDCORE_USE_PRIVATE_BLIS requires Linux")
  endif()
  if (NOT GXELDCORE_GEMM_INTEGRITY)
    message(FATAL_ERROR
      "The upstream BLIS candidate requires GXELDCORE_GEMM_INTEGRITY=ON")
  endif()
  if (NOT GXELDCORE_PRIVATE_BLIS_ARCHIVE OR
      NOT EXISTS "${GXELDCORE_PRIVATE_BLIS_ARCHIVE}")
    message(FATAL_ERROR
      "GXELDCORE_PRIVATE_BLIS_ARCHIVE must name an existing PIC static archive")
  endif()
  find_package(Threads REQUIRED)
  if (NOT GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR OR
      NOT EXISTS "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}/blis.h" OR
      NOT EXISTS "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}/cblas.h" OR
      NOT EXISTS "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}/blis/blis.h" OR
      NOT EXISTS "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}/blis/cblas.h")
    message(FATAL_ERROR
      "GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR lacks the required installed BLIS headers")
  endif()
  string(LENGTH "${GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT}"
    _GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT_LENGTH)
  if (NOT _GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT_LENGTH EQUAL 40 OR
      NOT GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT must be exactly 40 lowercase hex characters")
  endif()
  string(LENGTH "${GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256}"
    _GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256_LENGTH)
  if (NOT _GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256_LENGTH EQUAL 64 OR
      NOT GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256 MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256 must be exactly 64 lowercase hex characters")
  endif()
  if (NOT GXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY OR
      NOT GXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY MATCHES "^[A-Za-z0-9_.+-]+$")
    message(FATAL_ERROR
      "GXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY must be an explicit configuration name")
  endif()

  file(SHA256 "${GXELDCORE_PRIVATE_BLIS_ARCHIVE}"
    GXELDCORE_PRIVATE_BLAS_SHA256)
  file(SHA256 "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}/blis/blis.h"
    GXELDCORE_PRIVATE_BLAS_HEADER_SHA256)
  file(SHA256 "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}/blis/cblas.h"
    GXELDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256)
  find_program(GXELDCORE_GREP_EXECUTABLE grep REQUIRED)
  foreach(_GXELDCORE_BLIS_SYMBOL IN ITEMS
      cblas_dgemm
      cblas_sgemm
      bli_arch_query_id
      bli_arch_string
      bli_info_get_version_str
      bli_info_get_enable_pthreads
      bli_info_get_enable_pthreads_as_default
      bli_info_get_enable_tls
      bli_thread_get_num_threads
      bli_thread_get_thread_impl
      bli_thread_get_thread_impl_str
      bli_thread_get_jc_nt
      bli_thread_get_pc_nt
      bli_thread_get_ic_nt
      bli_thread_get_jr_nt
      bli_thread_get_ir_nt
      bli_thread_set_num_threads
      bli_thread_set_thread_impl
      bli_thread_set_ways)
    execute_process(
      COMMAND "${CMAKE_NM}" -g --defined-only "${GXELDCORE_PRIVATE_BLIS_ARCHIVE}"
      COMMAND "${GXELDCORE_GREP_EXECUTABLE}" -q -E
              "[[:space:]]${_GXELDCORE_BLIS_SYMBOL}$"
      RESULT_VARIABLE _GXELDCORE_BLIS_SYMBOL_RESULT
      ERROR_QUIET)
    if (NOT _GXELDCORE_BLIS_SYMBOL_RESULT EQUAL 0)
      message(FATAL_ERROR
        "Private BLIS archive lacks required symbol ${_GXELDCORE_BLIS_SYMBOL}")
    endif()
  endforeach()
  execute_process(
    COMMAND "${CMAKE_NM}" -u "${GXELDCORE_PRIVATE_BLIS_ARCHIVE}"
    COMMAND "${GXELDCORE_GREP_EXECUTABLE}" -q -E
            "[[:space:]]pthread_create$"
    RESULT_VARIABLE _GXELDCORE_PRIVATE_BLIS_PTHREAD_RESULT
    ERROR_QUIET)
  if (NOT _GXELDCORE_PRIVATE_BLIS_PTHREAD_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Private BLIS archive must use the pthreads threading implementation")
  endif()
  execute_process(
    COMMAND "${CMAKE_NM}" -u "${GXELDCORE_PRIVATE_BLIS_ARCHIVE}"
    COMMAND "${GXELDCORE_GREP_EXECUTABLE}" -q -E
            "[[:space:]](GOMP_|omp_get_|omp_set_)"
    RESULT_VARIABLE _GXELDCORE_PRIVATE_BLIS_OPENMP_RESULT
    ERROR_QUIET)
  if (_GXELDCORE_PRIVATE_BLIS_OPENMP_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Private pthreads BLIS archive unexpectedly depends on OpenMP")
  endif()

  set(GXELDCORE_PRIVATE_BLAS_ENABLED ON)
  set(GXELDCORE_PRIVATE_BLAS_BACKEND "upstream_blis")
  set(GXELDCORE_PRIVATE_BLAS_ARCHIVE
    "${GXELDCORE_PRIVATE_BLIS_ARCHIVE}")
  set(GXELDCORE_PRIVATE_BLAS_SOURCE_COMMIT
    "${GXELDCORE_PRIVATE_BLIS_SOURCE_COMMIT}")
  set(GXELDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256
    "${GXELDCORE_PRIVATE_BLIS_SOURCE_TREE_SHA256}")
  set(GXELDCORE_PRIVATE_BLAS_CONFIG_FAMILY
    "${GXELDCORE_PRIVATE_BLIS_CONFIG_FAMILY}")
  set(GXELDCORE_BLAS_VENDOR "BLIS")
  message(STATUS
    "Using private upstream BLIS archive: ${GXELDCORE_PRIVATE_BLIS_ARCHIVE}")
  message(STATUS
    "Private upstream BLIS archive SHA-256: ${GXELDCORE_PRIVATE_BLAS_SHA256}")
endif()
if (BLA_VENDOR STREQUAL "OpenBLAS" AND
    NOT GXELDCORE_USE_PRIVATE_OPENBLAS AND
    NOT GXELDCORE_USE_PRIVATE_BLIS AND
    (NOT GXELDCORE_GEMM_INTEGRITY OR NOT GXELDCORE_GEMM_CHECKSUM))
  message(FATAL_ERROR
    "Unchecked GxE GEMMs cannot use process-shared OpenBLAS; enable the private "
    "runtime or the integrity fallback")
endif()

# =====================================================================================
# Optional OpenMP
# =====================================================================================
option(GWLDCORE_USE_OPENMP "Enable OpenMP if available" ON)
if (GWLDCORE_USE_OPENMP)
  find_package(OpenMP)
endif()
if (GXELDCORE_USE_PRIVATE_BLIS AND NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR
    "The private upstream BLIS path requires OpenMP for application-level loops")
endif()

# =====================================================================================
# CPU tuning options
# =====================================================================================
option(GWLDCORE_ENABLE_NATIVE_OPT "Enable -march=native / architecture-specific local tuning" ON)
option(GWLDCORE_ENABLE_ASAN_UBSAN
  "Build all native targets with AddressSanitizer and UndefinedBehaviorSanitizer"
  OFF)
option(GWLDCORE_ENABLE_UBSAN_ONLY
  "Build all native targets with UndefinedBehaviorSanitizer only"
  OFF)

if (GWLDCORE_ENABLE_ASAN_UBSAN AND GWLDCORE_ENABLE_UBSAN_ONLY)
  message(FATAL_ERROR
    "GWLDCORE_ENABLE_ASAN_UBSAN and GWLDCORE_ENABLE_UBSAN_ONLY are mutually exclusive")
endif()

set(GWLDCORE_SANITIZER_ENABLED OFF)
set(GWLDCORE_SANITIZER_FLAGS "")
set(GWLDCORE_SANITIZER_MODE "none")
set(GWLDCORE_ASAN_ENABLED OFF)
set(GWLDCORE_UBSAN_ENABLED OFF)
if (GWLDCORE_ENABLE_ASAN_UBSAN OR GWLDCORE_ENABLE_UBSAN_ONLY)
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    message(FATAL_ERROR
      "SUMMIT sanitizer builds require a GNU- or Clang-compatible C++ compiler")
  endif()
  set(GWLDCORE_SANITIZER_ENABLED ON)
  set(GWLDCORE_UBSAN_ENABLED ON)
  if (GWLDCORE_ENABLE_ASAN_UBSAN)
    set(GWLDCORE_SANITIZER_FLAGS "-fsanitize=address,undefined")
    set(GWLDCORE_SANITIZER_MODE "asan_ubsan")
    set(GWLDCORE_ASAN_ENABLED ON)
    message(STATUS "SUMMIT native sanitizer mode: ASan+UBSan")
  else()
    set(GWLDCORE_SANITIZER_FLAGS "-fsanitize=undefined")
    set(GWLDCORE_SANITIZER_MODE "ubsan_only")
    message(STATUS "SUMMIT native sanitizer mode: UBSan-only")
  endif()
  if (GWLDCORE_ENABLE_NATIVE_OPT)
    message(STATUS "Disabling GWLDCORE_ENABLE_NATIVE_OPT for sanitizer build")
    set(GWLDCORE_ENABLE_NATIVE_OPT OFF CACHE BOOL
      "Enable -march=native / architecture-specific local tuning" FORCE)
  endif()
endif()

# Freeze the effective SUMMIT target policy into native build provenance.  The
# compiler flag string deliberately covers CMake's global/configuration flags
# plus the options applied by gwldcore_configure_module(); automatically added
# include paths, definitions, and ABI flags are reported through their own
# build_info fields rather than being mislabeled as a captured command line.
set(GWLDCORE_NATIVE_TUNING_ENABLED OFF)
if (MSVC)
  set(GWLDCORE_EFFECTIVE_OPTIMIZATION "/O2")
  set(GWLDCORE_ARCHITECTURE_TUNING "portable")
  set(GWLDCORE_TARGET_COMPILER_FLAGS "/O2")
elseif (GWLDCORE_SANITIZER_ENABLED)
  set(GWLDCORE_EFFECTIVE_OPTIMIZATION "-O1")
  set(GWLDCORE_ARCHITECTURE_TUNING "portable")
  set(GWLDCORE_TARGET_COMPILER_FLAGS
    "-O1 -g -fno-omit-frame-pointer -fno-sanitize-recover=all ${GWLDCORE_SANITIZER_FLAGS}")
else()
  set(GWLDCORE_EFFECTIVE_OPTIMIZATION "-O3")
  set(GWLDCORE_TARGET_COMPILER_FLAGS "-O3")
  if (GWLDCORE_ENABLE_NATIVE_OPT)
    set(GWLDCORE_NATIVE_TUNING_ENABLED ON)
    set(GWLDCORE_ARCHITECTURE_TUNING "-march=native")
    string(APPEND GWLDCORE_TARGET_COMPILER_FLAGS " -march=native")
  else()
    set(GWLDCORE_ARCHITECTURE_TUNING "portable")
  endif()
endif()

set(GWLDCORE_CONFIGURED_COMPILER_FLAGS "${CMAKE_CXX_FLAGS}")
if (CMAKE_CONFIGURATION_TYPES)
  foreach(_gwldcore_config IN ITEMS Debug Release MinSizeRel RelWithDebInfo)
    string(TOUPPER "${_gwldcore_config}" _gwldcore_config_upper)
    set(_gwldcore_config_flags
      "${CMAKE_CXX_FLAGS_${_gwldcore_config_upper}}")
    string(REPLACE ";" " " _gwldcore_config_flags
      "${_gwldcore_config_flags}")
    if (NOT _gwldcore_config_flags STREQUAL "")
      string(APPEND GWLDCORE_CONFIGURED_COMPILER_FLAGS
        " $<$<CONFIG:${_gwldcore_config}>:${_gwldcore_config_flags}>")
    endif()
  endforeach()
elseif (CMAKE_BUILD_TYPE)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _gwldcore_build_type_upper)
  set(_gwldcore_config_flags
    "${CMAKE_CXX_FLAGS_${_gwldcore_build_type_upper}}")
  string(REPLACE ";" " " _gwldcore_config_flags
    "${_gwldcore_config_flags}")
  string(APPEND GWLDCORE_CONFIGURED_COMPILER_FLAGS
    " ${_gwldcore_config_flags}")
endif()
string(APPEND GWLDCORE_CONFIGURED_COMPILER_FLAGS
  " ${GWLDCORE_TARGET_COMPILER_FLAGS}")
string(STRIP "${GWLDCORE_CONFIGURED_COMPILER_FLAGS}"
  GWLDCORE_CONFIGURED_COMPILER_FLAGS)
string(REPLACE "\\" "\\\\" GWLDCORE_CONFIGURED_COMPILER_FLAGS_ESCAPED
  "${GWLDCORE_CONFIGURED_COMPILER_FLAGS}")
string(REPLACE "\"" "\\\"" GWLDCORE_CONFIGURED_COMPILER_FLAGS_ESCAPED
  "${GWLDCORE_CONFIGURED_COMPILER_FLAGS_ESCAPED}")

function(gwldcore_apply_sanitizer target_name)
  if (GWLDCORE_SANITIZER_ENABLED)
    target_compile_options(${target_name} PRIVATE
      -O1
      -g
      -fno-omit-frame-pointer
      -fno-sanitize-recover=all
      ${GWLDCORE_SANITIZER_FLAGS})
    target_link_options(${target_name} PRIVATE
      -fno-omit-frame-pointer
      -fno-sanitize-recover=all
      ${GWLDCORE_SANITIZER_FLAGS})
  endif()
endfunction()

set(SUMMIT_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/summit")
set(SUMMIT_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/native")
set(SUMMIT_NATIVE_COMMON_DIR "${SUMMIT_NATIVE_DIR}/common")
configure_file(
  "${SUMMIT_PYTHON_PACKAGE_DIR}/_native_build_config.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_native_build_config.py"
  @ONLY)

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
elseif (GWLDCORE_SANITIZER_ENABLED)
  gwldcore_apply_sanitizer(gwldcore_core)
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
  elseif (GWLDCORE_SANITIZER_ENABLED)
    gwldcore_apply_sanitizer(${target_name})
  else()
    target_compile_options(${target_name} PRIVATE -O3)
    if (GWLDCORE_ENABLE_NATIVE_OPT)
      target_compile_options(${target_name} PRIVATE -march=native)
    endif()
  endif()

  if (GXELDCORE_PRIVATE_BLAS_ENABLED AND target_name STREQUAL "gxeldcore")
    target_compile_definitions(${target_name} PRIVATE
      GWLDCORE_USE_GENERIC_CBLAS
      GWLDCORE_USE_PRIVATE_VENDOR_BLAS)
    if (GXELDCORE_USE_PRIVATE_OPENBLAS)
      target_compile_definitions(${target_name} PRIVATE
        GWLDCORE_USE_OPENBLAS
        GWLDCORE_PRIVATE_OPENBLAS)
      if (DEFINED OPENBLAS_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE "${OPENBLAS_INCLUDE_DIR}")
      else()
        find_path(CBLAS_INCLUDE_DIR cblas.h
          HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
        if (CBLAS_INCLUDE_DIR)
          target_include_directories(${target_name} PRIVATE "${CBLAS_INCLUDE_DIR}")
        endif()
      endif()
    else()
      target_compile_definitions(${target_name} PRIVATE
        GWLDCORE_USE_BLIS
        GWLDCORE_PRIVATE_BLIS)
      # BEFORE prevents a process/shared cblas.h from silently overriding the
      # header paired with the attested private archive.
      target_include_directories(${target_name} BEFORE PRIVATE
        "${GXELDCORE_PRIVATE_BLIS_INCLUDE_DIR}")
    endif()
    target_link_libraries(${target_name} PRIVATE
      "${GXELDCORE_PRIVATE_BLAS_ARCHIVE}"
      Threads::Threads
      ${CMAKE_DL_LIBS}
      m)
    # Bind every private-vendor reference to this extension's archive and hide
    # all archive symbols from the process-wide dynamic namespace. NumPy and
    # threadpoolctl cannot resolve or mutate this runtime.
    target_link_options(${target_name} PRIVATE
      "LINKER:-Bsymbolic"
      "LINKER:--exclude-libs,ALL")
  endif()

  if (GXELDCORE_PRIVATE_BLAS_ENABLED AND target_name STREQUAL "gxeldcore")
    # The private archive was fully linked above.  Do not also put the shared
    # process BLAS on this extension's link line.
  elseif (APPLE)
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
  if (GXELDCORE_GEMM_INTEGRITY)
    target_compile_definitions(gxeldcore PRIVATE GWLDCORE_GEMM_INTEGRITY)
  endif()
  if (GXELDCORE_GEMM_CHECKSUM)
    target_compile_definitions(gxeldcore PRIVATE GWLDCORE_GEMM_CHECKSUM)
  endif()
  target_compile_definitions(gxeldcore PRIVATE
    GWLDCORE_SOURCE_COMMIT="${GWLDCORE_SOURCE_COMMIT}"
    GWLDCORE_SOURCE_TREE_SHA256="${GWLDCORE_SOURCE_TREE_SHA256}"
    GWLDCORE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    GWLDCORE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    GWLDCORE_BUILD_TYPE="$<CONFIG>"
    GWLDCORE_SANITIZER_MODE="${GWLDCORE_SANITIZER_MODE}"
    GWLDCORE_ASAN_ENABLED=$<BOOL:${GWLDCORE_ASAN_ENABLED}>
    GWLDCORE_UBSAN_ENABLED=$<BOOL:${GWLDCORE_UBSAN_ENABLED}>
    GWLDCORE_EFFECTIVE_OPTIMIZATION="${GWLDCORE_EFFECTIVE_OPTIMIZATION}"
    GWLDCORE_ARCHITECTURE_TUNING="${GWLDCORE_ARCHITECTURE_TUNING}"
    "GWLDCORE_CONFIGURED_COMPILER_FLAGS=\"${GWLDCORE_CONFIGURED_COMPILER_FLAGS_ESCAPED}\""
    GWLDCORE_BLAS_VENDOR="${GXELDCORE_BLAS_VENDOR}"
    GWLDCORE_GEMM_INTEGRITY_ENABLED=$<BOOL:${GXELDCORE_GEMM_INTEGRITY}>
    GWLDCORE_GEMM_CHECKSUM_ENABLED=$<BOOL:${GXELDCORE_GEMM_CHECKSUM}>
    GWLDCORE_PRIVATE_BLAS_ENABLED=$<BOOL:${GXELDCORE_PRIVATE_BLAS_ENABLED}>
    GWLDCORE_PRIVATE_BLAS_BACKEND="${GXELDCORE_PRIVATE_BLAS_BACKEND}"
    GWLDCORE_PRIVATE_BLAS_SHA256="${GXELDCORE_PRIVATE_BLAS_SHA256}"
    GWLDCORE_PRIVATE_BLAS_SOURCE_COMMIT="${GXELDCORE_PRIVATE_BLAS_SOURCE_COMMIT}"
    GWLDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256="${GXELDCORE_PRIVATE_BLAS_SOURCE_TREE_SHA256}"
    GWLDCORE_PRIVATE_BLAS_CONFIG_FAMILY="${GXELDCORE_PRIVATE_BLAS_CONFIG_FAMILY}"
    GWLDCORE_PRIVATE_BLAS_HEADER_SHA256="${GXELDCORE_PRIVATE_BLAS_HEADER_SHA256}"
    GWLDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256="${GXELDCORE_PRIVATE_BLAS_CBLAS_HEADER_SHA256}"
    GWLDCORE_PRIVATE_OPENBLAS_ENABLED=$<BOOL:${GXELDCORE_USE_PRIVATE_OPENBLAS}>
    GWLDCORE_PRIVATE_OPENBLAS_SHA256="${GXELDCORE_PRIVATE_OPENBLAS_SHA256}"
    GWLDCORE_NATIVE_OPT=$<BOOL:${GWLDCORE_NATIVE_TUNING_ENABLED}>
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
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_native_build_config.py"
  DESTINATION summit)
