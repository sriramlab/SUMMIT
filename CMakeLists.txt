cmake_minimum_required(VERSION 3.21)
project(gwldcore LANGUAGES CXX)

# Require modern enough GCC for your code
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR
      "ERROR: GCC >= 12.0 is required (found ${CMAKE_CXX_COMPILER_VERSION}).")
  endif()
endif()

# Use modern Python finder through pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# ---------------- BLAS autodetect & override knobs ----------------
# Absolute override: let user point straight at a .so (OpenBLAS or mkl_rt)
set(USER_BLAS_LIB "" CACHE FILEPATH
  "Absolute path to a BLAS library (.so), e.g. /path/to/libopenblas.so or libmkl_rt.so")

# Respect a user-specified BLA_VENDOR if provided, otherwise auto-pick
# Valid values include: Intel10_64lp, Intel10_64lp_seq, OpenBLAS, Generic, ...
set(BLA_VENDOR "${BLA_VENDOR}" CACHE STRING
  "BLAS vendor hint. Leave empty for auto-select.")

# Auto-pick vendor only if USER_BLAS_LIB is not given and BLA_VENDOR not preset
if (NOT USER_BLAS_LIB AND NOT BLA_VENDOR)
  if (DEFINED ENV{MKLROOT})
    set(BLA_VENDOR Intel10_64lp)
  elseif (EXISTS "$ENV{CONDA_PREFIX}/lib/libmkl_rt.so")
    set(BLA_VENDOR Intel10_64lp)
  elseif (EXISTS "/opt/intel/mkl/lib/intel64/libmkl_rt.so")
    set(BLA_VENDOR Intel10_64lp)
  elseif (EXISTS "$ENV{CONDA_PREFIX}/lib/libopenblas.so"
          OR EXISTS "$ENV{CONDA_PREFIX}/lib/libopenblas.so.0"
          OR EXISTS "/usr/lib/x86_64-linux-gnu/libopenblas.so")
    set(BLA_VENDOR OpenBLAS)
  else()
    set(BLA_VENDOR Generic)
  endif()
  message(STATUS "Auto-selected BLA_VENDOR=${BLA_VENDOR} (override with -DBLA_VENDOR=...)")
endif()

# Resolve BLAS
if (USER_BLAS_LIB)
  add_library(BLAS::BLAS UNKNOWN IMPORTED)
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_LOCATION "${USER_BLAS_LIB}")
  set(GWLDCORE_BLAS_TARGET BLAS::BLAS)
else()
  set(BLA_STATIC OFF CACHE BOOL "" FORCE)
  find_package(BLAS REQUIRED)  # honors BLA_VENDOR if set
  if (TARGET BLAS::BLAS)
    set(GWLDCORE_BLAS_TARGET BLAS::BLAS)
  else()
    add_library(BLAS::BLAS UNKNOWN IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES
      IMPORTED_LOCATION "${BLAS_LIBRARIES}")
    set(GWLDCORE_BLAS_TARGET BLAS::BLAS)
  endif()
endif()

# ---------------- Build extension ----------------
pybind11_add_module(gwldcore src/gwldcore.cpp)
target_compile_features(gwldcore PUBLIC cxx_std_17)
target_compile_options(gwldcore PRIVATE -O3 -march=native)

# Tell the code which CBLAS header to include
# If we selected an Intel vendor, prefer MKL's CBLAS header; else generic cblas.h
if (BLA_VENDOR MATCHES "Intel")
  target_compile_definitions(gwldcore PRIVATE GWLDCORE_USE_MKL_CBLAS=1)
  # Add MKL include dir if MKLROOT is present (helps when system doesn't ship generic cblas.h)
  if (DEFINED ENV{MKLROOT})
    target_include_directories(gwldcore PRIVATE "$ENV{MKLROOT}/include")
  endif()
else()
  # Try to add a path that likely contains cblas.h (Conda or system)
  find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS "$ENV{CONDA_PREFIX}/include" "/usr/include" "/usr/local/include")
  if (CBLAS_INCLUDE_DIR)
    target_include_directories(gwldcore PRIVATE "${CBLAS_INCLUDE_DIR}")
  endif()
endif()

target_link_libraries(gwldcore PRIVATE ${GWLDCORE_BLAS_TARGET})

install(TARGETS gwldcore
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .)
