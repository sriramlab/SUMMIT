cmake_minimum_required(VERSION 3.21)
project(gwldcore LANGUAGES CXX)

cmake_minimum_required(VERSION 3.21)
project(gwldcore LANGUAGES CXX)

# Silence pybind11 legacy finder warning and use modern FindPython
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# --- BLAS detection (OpenBLAS path or FindBLAS) ---
# Let the user bypass FindBLAS with an absolute .so
set(USER_BLAS_LIB "" CACHE STRING "Absolute path to a BLAS library (.so), e.g. /path/to/libopenblas.so")

if (USER_BLAS_LIB)
  message(STATUS "Using user-provided BLAS: ${USER_BLAS_LIB}")
  set(GWLDCORE_BLAS_LIBS "${USER_BLAS_LIB}")
else()
  find_package(BLAS)
  if (BLAS_FOUND AND BLAS_LIBRARIES)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
    set(GWLDCORE_BLAS_LIBS "${BLAS_LIBRARIES}")
  else()
    message(FATAL_ERROR "BLAS not found. Provide -DUSER_BLAS_LIB=/full/path/to/libopenblas*.so")
  endif()
endif()

pybind11_add_module(gwldcore src/gwldcore.cpp)
target_compile_features(gwldcore PUBLIC cxx_std_17)
target_compile_options(gwldcore PRIVATE -O3 -march=native)

target_link_libraries(gwldcore PRIVATE ${GWLDCORE_BLAS_LIBS})


install(TARGETS gwldcore
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .)
