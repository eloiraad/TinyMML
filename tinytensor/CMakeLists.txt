cmake_minimum_required(VERSION 3.18)
project(tinytensor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TINYTENSOR_ENABLE_CUDA "Build the CUDA backend" OFF)
option(TINYTENSOR_ENABLE_OPENMP "Enable OpenMP for CPU tensor operations" ON)

set(TENSOR_SOURCES
    sources/core/tensor/tensor_autograd.cpp
    sources/core/tensor/tensor_broadcast.cpp
    sources/core/tensor/tensor_memory.cpp
    sources/core/tensor/tensor_ops_conv.cpp
    sources/core/tensor/tensor_ops_elementwise.cpp
    sources/core/tensor/tensor_ops_linalg.cpp
    sources/core/tensor/tensor_ops_reduction.cpp
)

if(TINYTENSOR_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "TINYTENSOR_ENABLE_CUDA=ON, but no CUDA compiler was found. "
            "Install the CUDA toolkit or configure a CPU build without --cuda."
        )
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND TENSOR_SOURCES
        sources/core/cuda/tensor_cuda.cu
        sources/core/cuda/cuda_kernels_advanced.cu
        sources/core/cuda/cuda_kernels_backward.cu
        sources/core/cuda/cuda_kernels_conv.cu
        sources/core/cuda/cuda_kernels_elementwise.cu
        sources/core/cuda/cuda_kernels_reduction.cu
    )
else()
    list(APPEND TENSOR_SOURCES sources/core/cuda/cuda_stubs.cpp)
endif()

add_library(tinytensor_core OBJECT ${TENSOR_SOURCES})
target_include_directories(tinytensor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)
target_compile_definitions(tinytensor_core
    PUBLIC TINYTENSOR_ENABLE_CUDA=$<BOOL:${TINYTENSOR_ENABLE_CUDA}>
)
target_compile_options(tinytensor_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(tinytensor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TINYTENSOR_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tinytensor_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "TinyMML OpenMP support: enabled")
    else()
        message(STATUS "TinyMML OpenMP support: unavailable; using single-thread CPU fallback")
    endif()
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tinytensor sources/api/bindings.cpp)
target_link_libraries(tinytensor PRIVATE tinytensor_core)
target_include_directories(tinytensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes)
target_compile_definitions(tinytensor
    PRIVATE TINYTENSOR_ENABLE_CUDA=$<BOOL:${TINYTENSOR_ENABLE_CUDA}>
)
target_compile_options(tinytensor PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(tinytensor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
