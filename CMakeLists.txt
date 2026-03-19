cmake_minimum_required(VERSION 3.15)
project(cvmml VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CVMML_ENABLE_OPENMP "Enable OpenMP for CPU tensor parallelism" ON)

set(CORE_SOURCES
	sources/core/tensor_cuda.cu
	sources/core/tensor/tensor_broadcast.cpp
	sources/core/tensor/tensor_memory.cpp
	sources/core/tensor/tensor_ops_elementwise.cpp
	sources/core/tensor/tensor_ops_reduction.cpp
	sources/core/tensor/tensor_ops_linalg.cpp
	sources/core/tensor/tensor_ops_conv.cpp
	sources/core/tensor/tensor_autograd.cpp
	sources/core/cuda/cuda_kernels_elementwise.cu
	sources/core/cuda/cuda_kernels_backward.cu
	sources/core/cuda/cuda_kernels_reduction.cu
	sources/core/cuda/cuda_kernels_advanced.cu
	sources/core/cuda/cuda_kernels_conv.cu
)
add_library(cvmml_core OBJECT ${CORE_SOURCES})
target_include_directories(cvmml_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)
set_target_properties(cvmml_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CVMML_ENABLE_OPENMP)
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(cvmml_core PUBLIC OpenMP::OpenMP_CXX)
	endif()
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(cvmml_api sources/api/bindings.cpp)

target_link_libraries(cvmml_api PRIVATE cvmml_core)
target_include_directories(cvmml_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes)

set_target_properties(cvmml_api PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)