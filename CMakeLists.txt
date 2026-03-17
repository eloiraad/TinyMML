cmake_minimum_required(VERSION 3.15)
project(cvmml VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB_RECURSE CORE_SOURCES "sources/core/*.cpp" "sources/core/*.cu")
add_library(cvmml_core OBJECT ${CORE_SOURCES})
target_include_directories(cvmml_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)
set_target_properties(cvmml_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(cvmml_api sources/api/bindings.cpp)

target_link_libraries(cvmml_api PRIVATE cvmml_core)
target_include_directories(cvmml_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes)

set_target_properties(cvmml_api PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)