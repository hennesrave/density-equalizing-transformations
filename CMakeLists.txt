cmake_minimum_required(VERSION 3.18)
project(density_equalizing_transformations LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_library(density_equalizing_transformations
    source/density_equalizing_transformations.hpp
    source/density_equalizing_transformations.cu
)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using nanobind" ON)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_density_equalizing_transformations source/bindings.cpp)
    target_link_libraries(_density_equalizing_transformations PRIVATE density_equalizing_transformations)

    install(
        TARGETS _density_equalizing_transformations
        LIBRARY DESTINATION density_equalizing_transformations
    )
    install(
        FILES source/density_equalizing_transformations/__init__.py
        DESTINATION density_equalizing_transformations
    )
endif()