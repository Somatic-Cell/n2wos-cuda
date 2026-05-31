cmake_minimum_required(VERSION 3.24)

project(n2wos_cuda
  VERSION 0.1.0
  DESCRIPTION "CUDA-resident Walk-on-Spheres geometry bootstrap"
  LANGUAGES CXX CUDA)

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
include(cmake/Dependencies.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # RTX 3070 is sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> if needed.
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

add_library(n2wos_core STATIC
  src/mesh/obj_loader.cpp
  src/mesh/procedural_bumpy_sphere.cpp
  src/mesh/normalize.cpp
  src/geometry/closest_point.cpp)

target_include_directories(n2wos_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(n2wos_core PUBLIC cxx_std_17)

add_library(n2wos_cuda STATIC
  src/cuda/bvh_build_host.cpp
  src/cuda/bvh_query.cu)

target_include_directories(n2wos_cuda
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

target_link_libraries(n2wos_cuda
  PUBLIC
    n2wos_core
    CUDA::cudart)

target_compile_features(n2wos_cuda PUBLIC cxx_std_17)
set_target_properties(n2wos_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

target_compile_options(n2wos_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

if(N2WOS_HAS_CUBQL)
  add_library(n2wos_cubql STATIC
    src/cuda/cubql_bvh.cu)

  target_include_directories(n2wos_cubql
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

  target_link_libraries(n2wos_cubql
    PUBLIC
      n2wos_core
      CUDA::cudart
      cuBQL)

  target_compile_features(n2wos_cubql PUBLIC cxx_std_17)
  target_compile_definitions(n2wos_cubql PUBLIC N2WOS_HAS_CUBQL=1)
  set_target_properties(n2wos_cubql PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

  target_compile_options(n2wos_cubql PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

add_executable(n2wos_probe_cuda_bvh
  src/tools/probe_cuda_bvh.cpp)

target_link_libraries(n2wos_probe_cuda_bvh
  PRIVATE
    n2wos_cuda)

install(TARGETS n2wos_probe_cuda_bvh RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


add_executable(n2wos_probe_geometry_backends
  src/tools/probe_geometry_backends.cu)

target_link_libraries(n2wos_probe_geometry_backends
  PRIVATE
    n2wos_cuda)

if(N2WOS_HAS_CUBQL)
  target_link_libraries(n2wos_probe_geometry_backends PRIVATE n2wos_cubql)
  target_compile_definitions(n2wos_probe_geometry_backends PRIVATE N2WOS_HAS_CUBQL=1)
endif()

set_target_properties(n2wos_probe_geometry_backends PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

target_compile_options(n2wos_probe_geometry_backends PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

install(TARGETS n2wos_probe_geometry_backends RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
