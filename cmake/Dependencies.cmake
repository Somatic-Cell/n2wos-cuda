# Optional third-party dependencies for production-candidate backends.
#
# 0002 deliberately keeps cuBQL optional. The repository does not vendor it in
# this patch; fetch it with scripts/fetch_cubql.py or place a checkout at
# external/cuBQL, then configure with -DN2WOS_ENABLE_CUBQL=ON.

option(N2WOS_ENABLE_CUBQL "Enable NVIDIA cuBQL CUDA BVH backend" OFF)
set(N2WOS_CUBQL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/cuBQL" CACHE PATH "Path to a local NVIDIA/cuBQL checkout")

set(N2WOS_HAS_CUBQL OFF)
if(N2WOS_ENABLE_CUBQL)
  if(NOT EXISTS "${N2WOS_CUBQL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "N2WOS_ENABLE_CUBQL=ON but cuBQL was not found at ${N2WOS_CUBQL_DIR}.\n"
      "Run: python3 scripts/fetch_cubql.py --dest external/cuBQL\n"
      "or set -DN2WOS_CUBQL_DIR=/path/to/cuBQL")
  endif()

  add_subdirectory("${N2WOS_CUBQL_DIR}" "${CMAKE_BINARY_DIR}/external/cuBQL" EXCLUDE_FROM_ALL)
  if(NOT TARGET cuBQL)
    message(FATAL_ERROR "cuBQL checkout did not define the expected CMake target 'cuBQL'")
  endif()
  set(N2WOS_HAS_CUBQL ON)
endif()


# Optional tiny-cuda-nn dependency for native C++/CUDA cache training and inference probes.
# 0003 keeps it optional and does not use Python bindings. Fetch it with
# scripts/fetch_tcnn.py, which must clone submodules recursively because
# tiny-cuda-nn requires CUTLASS, fmt, cmrc, and other bundled dependencies.
option(N2WOS_ENABLE_TCNN "Enable tiny-cuda-nn native C++/CUDA cache backend" OFF)
# Keep tiny-cuda-nn RTC support enabled by default even though the probe does
# not request JIT fusion at runtime unless --jit 1 is passed.  Current
# tiny-cuda-nn headers instantiate training/backward templates that reference
# CudaRtcKernel::set; that symbol is only defined by tiny-cuda-nn when
# TCNN_BUILD_WITH_RTC is enabled.  Disabling RTC therefore creates a link-time
# failure in native Trainer-based probes.
option(N2WOS_ENABLE_TCNN_RTC "Build tiny-cuda-nn with RTC/JIT support; runtime JIT is still controlled by --jit" ON)
set(N2WOS_TCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/tiny-cuda-nn" CACHE PATH "Path to a local NVlabs/tiny-cuda-nn checkout")

set(N2WOS_HAS_TCNN OFF)
if(N2WOS_ENABLE_TCNN)
  if(NOT EXISTS "${N2WOS_TCNN_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "N2WOS_ENABLE_TCNN=ON but tiny-cuda-nn was not found at ${N2WOS_TCNN_DIR}.\n"
      "Run: python3 scripts/fetch_tcnn.py --dest external/tiny-cuda-nn\n"
      "or set -DN2WOS_TCNN_DIR=/path/to/tiny-cuda-nn")
  endif()
  if(NOT EXISTS "${N2WOS_TCNN_DIR}/dependencies/cutlass/CMakeLists.txt")
    message(FATAL_ERROR
      "tiny-cuda-nn dependencies appear to be missing.\n"
      "Run: python3 scripts/fetch_tcnn.py --dest external/tiny-cuda-nn --update\n"
      "or inside the checkout: git submodule update --init --recursive")
  endif()

  set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(TCNN_BUILD_WITH_RTC ${N2WOS_ENABLE_TCNN_RTC} CACHE BOOL "" FORCE)
  message(STATUS "N2WOS tiny-cuda-nn RTC support: ${N2WOS_ENABLE_TCNN_RTC}")
  set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF CACHE BOOL "" FORCE)
  set(TCNN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)

  add_subdirectory("${N2WOS_TCNN_DIR}" "${CMAKE_BINARY_DIR}/external/tiny-cuda-nn" EXCLUDE_FROM_ALL)
  if(NOT TARGET tiny-cuda-nn)
    message(FATAL_ERROR "tiny-cuda-nn checkout did not define the expected CMake target 'tiny-cuda-nn'")
  endif()
  set(N2WOS_HAS_TCNN ON)
endif()
