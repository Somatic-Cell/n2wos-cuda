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
