#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "n2wos/cuda_bvh.hpp"
#include "n2wos/types.hpp"

namespace n2wos {

// NVIDIA cuBQL-backed triangle closest-point backend.
//
// This backend is optional and is built only when N2WOS_ENABLE_CUBQL=ON. It is
// intentionally wrapped behind the same device-pointer query shape as the
// in-tree median BVH backend so later WoS/wavefront kernels can avoid host
// transfers independent of which backend is selected.
class CuBqlBvh {
 public:
  CuBqlBvh();
  explicit CuBqlBvh(const Mesh& mesh,
                    int leaf_size = 8,
                    const std::string& build_method = "spatial_median");
  ~CuBqlBvh();

  CuBqlBvh(const CuBqlBvh&) = delete;
  CuBqlBvh& operator=(const CuBqlBvh&) = delete;
  CuBqlBvh(CuBqlBvh&&) noexcept;
  CuBqlBvh& operator=(CuBqlBvh&&) noexcept;

  CudaBvhQueryResult query(const std::vector<Vec3f>& points, int block_size = 128) const;

  void query_device(const DeviceVec3* d_points,
                    int query_count,
                    float* d_distance2,
                    DeviceVec3* d_closest,
                    int* d_triangle_id,
                    int* d_overflow,
                    int block_size = 128,
                    cudaStream_t stream = 0) const;

  // Wavefront variant. Slots with d_active[i] == 0 skip BVH traversal and
  // produce neutral outputs. This avoids per-step active-count readback while
  // keeping walker state resident on the GPU.
  void query_device_masked(const DeviceVec3* d_points,
                           const std::uint8_t* d_active,
                           int query_count,
                           float* d_distance2,
                           DeviceVec3* d_closest,
                           int* d_triangle_id,
                           int* d_overflow,
                           int block_size = 128,
                           cudaStream_t stream = 0) const;

  std::size_t triangle_count() const;
  std::size_t node_count() const;
  std::size_t prim_id_count() const;
  int leaf_size() const;
  float build_milliseconds() const;
  std::string build_method() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace n2wos
