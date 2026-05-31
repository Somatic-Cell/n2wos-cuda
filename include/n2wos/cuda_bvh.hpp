#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "n2wos/types.hpp"
#include "bvh_types.cuh"

namespace n2wos {

struct HostBvhData {
  std::vector<DeviceTriangle> triangles;
  std::vector<DeviceBvhNode> nodes;
  std::vector<int> triangle_indices;
  int leaf_size = 8;
  int max_depth = 0;
};

HostBvhData build_host_bvh(const Mesh& mesh, int leaf_size);

struct CudaBvhQueryResult {
  std::vector<float> distance2;
  std::vector<Vec3f> closest;
  std::vector<int> triangle_id;
  float kernel_milliseconds = 0.0f;
  int overflow_count = 0;
};

class CudaBvh {
 public:
  CudaBvh() = default;
  explicit CudaBvh(const Mesh& mesh, int leaf_size = 8);
  ~CudaBvh();

  CudaBvh(const CudaBvh&) = delete;
  CudaBvh& operator=(const CudaBvh&) = delete;
  CudaBvh(CudaBvh&& other) noexcept;
  CudaBvh& operator=(CudaBvh&& other) noexcept;

  CudaBvhQueryResult query(const std::vector<Vec3f>& points, int block_size = 128) const;

  // Production-oriented device-resident entry point. Inputs and outputs are CUDA
  // device pointers owned by the caller. No host transfer or allocation occurs
  // in this call. It is the entry point that later WoS kernels/wavefront stages
  // should use, rather than the host-vector probe wrapper above.
  void query_device(const DeviceVec3* d_points,
                    int query_count,
                    float* d_distance2,
                    DeviceVec3* d_closest,
                    int* d_triangle_id,
                    int* d_overflow,
                    int block_size = 128,
                    cudaStream_t stream = 0) const;

  std::size_t triangle_count() const { return triangle_count_; }
  std::size_t node_count() const { return node_count_; }
  std::size_t index_count() const { return index_count_; }
  int leaf_size() const { return leaf_size_; }
  int max_depth() const { return max_depth_; }

 private:
  void release();

  DeviceTriangle* d_triangles_ = nullptr;
  DeviceBvhNode* d_nodes_ = nullptr;
  int* d_triangle_indices_ = nullptr;

  std::size_t triangle_count_ = 0;
  std::size_t node_count_ = 0;
  std::size_t index_count_ = 0;
  int leaf_size_ = 8;
  int max_depth_ = 0;
};

std::string cuda_runtime_version_string();
std::string cuda_device_summary();

}  // namespace n2wos
