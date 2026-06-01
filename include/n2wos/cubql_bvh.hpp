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

enum class WavefrontMethod;
struct WavefrontRunOptions;
struct WavefrontRunStats;
struct SliceRenderOptions;
struct SliceRenderResult;
struct NcDeviceDatasetOptions;
struct NcDeviceSampleOptions;

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
  friend WavefrontRunStats run_persistent_harmonic(const CuBqlBvh& bvh,
                                                   WavefrontMethod method,
                                                   const WavefrontRunOptions& options);
  friend SliceRenderResult render_persistent_harmonic_slice(const CuBqlBvh& bvh,
                                                            const Mesh& mesh,
                                                            const SliceRenderOptions& options);
  friend void launch_nc_update_labels(const CuBqlBvh& bvh,
                                      const NcDeviceDatasetOptions& options,
                                      float* d_inputs,
                                      float* d_targets,
                                      int* d_counts,
                                      cudaStream_t stream);
  friend void launch_nc_pure_wos_samples(const CuBqlBvh& bvh,
                                         const NcDeviceSampleOptions& options,
                                         float* d_sample_values,
                                         int* d_step_count,
                                         int* d_forced_max_steps,
                                         int* d_query_overflow,
                                         cudaStream_t stream);
  friend void launch_nc_hybrid_prefix(const CuBqlBvh& bvh,
                                      const NcDeviceSampleOptions& options,
                                      float* d_prefix_inputs,
                                      float* d_boundary_values,
                                      std::uint8_t* d_needs_cache,
                                      int* d_step_count,
                                      int* d_forced_max_steps,
                                      int* d_query_overflow,
                                      cudaStream_t stream);
  friend void launch_nc_2lmc_prefix_continue(const CuBqlBvh& bvh,
                                             const NcDeviceSampleOptions& options,
                                             float* d_prefix_inputs,
                                             float* d_boundary_values,
                                             float* d_continuation_values,
                                             std::uint8_t* d_needs_cache,
                                             int* d_step_count,
                                             int* d_forced_max_steps,
                                             int* d_query_overflow,
                                             cudaStream_t stream);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace n2wos
