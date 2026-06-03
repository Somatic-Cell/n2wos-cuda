#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "n2wos/cuda_bvh.hpp"
#include "n2wos/types.hpp"

namespace n2wos {

class CuBqlBvh;

enum class NcBoundaryMode {
  HarmonicX2MinusY2 = 0,
  ExternalChargesMedium = 1,
  ExternalChargesHigh = 2,
  HarmonicZebraK4 = 3,
  HarmonicZebraK8 = 4,
  HarmonicZebraK12 = 5,
  ExternalChargesShellK8 = 6,
  ExternalChargesShellK16 = 7,
  BoundaryTextureStripesK8 = 8,
  BoundaryTextureStripesK16 = 9,
  BoundaryTextureCheckerK8 = 10,
  BoundaryTextureCheckerK16 = 11,
};

enum class NcLabelSource {
  ExactAnalytic = 0,
  WosSupervision = 1,
};

struct NcDeviceDatasetOptions {
  const DeviceVec3* d_world_points = nullptr;
  int point_count = 0;
  int walks_per_point = 16;
  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  std::uint64_t seed = 12345;
  int refresh_index = 0;
  NcBoundaryMode boundary_mode = NcBoundaryMode::ExternalChargesHigh;
  NcLabelSource label_source = NcLabelSource::WosSupervision;
  Vec3f input_min = {-1.0f, -1.0f, -1.0f};
  Vec3f input_extent = {2.0f, 2.0f, 2.0f};
  int block_size = 128;
};

struct NcDeviceSampleOptions {
  const DeviceVec3* d_eval_points = nullptr;
  int eval_point_count = 0;
  int walks_per_point = 1;
  int depth_m = 1;
  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  std::uint64_t seed = 12345;
  NcBoundaryMode boundary_mode = NcBoundaryMode::ExternalChargesHigh;
  Vec3f input_min = {-1.0f, -1.0f, -1.0f};
  Vec3f input_extent = {2.0f, 2.0f, 2.0f};
  int block_size = 128;
};

const char* nc_boundary_mode_name(NcBoundaryMode mode);
NcBoundaryMode parse_nc_boundary_mode(const char* text);
const char* nc_label_source_name(NcLabelSource source);
NcLabelSource parse_nc_label_source(const char* text);
float nc_boundary_value_host(Vec3f p, NcBoundaryMode mode);

void launch_nc_fill_inputs(const NcDeviceDatasetOptions& options,
                           float* d_inputs,
                           cudaStream_t stream = 0);

void launch_nc_update_labels(const CuBqlBvh& bvh,
                             const NcDeviceDatasetOptions& options,
                             float* d_inputs,
                             float* d_targets,
                             int* d_counts,
                             cudaStream_t stream = 0);

void launch_nc_pure_wos_samples(const CuBqlBvh& bvh,
                                const NcDeviceSampleOptions& options,
                                float* d_sample_values,
                                int* d_step_count,
                                int* d_forced_max_steps,
                                int* d_query_overflow,
                                cudaStream_t stream = 0);

void launch_nc_hybrid_prefix(const CuBqlBvh& bvh,
                             const NcDeviceSampleOptions& options,
                             float* d_prefix_inputs,
                             float* d_boundary_values,
                             std::uint8_t* d_needs_cache,
                             int* d_step_count,
                             int* d_forced_max_steps,
                             int* d_query_overflow,
                             cudaStream_t stream = 0);

// Prefix to depth_m, then continue from X_m to the boundary for the residual
// term W(X_m)-C_theta(X_m). If the path reaches the boundary before depth_m,
// d_needs_cache is 0, d_boundary_values and d_continuation_values are equal,
// and the residual becomes zero after combination.
void launch_nc_2lmc_prefix_continue(const CuBqlBvh& bvh,
                                    const NcDeviceSampleOptions& options,
                                    float* d_prefix_inputs,
                                    float* d_boundary_values,
                                    float* d_continuation_values,
                                    std::uint8_t* d_needs_cache,
                                    int* d_step_count,
                                    int* d_forced_max_steps,
                                    int* d_query_overflow,
                                    cudaStream_t stream = 0);

// Device-side consumer for tiny-cuda-nn outputs. It produces NC-only sample
// values and residual sample values without routing TCNN outputs through host
// memory.
void launch_nc_combine_cache_and_residual(const float* d_cache_outputs,
                                          const float* d_boundary_values,
                                          const float* d_continuation_values,
                                          const std::uint8_t* d_needs_cache,
                                          int sample_count,
                                          float* d_nc_values,
                                          float* d_residual_values,
                                          int block_size = 128,
                                          cudaStream_t stream = 0);

}  // namespace n2wos
