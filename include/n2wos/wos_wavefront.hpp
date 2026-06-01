#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "n2wos/cubql_bvh.hpp"
#include "n2wos/types.hpp"

namespace n2wos {

enum class WavefrontMethod {
  PureWos,
  OracleCoarse,
  OracleResidual,
};

struct WavefrontRunOptions {
  int samples = 65536;
  Vec3f x0 = {0.1f, 0.05f, 0.0f};
  int max_steps = 256;
  int depth_m = 8;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  std::uint64_t seed = 12345;
  int block_size = 128;
};

struct WavefrontRunStats {
  int samples = 0;
  double mean = 0.0;
  double sample_variance = 0.0;
  double estimator_variance = 0.0;
  double stderr = 0.0;
  double mean_steps = 0.0;
  double elapsed_ms = 0.0;
  double us_per_sample = 0.0;
  double launched_query_slots = 0.0;
  double mean_launched_query_slots_per_sample = 0.0;
  int scheduled_query_rounds = 0;
  unsigned long long forced_max_steps = 0;
  unsigned long long overflow_count = 0;
};

struct SliceRenderOptions {
  int width = 128;
  int height = 128;
  int samples_per_pixel = 256;
  std::string slice_view = "xy";
  // For slice_view=xy, this is z.  For xz, this is y.  For yz, this is x.
  float plane_z = 0.0f;
  bool use_mesh_bounds = true;
  bool preserve_world_aspect = true;
  float x_min = -1.0f;
  float x_max = 1.0f;
  float y_min = -1.0f;
  float y_max = 1.0f;
  float bounds_padding_fraction = 0.02f;
  bool mask_outside_mesh = true;
  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  std::uint64_t seed = 12345;
  int block_size = 128;
};

struct SlicePixelStats {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  std::uint8_t inside = 0;
  int samples = 0;
  double mean = 0.0;
  double exact = 0.0;
  double error = 0.0;
  double sample_variance = 0.0;
  double estimator_variance = 0.0;
  double stderr = 0.0;
  double mean_steps = 0.0;
  unsigned long long forced_max_steps = 0;
  unsigned long long overflow_count = 0;
};

struct SliceRenderResult {
  int width = 0;
  int height = 0;
  int samples_per_pixel = 0;
  int inside_pixels = 0;
  double elapsed_ms = 0.0;
  double us_per_active_pixel = 0.0;
  double us_per_launched_sample = 0.0;
  double rmse_inside = 0.0;
  double mae_inside = 0.0;
  double max_abs_error_inside = 0.0;
  unsigned long long forced_max_steps = 0;
  unsigned long long overflow_count = 0;
  float frame_u_min = 0.0f;
  float frame_u_max = 0.0f;
  float frame_v_min = 0.0f;
  float frame_v_max = 0.0f;
  double world_units_per_pixel_u = 0.0;
  double world_units_per_pixel_v = 0.0;
  std::string slice_view = "xy";
  std::vector<SlicePixelStats> pixels;
};

std::string wavefront_method_name(WavefrontMethod method);
float harmonic_x2_minus_y2(Vec3f p);

WavefrontRunStats run_wavefront_harmonic(const CuBqlBvh& bvh,
                                          WavefrontMethod method,
                                          const WavefrontRunOptions& options);

// Persistent per-sample kernel variant.  Each CUDA thread owns one walk and
// performs its own cuBQL closest-point traversal loop on device, avoiding the
// host-controlled global step loop used by run_wavefront_harmonic.  This is a
// more production-like path for pure WoS and oracle diagnostics; TCNN cache
// integration will still use batched inference between prefix and residual
// stages.
WavefrontRunStats run_persistent_harmonic(const CuBqlBvh& bvh,
                                          WavefrontMethod method,
                                          const WavefrontRunOptions& options);

SliceRenderResult render_persistent_harmonic_slice(const CuBqlBvh& bvh,
                                                   const Mesh& mesh,
                                                   const SliceRenderOptions& options);

}  // namespace n2wos
