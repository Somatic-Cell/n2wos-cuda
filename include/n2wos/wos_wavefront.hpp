#pragma once

#include <cstdint>
#include <string>

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
  unsigned long long forced_max_steps = 0;
  unsigned long long overflow_count = 0;
};

std::string wavefront_method_name(WavefrontMethod method);
float harmonic_x2_minus_y2(Vec3f p);

WavefrontRunStats run_wavefront_harmonic(const CuBqlBvh& bvh,
                                          WavefrontMethod method,
                                          const WavefrontRunOptions& options);

}  // namespace n2wos
