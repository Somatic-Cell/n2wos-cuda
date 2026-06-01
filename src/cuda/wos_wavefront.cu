#include "n2wos/wos_wavefront.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace n2wos {
namespace {

#define N2WOS_CUDA_CHECK(expr) \
  do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                               ": " + cudaGetErrorString(err__)); \
    } \
  } while (false)

constexpr int kPureWos = 0;
constexpr int kOracleCoarse = 1;
constexpr int kOracleResidual = 2;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kReduceBlockSize = 256;

__host__ __device__ inline float harmonic_device(DeviceVec3 p) {
  return p.x * p.x - p.y * p.y;
}

__host__ __device__ inline DeviceVec3 make_device_vec3(float x, float y, float z) {
  DeviceVec3 p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

__device__ inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__device__ inline std::uint32_t pcg32(std::uint64_t* state) {
  const std::uint64_t oldstate = *state;
  *state = oldstate * 6364136223846793005ull + 1442695040888963407ull;
  const std::uint32_t xorshifted = static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
  const std::uint32_t rot = static_cast<std::uint32_t>(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ inline float uniform01(std::uint64_t* state) {
  return (static_cast<float>(pcg32(state)) + 0.5f) * 2.3283064365386963e-10f;
}

__device__ inline DeviceVec3 sample_unit_sphere(std::uint64_t* state) {
  const float u = uniform01(state);
  const float v = uniform01(state);
  const float z = 1.0f - 2.0f * u;
  const float phi = 2.0f * kPi * v;
  const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  return make_device_vec3(r * cosf(phi), r * sinf(phi), z);
}

__global__ void init_walks_kernel(DeviceVec3* positions,
                                  std::uint64_t* rng_states,
                                  std::uint8_t* active,
                                  int* prefix_done,
                                  int* step_count,
                                  int* forced_max_steps,
                                  float* cache_value,
                                  float* sample_value,
                                  int n,
                                  DeviceVec3 x0,
                                  std::uint64_t seed) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  positions[tid] = x0;
  const std::uint64_t s0 = splitmix64(seed ^ (static_cast<std::uint64_t>(tid) + 0x9e3779b97f4a7c15ull));
  rng_states[tid] = s0 ? s0 : 0x853c49e6748fea9bull;
  active[tid] = 1;
  prefix_done[tid] = 0;
  step_count[tid] = 0;
  forced_max_steps[tid] = 0;
  cache_value[tid] = 0.0f;
  sample_value[tid] = 0.0f;
}

__global__ void wos_wavefront_step_kernel(DeviceVec3* positions,
                                          std::uint64_t* rng_states,
                                          std::uint8_t* active,
                                          int* prefix_done,
                                          int* step_count,
                                          int* forced_max_steps,
                                          float* cache_value,
                                          float* sample_value,
                                          const float* distance2,
                                          const DeviceVec3* closest,
                                          const int* query_overflow,
                                          int n,
                                          int method,
                                          int depth_m,
                                          int max_steps,
                                          float epsilon,
                                          float step_scale) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n || !active[tid]) return;

  if (query_overflow && query_overflow[tid]) {
    sample_value[tid] = 0.0f;
    active[tid] = 0;
    forced_max_steps[tid] = 1;
    return;
  }

  const DeviceVec3 p = positions[tid];
  const float d2 = fmaxf(distance2[tid], 0.0f);
  const float d = sqrtf(d2);
  const bool at_boundary = d <= epsilon;
  const bool forced = step_count[tid] >= max_steps;

  if (at_boundary || forced || !isfinite(d)) {
    const float boundary_value = harmonic_device(closest[tid]);
    if (method == kOracleResidual) {
      sample_value[tid] = prefix_done[tid] ? boundary_value - cache_value[tid] : 0.0f;
    } else {
      sample_value[tid] = boundary_value;
    }
    forced_max_steps[tid] = forced ? 1 : 0;
    active[tid] = 0;
    return;
  }

  if (method == kOracleCoarse && step_count[tid] >= depth_m) {
    sample_value[tid] = harmonic_device(p);
    active[tid] = 0;
    return;
  }

  if (method == kOracleResidual && !prefix_done[tid] && step_count[tid] >= depth_m) {
    cache_value[tid] = harmonic_device(p);
    prefix_done[tid] = 1;
  }

  DeviceVec3 dir = sample_unit_sphere(&rng_states[tid]);
  const float radius = step_scale * d;
  positions[tid] = make_device_vec3(p.x + radius * dir.x,
                                    p.y + radius * dir.y,
                                    p.z + radius * dir.z);
  step_count[tid] += 1;
}

__global__ void reduce_wavefront_stats_kernel(const float* sample_value,
                                              const int* step_count,
                                              const int* forced_max_steps,
                                              const int* query_overflow,
                                              int n,
                                              double* block_sum,
                                              double* block_sum_sq,
                                              double* block_steps,
                                              unsigned long long* block_forced,
                                              unsigned long long* block_overflow) {
  __shared__ double s_sum[kReduceBlockSize];
  __shared__ double s_sum_sq[kReduceBlockSize];
  __shared__ double s_steps[kReduceBlockSize];
  __shared__ unsigned long long s_forced[kReduceBlockSize];
  __shared__ unsigned long long s_overflow[kReduceBlockSize];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;

  double v = 0.0;
  double steps = 0.0;
  unsigned long long forced = 0;
  unsigned long long overflow = 0;
  if (gid < n) {
    v = static_cast<double>(sample_value[gid]);
    steps = static_cast<double>(step_count[gid]);
    forced = forced_max_steps[gid] ? 1ull : 0ull;
    overflow = query_overflow[gid] ? 1ull : 0ull;
  }

  s_sum[tid] = v;
  s_sum_sq[tid] = v * v;
  s_steps[tid] = steps;
  s_forced[tid] = forced;
  s_overflow[tid] = overflow;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
      s_sum_sq[tid] += s_sum_sq[tid + offset];
      s_steps[tid] += s_steps[tid + offset];
      s_forced[tid] += s_forced[tid + offset];
      s_overflow[tid] += s_overflow[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sum[blockIdx.x] = s_sum[0];
    block_sum_sq[blockIdx.x] = s_sum_sq[0];
    block_steps[blockIdx.x] = s_steps[0];
    block_forced[blockIdx.x] = s_forced[0];
    block_overflow[blockIdx.x] = s_overflow[0];
  }
}

int method_to_device(WavefrontMethod method) {
  switch (method) {
    case WavefrontMethod::PureWos: return kPureWos;
    case WavefrontMethod::OracleCoarse: return kOracleCoarse;
    case WavefrontMethod::OracleResidual: return kOracleResidual;
  }
  throw std::runtime_error("unknown WavefrontMethod");
}

DeviceVec3 to_device(Vec3f p) {
  return DeviceVec3{p.x, p.y, p.z};
}

void validate_options(const WavefrontRunOptions& options) {
  if (options.samples <= 0) throw std::runtime_error("samples must be positive");
  if (options.max_steps <= 0) throw std::runtime_error("max_steps must be positive");
  if (options.depth_m < 0) throw std::runtime_error("depth_m must be non-negative");
  if (!(options.epsilon > 0.0f)) throw std::runtime_error("epsilon must be positive");
  if (!(options.step_scale > 0.0f && options.step_scale <= 1.0f)) {
    throw std::runtime_error("step_scale must be in (0, 1]");
  }
  if (options.block_size <= 0 || options.block_size > 1024) {
    throw std::runtime_error("block_size must be in [1, 1024]");
  }
}

}  // namespace

std::string wavefront_method_name(WavefrontMethod method) {
  switch (method) {
    case WavefrontMethod::PureWos: return "pure_wos";
    case WavefrontMethod::OracleCoarse: return "oracle_coarse";
    case WavefrontMethod::OracleResidual: return "oracle_residual";
  }
  return "unknown";
}

float harmonic_x2_minus_y2(Vec3f p) {
  return p.x * p.x - p.y * p.y;
}

WavefrontRunStats run_wavefront_harmonic(const CuBqlBvh& bvh,
                                          WavefrontMethod method,
                                          const WavefrontRunOptions& options) {
  validate_options(options);

  const int n = options.samples;
  const int block_size = options.block_size;
  const int grid_size = (n + block_size - 1) / block_size;
  const int reduce_grid = (n + kReduceBlockSize - 1) / kReduceBlockSize;

  DeviceVec3* d_positions = nullptr;
  std::uint64_t* d_rng_states = nullptr;
  std::uint8_t* d_active = nullptr;
  int* d_prefix_done = nullptr;
  int* d_step_count = nullptr;
  int* d_forced_max_steps = nullptr;
  float* d_cache_value = nullptr;
  float* d_sample_value = nullptr;
  float* d_distance2 = nullptr;
  DeviceVec3* d_closest = nullptr;
  int* d_triangle_id = nullptr;
  int* d_query_overflow = nullptr;
  double* d_block_sum = nullptr;
  double* d_block_sum_sq = nullptr;
  double* d_block_steps = nullptr;
  unsigned long long* d_block_forced = nullptr;
  unsigned long long* d_block_overflow = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  try {
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_positions), n * sizeof(DeviceVec3)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_states), n * sizeof(std::uint64_t)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_active), n * sizeof(std::uint8_t)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prefix_done), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_step_count), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_forced_max_steps), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cache_value), n * sizeof(float)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sample_value), n * sizeof(float)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distance2), n * sizeof(float)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_closest), n * sizeof(DeviceVec3)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_id), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_overflow), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_sum), reduce_grid * sizeof(double)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_sum_sq), reduce_grid * sizeof(double)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_steps), reduce_grid * sizeof(double)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_forced), reduce_grid * sizeof(unsigned long long)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_overflow), reduce_grid * sizeof(unsigned long long)));

    init_walks_kernel<<<grid_size, block_size>>>(d_positions,
                                                 d_rng_states,
                                                 d_active,
                                                 d_prefix_done,
                                                 d_step_count,
                                                 d_forced_max_steps,
                                                 d_cache_value,
                                                 d_sample_value,
                                                 n,
                                                 to_device(options.x0),
                                                 options.seed);
    N2WOS_CUDA_CHECK(cudaGetLastError());

    N2WOS_CUDA_CHECK(cudaEventCreate(&start));
    N2WOS_CUDA_CHECK(cudaEventCreate(&stop));
    N2WOS_CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter <= options.max_steps; ++iter) {
      bvh.query_device_masked(d_positions,
                              d_active,
                              n,
                              d_distance2,
                              d_closest,
                              d_triangle_id,
                              d_query_overflow,
                              block_size,
                              0);
      wos_wavefront_step_kernel<<<grid_size, block_size>>>(d_positions,
                                                           d_rng_states,
                                                           d_active,
                                                           d_prefix_done,
                                                           d_step_count,
                                                           d_forced_max_steps,
                                                           d_cache_value,
                                                           d_sample_value,
                                                           d_distance2,
                                                           d_closest,
                                                           d_query_overflow,
                                                           n,
                                                           method_to_device(method),
                                                           options.depth_m,
                                                           options.max_steps,
                                                           options.epsilon,
                                                           options.step_scale);
      N2WOS_CUDA_CHECK(cudaGetLastError());
    }

    reduce_wavefront_stats_kernel<<<reduce_grid, kReduceBlockSize>>>(d_sample_value,
                                                                     d_step_count,
                                                                     d_forced_max_steps,
                                                                     d_query_overflow,
                                                                     n,
                                                                     d_block_sum,
                                                                     d_block_sum_sq,
                                                                     d_block_steps,
                                                                     d_block_forced,
                                                                     d_block_overflow);
    N2WOS_CUDA_CHECK(cudaGetLastError());
    N2WOS_CUDA_CHECK(cudaEventRecord(stop));
    N2WOS_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    N2WOS_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<double> h_sum(reduce_grid);
    std::vector<double> h_sum_sq(reduce_grid);
    std::vector<double> h_steps(reduce_grid);
    std::vector<unsigned long long> h_forced(reduce_grid);
    std::vector<unsigned long long> h_overflow(reduce_grid);
    N2WOS_CUDA_CHECK(cudaMemcpy(h_sum.data(), d_block_sum, reduce_grid * sizeof(double), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(h_sum_sq.data(), d_block_sum_sq, reduce_grid * sizeof(double), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(h_steps.data(), d_block_steps, reduce_grid * sizeof(double), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(h_forced.data(), d_block_forced, reduce_grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(h_overflow.data(), d_block_overflow, reduce_grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    double sum_sq = 0.0;
    double steps = 0.0;
    unsigned long long forced = 0;
    unsigned long long overflow = 0;
    for (int i = 0; i < reduce_grid; ++i) {
      sum += h_sum[i];
      sum_sq += h_sum_sq[i];
      steps += h_steps[i];
      forced += h_forced[i];
      overflow += h_overflow[i];
    }

    WavefrontRunStats stats;
    stats.samples = n;
    stats.mean = sum / static_cast<double>(n);
    const double centered = sum_sq - static_cast<double>(n) * stats.mean * stats.mean;
    stats.sample_variance = n > 1 ? std::max(0.0, centered) / static_cast<double>(n - 1) : 0.0;
    stats.estimator_variance = stats.sample_variance / static_cast<double>(n);
    stats.stderr = std::sqrt(stats.estimator_variance);
    stats.mean_steps = steps / static_cast<double>(n);
    stats.elapsed_ms = elapsed_ms;
    stats.us_per_sample = 1000.0 * static_cast<double>(elapsed_ms) / static_cast<double>(n);
    stats.launched_query_slots = static_cast<double>(n) * static_cast<double>(options.max_steps + 1);
    stats.mean_launched_query_slots_per_sample = static_cast<double>(options.max_steps + 1);
    stats.forced_max_steps = forced;
    stats.overflow_count = overflow;

    N2WOS_CUDA_CHECK(cudaEventDestroy(start));
    start = nullptr;
    N2WOS_CUDA_CHECK(cudaEventDestroy(stop));
    stop = nullptr;

    cudaFree(d_positions);
    cudaFree(d_rng_states);
    cudaFree(d_active);
    cudaFree(d_prefix_done);
    cudaFree(d_step_count);
    cudaFree(d_forced_max_steps);
    cudaFree(d_cache_value);
    cudaFree(d_sample_value);
    cudaFree(d_distance2);
    cudaFree(d_closest);
    cudaFree(d_triangle_id);
    cudaFree(d_query_overflow);
    cudaFree(d_block_sum);
    cudaFree(d_block_sum_sq);
    cudaFree(d_block_steps);
    cudaFree(d_block_forced);
    cudaFree(d_block_overflow);
    return stats;
  } catch (...) {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (d_positions) cudaFree(d_positions);
    if (d_rng_states) cudaFree(d_rng_states);
    if (d_active) cudaFree(d_active);
    if (d_prefix_done) cudaFree(d_prefix_done);
    if (d_step_count) cudaFree(d_step_count);
    if (d_forced_max_steps) cudaFree(d_forced_max_steps);
    if (d_cache_value) cudaFree(d_cache_value);
    if (d_sample_value) cudaFree(d_sample_value);
    if (d_distance2) cudaFree(d_distance2);
    if (d_closest) cudaFree(d_closest);
    if (d_triangle_id) cudaFree(d_triangle_id);
    if (d_query_overflow) cudaFree(d_query_overflow);
    if (d_block_sum) cudaFree(d_block_sum);
    if (d_block_sum_sq) cudaFree(d_block_sum_sq);
    if (d_block_steps) cudaFree(d_block_steps);
    if (d_block_forced) cudaFree(d_block_forced);
    if (d_block_overflow) cudaFree(d_block_overflow);
    throw;
  }
}

}  // namespace n2wos
