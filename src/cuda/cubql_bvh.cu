#include "n2wos/cubql_bvh.hpp"
#include "n2wos/wos_wavefront.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// cuBQL header-only builder/query implementation points.
// These are the macros used by cuBQL's triangle-mesh distance sample and
// expose the concrete CUDA builders selected below.
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_TRIANGLE_CPAT_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include <cuBQL/queries/triangleData/math/pointToTriangleDistance.h>
#include <cuBQL/queries/triangleData/closestPointOnAnyTriangle.h>

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

cuBQL::vec3f to_cubql_vec3(const Vec3f& p) {
  return cuBQL::vec3f(p.x, p.y, p.z);
}

__host__ __device__ inline cuBQL::vec3f to_cubql_vec3(DeviceVec3 p) {
  return cuBQL::vec3f(p.x, p.y, p.z);
}

__host__ __device__ inline DeviceVec3 from_cubql_vec3(cuBQL::vec3f p) {
  return DeviceVec3{p.x, p.y, p.z};
}

DeviceVec3 to_device_vec3(const Vec3f& p) {
  return DeviceVec3{p.x, p.y, p.z};
}

Vec3f from_device_vec3(const DeviceVec3& p) {
  return Vec3f{p.x, p.y, p.z};
}

std::string canonical_build_method(std::string build_method) {
  if (build_method == "sm" || build_method == "spatial-median") return "spatial_median";
  if (build_method == "surface_area_heuristic") return "sah";
  if (build_method == "morton" || build_method == "radix_morton") return "radix";
  if (build_method == "rebin" || build_method == "robust_radix" || build_method == "modified_radix" ||
      build_method == "rebin_radix" || build_method == "rebin-radix") {
    throw std::runtime_error("cuBQL build method rebin_radix is disabled: current cuBQL main declares rebinRadixBuilder but does not provide a linkable implementation header");
  }
  return build_method;
}

std::vector<cuBQL::Triangle> make_cubql_triangles(const Mesh& mesh) {
  require_mesh_valid(mesh);
  std::vector<cuBQL::Triangle> triangles;
  triangles.reserve(mesh.triangles.size());
  for (const Triangle& tri : mesh.triangles) {
    triangles.emplace_back(to_cubql_vec3(mesh.vertices[tri.v0]),
                           to_cubql_vec3(mesh.vertices[tri.v1]),
                           to_cubql_vec3(mesh.vertices[tri.v2]));
  }
  return triangles;
}

cuBQL::BuildConfig make_build_config(int leaf_size, const std::string& build_method) {
  if (leaf_size <= 0) {
    throw std::runtime_error("cuBQL leaf size must be positive");
  }
  cuBQL::BuildConfig config(leaf_size);
  config.maxAllowedLeafSize = leaf_size;
  if (build_method == "spatial_median" || build_method == "radix") {
    return config;
  }
  if (build_method == "sah") {
    config.enableSAH();
    return config;
  }
  if (build_method == "elh") {
    config.enableELH();
    return config;
  }
  throw std::runtime_error("unknown cuBQL build method: " + build_method +
                           " (expected spatial_median, sah, elh, or radix; rebin_radix disabled)");
}

void build_cubql_bvh(cuBQL::bvh3f& bvh,
                     const cuBQL::box3f* d_boxes,
                     int primitive_count,
                     const cuBQL::BuildConfig& config,
                     const std::string& build_method) {
  if (build_method == "spatial_median" || build_method == "elh") {
    // gpuBuilder dispatches by BuildConfig::buildMethod. In the default case
    // this is the adaptive spatial median builder; with enableELH() it uses
    // cuBQL's experimental edge-length heuristic builder.
    cuBQL::gpuBuilder(bvh, d_boxes, primitive_count, config);
  } else if (build_method == "sah") {
    // Direct call keeps the JSON label unambiguous and avoids relying on an
    // internal dispatch path.
    cuBQL::cuda::sahBuilder(bvh, d_boxes, primitive_count, config);
  } else if (build_method == "radix") {
    cuBQL::cuda::radixBuilder(bvh, d_boxes, primitive_count, config);
  } else {
    throw std::runtime_error("unknown cuBQL build method: " + build_method);
  }
}

__global__ void generate_cubql_boxes_kernel(cuBQL::box3f* box_for_builder,
                                            const cuBQL::Triangle* triangles,
                                            int triangle_count) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= triangle_count) return;
  box_for_builder[tid] = triangles[tid].bounds();
}

__global__ void cubql_closest_point_kernel(const DeviceVec3* __restrict__ query_points,
                                           const std::uint8_t* __restrict__ active_mask,
                                           int query_count,
                                           const cuBQL::Triangle* __restrict__ triangles,
                                           cuBQL::bvh3f bvh,
                                           float* __restrict__ out_distance2,
                                           DeviceVec3* __restrict__ out_closest,
                                           int* __restrict__ out_triangle_id,
                                           int* __restrict__ out_overflow) {
  const int qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= query_count) return;
  if (active_mask && active_mask[qid] == 0) {
    out_distance2[qid] = 0.0f;
    out_closest[qid] = query_points[qid];
    out_triangle_id[qid] = -1;
    out_overflow[qid] = 0;
    return;
  }

  const cuBQL::vec3f query_point = to_cubql_vec3(query_points[qid]);
  cuBQL::triangles::CPAT cpat;
  cpat.runQuery(triangles, bvh, query_point);

  out_distance2[qid] = cpat.sqrDist;
  out_closest[qid] = from_cubql_vec3(cpat.P);
  out_triangle_id[qid] = cpat.triangleIdx;
  out_overflow[qid] = 0;
}


constexpr int kPersistentPureWos = 0;
constexpr int kPersistentOracleCoarse = 1;
constexpr int kPersistentOracleResidual = 2;
constexpr float kPersistentPi = 3.14159265358979323846f;
constexpr int kPersistentReduceBlockSize = 256;

__host__ __device__ inline float persistent_harmonic(DeviceVec3 p) {
  return p.x * p.x - p.y * p.y;
}

__host__ __device__ inline DeviceVec3 persistent_make_vec3(float x, float y, float z) {
  DeviceVec3 p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

__device__ inline std::uint64_t persistent_splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__device__ inline std::uint32_t persistent_pcg32(std::uint64_t* state) {
  const std::uint64_t oldstate = *state;
  *state = oldstate * 6364136223846793005ull + 1442695040888963407ull;
  const std::uint32_t xorshifted = static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
  const std::uint32_t rot = static_cast<std::uint32_t>(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ inline float persistent_uniform01(std::uint64_t* state) {
  return (static_cast<float>(persistent_pcg32(state)) + 0.5f) * 2.3283064365386963e-10f;
}

__device__ inline DeviceVec3 persistent_sample_unit_sphere(std::uint64_t* state) {
  const float u = persistent_uniform01(state);
  const float v = persistent_uniform01(state);
  const float z = 1.0f - 2.0f * u;
  const float phi = 2.0f * kPersistentPi * v;
  const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  return persistent_make_vec3(r * cosf(phi), r * sinf(phi), z);
}

__global__ void persistent_harmonic_wos_kernel(int n,
                                               DeviceVec3 x0,
                                               std::uint64_t seed,
                                               int method,
                                               int depth_m,
                                               int max_steps,
                                               float epsilon,
                                               float step_scale,
                                               const cuBQL::Triangle* __restrict__ triangles,
                                               cuBQL::bvh3f bvh,
                                               float* __restrict__ sample_value,
                                               int* __restrict__ step_count,
                                               int* __restrict__ forced_max_steps,
                                               int* __restrict__ query_overflow) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  std::uint64_t rng = persistent_splitmix64(seed ^ (static_cast<std::uint64_t>(tid) + 0x9e3779b97f4a7c15ull));
  if (!rng) rng = 0x853c49e6748fea9bull;

  DeviceVec3 p = x0;
  bool prefix_done = false;
  float cache_value = 0.0f;

  query_overflow[tid] = 0;
  forced_max_steps[tid] = 0;
  step_count[tid] = 0;
  sample_value[tid] = 0.0f;

  for (int step = 0; step <= max_steps; ++step) {
    cuBQL::triangles::CPAT cpat;
    cpat.runQuery(triangles, bvh, to_cubql_vec3(p));

    const float d2 = fmaxf(cpat.sqrDist, 0.0f);
    const float d = sqrtf(d2);
    const bool at_boundary = d <= epsilon;
    const bool forced = step >= max_steps;

    if (at_boundary || forced || !isfinite(d)) {
      const float boundary_value = persistent_harmonic(from_cubql_vec3(cpat.P));
      if (method == kPersistentOracleResidual) {
        sample_value[tid] = prefix_done ? boundary_value - cache_value : 0.0f;
      } else {
        sample_value[tid] = boundary_value;
      }
      step_count[tid] = step;
      forced_max_steps[tid] = forced ? 1 : 0;
      return;
    }

    if (method == kPersistentOracleCoarse && step >= depth_m) {
      sample_value[tid] = persistent_harmonic(p);
      step_count[tid] = step;
      return;
    }

    if (method == kPersistentOracleResidual && !prefix_done && step >= depth_m) {
      cache_value = persistent_harmonic(p);
      prefix_done = true;
    }

    const DeviceVec3 dir = persistent_sample_unit_sphere(&rng);
    const float radius = step_scale * d;
    p = persistent_make_vec3(p.x + radius * dir.x,
                             p.y + radius * dir.y,
                             p.z + radius * dir.z);
  }

  // The loop should always return at step == max_steps. Keep a defensive
  // fallback so uninitialized output cannot leak into reductions.
  sample_value[tid] = 0.0f;
  step_count[tid] = max_steps;
  forced_max_steps[tid] = 1;
}

__global__ void persistent_reduce_stats_kernel(const float* sample_value,
                                               const int* step_count,
                                               const int* forced_max_steps,
                                               const int* query_overflow,
                                               int n,
                                               double* block_sum,
                                               double* block_sum_sq,
                                               double* block_steps,
                                               unsigned long long* block_forced,
                                               unsigned long long* block_overflow) {
  __shared__ double s_sum[kPersistentReduceBlockSize];
  __shared__ double s_sum_sq[kPersistentReduceBlockSize];
  __shared__ double s_steps[kPersistentReduceBlockSize];
  __shared__ unsigned long long s_forced[kPersistentReduceBlockSize];
  __shared__ unsigned long long s_overflow[kPersistentReduceBlockSize];

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

int persistent_method_to_device(WavefrontMethod method) {
  switch (method) {
    case WavefrontMethod::PureWos: return kPersistentPureWos;
    case WavefrontMethod::OracleCoarse: return kPersistentOracleCoarse;
    case WavefrontMethod::OracleResidual: return kPersistentOracleResidual;
  }
  throw std::runtime_error("unknown WavefrontMethod");
}

DeviceVec3 persistent_to_device(Vec3f p) {
  return DeviceVec3{p.x, p.y, p.z};
}

void validate_persistent_options(const WavefrontRunOptions& options) {
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

struct CuBqlBvh::Impl {
  cuBQL::Triangle* d_triangles = nullptr;
  cuBQL::bvh3f bvh{};
  std::size_t triangle_count = 0;
  int leaf_size = 8;
  float build_milliseconds = 0.0f;
  std::string build_method = "spatial_median";

  ~Impl() { release(); }

  void release() {
    if (d_triangles) {
      cudaFree(d_triangles);
      d_triangles = nullptr;
    }
    if (bvh.nodes || bvh.primIDs) {
      cuBQL::cuda::free(bvh);
      bvh = cuBQL::bvh3f{};
    }
    triangle_count = 0;
  }
};

CuBqlBvh::CuBqlBvh() = default;

CuBqlBvh::CuBqlBvh(const Mesh& mesh, int leaf_size, const std::string& build_method)
    : impl_(std::make_unique<Impl>()) {
  const std::vector<cuBQL::Triangle> h_triangles = make_cubql_triangles(mesh);
  impl_->triangle_count = h_triangles.size();
  impl_->leaf_size = leaf_size;
  impl_->build_method = canonical_build_method(build_method);

  N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl_->d_triangles),
                              h_triangles.size() * sizeof(cuBQL::Triangle)));
  N2WOS_CUDA_CHECK(cudaMemcpy(impl_->d_triangles,
                              h_triangles.data(),
                              h_triangles.size() * sizeof(cuBQL::Triangle),
                              cudaMemcpyHostToDevice));

  cuBQL::box3f* d_boxes = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  try {
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_boxes), h_triangles.size() * sizeof(cuBQL::box3f)));
    N2WOS_CUDA_CHECK(cudaEventCreate(&start));
    N2WOS_CUDA_CHECK(cudaEventCreate(&stop));

    const int n = static_cast<int>(h_triangles.size());
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    N2WOS_CUDA_CHECK(cudaEventRecord(start));
    generate_cubql_boxes_kernel<<<grid_size, block_size>>>(d_boxes, impl_->d_triangles, n);
    N2WOS_CUDA_CHECK(cudaGetLastError());

    cuBQL::BuildConfig config = make_build_config(leaf_size, impl_->build_method);
    build_cubql_bvh(impl_->bvh, d_boxes, n, config, impl_->build_method);

    N2WOS_CUDA_CHECK(cudaEventRecord(stop));
    N2WOS_CUDA_CHECK(cudaEventSynchronize(stop));
    N2WOS_CUDA_CHECK(cudaEventElapsedTime(&impl_->build_milliseconds, start, stop));

    N2WOS_CUDA_CHECK(cudaEventDestroy(start));
    start = nullptr;
    N2WOS_CUDA_CHECK(cudaEventDestroy(stop));
    stop = nullptr;
    N2WOS_CUDA_CHECK(cudaFree(d_boxes));
    d_boxes = nullptr;
  } catch (...) {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (d_boxes) cudaFree(d_boxes);
    impl_->release();
    throw;
  }
}

CuBqlBvh::~CuBqlBvh() = default;

CuBqlBvh::CuBqlBvh(CuBqlBvh&& other) noexcept = default;
CuBqlBvh& CuBqlBvh::operator=(CuBqlBvh&& other) noexcept = default;

void CuBqlBvh::query_device(const DeviceVec3* d_points,
                            int query_count,
                            float* d_distance2,
                            DeviceVec3* d_closest,
                            int* d_triangle_id,
                            int* d_overflow,
                            int block_size,
                            cudaStream_t stream) const {
  if (!impl_ || !impl_->d_triangles || !impl_->bvh.nodes || !impl_->bvh.primIDs) {
    throw std::runtime_error("CuBqlBvh::query_device called on empty BVH");
  }
  if (query_count < 0) {
    throw std::runtime_error("CuBqlBvh::query_device query_count must be non-negative");
  }
  if (query_count == 0) return;
  if (!d_points || !d_distance2 || !d_closest || !d_triangle_id || !d_overflow) {
    throw std::runtime_error("CuBqlBvh::query_device received a null device pointer");
  }
  if (block_size <= 0 || block_size > 1024) {
    throw std::runtime_error("CUDA block size must be in [1, 1024]");
  }

  const int grid_size = (query_count + block_size - 1) / block_size;
  cubql_closest_point_kernel<<<grid_size, block_size, 0, stream>>>(d_points,
                                                                  nullptr,
                                                                  query_count,
                                                                  impl_->d_triangles,
                                                                  impl_->bvh,
                                                                  d_distance2,
                                                                  d_closest,
                                                                  d_triangle_id,
                                                                  d_overflow);
  N2WOS_CUDA_CHECK(cudaGetLastError());
}

void CuBqlBvh::query_device_masked(const DeviceVec3* d_points,
                                   const std::uint8_t* d_active,
                                   int query_count,
                                   float* d_distance2,
                                   DeviceVec3* d_closest,
                                   int* d_triangle_id,
                                   int* d_overflow,
                                   int block_size,
                                   cudaStream_t stream) const {
  if (!impl_ || !impl_->d_triangles || !impl_->bvh.nodes || !impl_->bvh.primIDs) {
    throw std::runtime_error("CuBqlBvh::query_device_masked called on empty BVH");
  }
  if (query_count < 0) {
    throw std::runtime_error("CuBqlBvh::query_device_masked query_count must be non-negative");
  }
  if (query_count == 0) return;
  if (!d_points || !d_active || !d_distance2 || !d_closest || !d_triangle_id || !d_overflow) {
    throw std::runtime_error("CuBqlBvh::query_device_masked received a null device pointer");
  }
  if (block_size <= 0 || block_size > 1024) {
    throw std::runtime_error("CUDA block size must be in [1, 1024]");
  }

  const int grid_size = (query_count + block_size - 1) / block_size;
  cubql_closest_point_kernel<<<grid_size, block_size, 0, stream>>>(d_points,
                                                                  d_active,
                                                                  query_count,
                                                                  impl_->d_triangles,
                                                                  impl_->bvh,
                                                                  d_distance2,
                                                                  d_closest,
                                                                  d_triangle_id,
                                                                  d_overflow);
  N2WOS_CUDA_CHECK(cudaGetLastError());
}

CudaBvhQueryResult CuBqlBvh::query(const std::vector<Vec3f>& points, int block_size) const {
  if (points.empty()) {
    return {};
  }

  std::vector<DeviceVec3> h_points(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    h_points[i] = to_device_vec3(points[i]);
  }

  DeviceVec3* d_points = nullptr;
  float* d_distance2 = nullptr;
  DeviceVec3* d_closest = nullptr;
  int* d_triangle_id = nullptr;
  int* d_overflow = nullptr;

  try {
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points), h_points.size() * sizeof(DeviceVec3)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distance2), h_points.size() * sizeof(float)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_closest), h_points.size() * sizeof(DeviceVec3)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_id), h_points.size() * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_overflow), h_points.size() * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), h_points.size() * sizeof(DeviceVec3), cudaMemcpyHostToDevice));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    N2WOS_CUDA_CHECK(cudaEventCreate(&start));
    N2WOS_CUDA_CHECK(cudaEventCreate(&stop));
    N2WOS_CUDA_CHECK(cudaEventRecord(start));
    query_device(d_points,
                 static_cast<int>(points.size()),
                 d_distance2,
                 d_closest,
                 d_triangle_id,
                 d_overflow,
                 block_size,
                 0);
    N2WOS_CUDA_CHECK(cudaEventRecord(stop));
    N2WOS_CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    N2WOS_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    N2WOS_CUDA_CHECK(cudaEventDestroy(start));
    N2WOS_CUDA_CHECK(cudaEventDestroy(stop));

    CudaBvhQueryResult result;
    result.distance2.resize(points.size());
    result.closest.resize(points.size());
    result.triangle_id.resize(points.size());
    std::vector<DeviceVec3> h_closest(points.size());
    std::vector<int> h_overflow(points.size());

    N2WOS_CUDA_CHECK(cudaMemcpy(result.distance2.data(), d_distance2, points.size() * sizeof(float), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(h_closest.data(), d_closest, points.size() * sizeof(DeviceVec3), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(result.triangle_id.data(), d_triangle_id, points.size() * sizeof(int), cudaMemcpyDeviceToHost));
    N2WOS_CUDA_CHECK(cudaMemcpy(h_overflow.data(), d_overflow, points.size() * sizeof(int), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < points.size(); ++i) {
      result.closest[i] = from_device_vec3(h_closest[i]);
      result.overflow_count += h_overflow[i] != 0 ? 1 : 0;
    }
    result.kernel_milliseconds = milliseconds;

    cudaFree(d_points);
    cudaFree(d_distance2);
    cudaFree(d_closest);
    cudaFree(d_triangle_id);
    cudaFree(d_overflow);
    return result;
  } catch (...) {
    if (d_points) cudaFree(d_points);
    if (d_distance2) cudaFree(d_distance2);
    if (d_closest) cudaFree(d_closest);
    if (d_triangle_id) cudaFree(d_triangle_id);
    if (d_overflow) cudaFree(d_overflow);
    throw;
  }
}

std::size_t CuBqlBvh::triangle_count() const {
  return impl_ ? impl_->triangle_count : 0;
}

std::size_t CuBqlBvh::node_count() const {
  return impl_ ? impl_->bvh.numNodes : 0;
}

std::size_t CuBqlBvh::prim_id_count() const {
  return impl_ ? impl_->bvh.numPrims : 0;
}

int CuBqlBvh::leaf_size() const {
  return impl_ ? impl_->leaf_size : 0;
}

float CuBqlBvh::build_milliseconds() const {
  return impl_ ? impl_->build_milliseconds : 0.0f;
}

std::string CuBqlBvh::build_method() const {
  return impl_ ? impl_->build_method : std::string();
}


WavefrontRunStats run_persistent_harmonic(const CuBqlBvh& bvh,
                                          WavefrontMethod method,
                                          const WavefrontRunOptions& options) {
  validate_persistent_options(options);
  if (!bvh.impl_ || !bvh.impl_->d_triangles || !bvh.impl_->bvh.nodes || !bvh.impl_->bvh.primIDs) {
    throw std::runtime_error("run_persistent_harmonic called with an empty cuBQL BVH");
  }

  const int n = options.samples;
  const int block_size = options.block_size;
  const int grid_size = (n + block_size - 1) / block_size;
  const int reduce_grid = (n + kPersistentReduceBlockSize - 1) / kPersistentReduceBlockSize;

  float* d_sample_value = nullptr;
  int* d_step_count = nullptr;
  int* d_forced_max_steps = nullptr;
  int* d_query_overflow = nullptr;
  double* d_block_sum = nullptr;
  double* d_block_sum_sq = nullptr;
  double* d_block_steps = nullptr;
  unsigned long long* d_block_forced = nullptr;
  unsigned long long* d_block_overflow = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  try {
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sample_value), n * sizeof(float)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_step_count), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_forced_max_steps), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_overflow), n * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_sum), reduce_grid * sizeof(double)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_sum_sq), reduce_grid * sizeof(double)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_steps), reduce_grid * sizeof(double)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_forced), reduce_grid * sizeof(unsigned long long)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_overflow), reduce_grid * sizeof(unsigned long long)));

    N2WOS_CUDA_CHECK(cudaEventCreate(&start));
    N2WOS_CUDA_CHECK(cudaEventCreate(&stop));
    N2WOS_CUDA_CHECK(cudaEventRecord(start));

    persistent_harmonic_wos_kernel<<<grid_size, block_size>>>(n,
                                                              persistent_to_device(options.x0),
                                                              options.seed,
                                                              persistent_method_to_device(method),
                                                              options.depth_m,
                                                              options.max_steps,
                                                              options.epsilon,
                                                              options.step_scale,
                                                              bvh.impl_->d_triangles,
                                                              bvh.impl_->bvh,
                                                              d_sample_value,
                                                              d_step_count,
                                                              d_forced_max_steps,
                                                              d_query_overflow);
    N2WOS_CUDA_CHECK(cudaGetLastError());

    persistent_reduce_stats_kernel<<<reduce_grid, kPersistentReduceBlockSize>>>(d_sample_value,
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
    stats.launched_query_slots = steps + static_cast<double>(n);
    stats.mean_launched_query_slots_per_sample = stats.mean_steps + 1.0;
    stats.scheduled_query_rounds = 1;
    stats.forced_max_steps = forced;
    stats.overflow_count = overflow;

    N2WOS_CUDA_CHECK(cudaEventDestroy(start));
    start = nullptr;
    N2WOS_CUDA_CHECK(cudaEventDestroy(stop));
    stop = nullptr;
    cudaFree(d_sample_value);
    cudaFree(d_step_count);
    cudaFree(d_forced_max_steps);
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
    if (d_sample_value) cudaFree(d_sample_value);
    if (d_step_count) cudaFree(d_step_count);
    if (d_forced_max_steps) cudaFree(d_forced_max_steps);
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
