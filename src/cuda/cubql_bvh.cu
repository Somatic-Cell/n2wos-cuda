#include "n2wos/cubql_bvh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
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
  if (build_method == "rebin" || build_method == "robust_radix" || build_method == "modified_radix") return "rebin_radix";
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
  if (build_method == "spatial_median" || build_method == "radix" || build_method == "rebin_radix") {
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
                           " (expected spatial_median, sah, elh, radix, or rebin_radix)");
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
  } else if (build_method == "rebin_radix") {
    cuBQL::cuda::rebinRadixBuilder(bvh, d_boxes, primitive_count, config);
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
                                           int query_count,
                                           const cuBQL::Triangle* __restrict__ triangles,
                                           cuBQL::bvh3f bvh,
                                           float* __restrict__ out_distance2,
                                           DeviceVec3* __restrict__ out_closest,
                                           int* __restrict__ out_triangle_id,
                                           int* __restrict__ out_overflow) {
  const int qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= query_count) return;

  const cuBQL::vec3f query_point = to_cubql_vec3(query_points[qid]);
  cuBQL::triangles::CPAT cpat;
  cpat.runQuery(triangles, bvh, query_point);

  out_distance2[qid] = cpat.sqrDist;
  out_closest[qid] = from_cubql_vec3(cpat.P);
  out_triangle_id[qid] = cpat.triangleIdx;
  out_overflow[qid] = 0;
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

}  // namespace n2wos
