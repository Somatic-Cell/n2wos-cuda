#include "n2wos/cuda_bvh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

DeviceVec3 to_device_vec3(const Vec3f& p) {
  return DeviceVec3{p.x, p.y, p.z};
}

Vec3f from_device_vec3(const DeviceVec3& p) {
  return Vec3f{p.x, p.y, p.z};
}

__device__ DeviceVec3 closest_point_triangle_device(DeviceVec3 p, DeviceVec3 a, DeviceVec3 b, DeviceVec3 c) {
  const DeviceVec3 ab = d_sub(b, a);
  const DeviceVec3 ac = d_sub(c, a);
  const DeviceVec3 ap = d_sub(p, a);
  const float d1 = d_dot(ab, ap);
  const float d2 = d_dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return a;

  const DeviceVec3 bp = d_sub(p, b);
  const float d3 = d_dot(ab, bp);
  const float d4 = d_dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return b;

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = d1 / (d1 - d3);
    return d_add(a, d_mul(ab, v));
  }

  const DeviceVec3 cp = d_sub(p, c);
  const float d5 = d_dot(ab, cp);
  const float d6 = d_dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return c;

  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = d2 / (d2 - d6);
    return d_add(a, d_mul(ac, w));
  }

  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return d_add(b, d_mul(d_sub(c, b), w));
  }

  const float denom = 1.0f / (va + vb + vc);
  const float v = vb * denom;
  const float w = vc * denom;
  return d_add(a, d_add(d_mul(ab, v), d_mul(ac, w)));
}

__global__ void closest_point_bvh_kernel(const DeviceVec3* __restrict__ query_points,
                                         int query_count,
                                         const DeviceTriangle* __restrict__ triangles,
                                         const DeviceBvhNode* __restrict__ nodes,
                                         const int* __restrict__ triangle_indices,
                                         float* __restrict__ out_distance2,
                                         DeviceVec3* __restrict__ out_closest,
                                         int* __restrict__ out_triangle_id,
                                         int* __restrict__ out_overflow) {
  const int qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= query_count) return;

  const DeviceVec3 p = query_points[qid];
  float best_d2 = FLT_MAX;
  DeviceVec3 best_p = d_make_vec3(0.0f, 0.0f, 0.0f);
  int best_tri = -1;

  constexpr int kStackCapacity = 128;
  int stack[kStackCapacity];
  int sp = 0;
  int overflow = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    const int node_id = stack[--sp];
    const DeviceBvhNode node = nodes[node_id];
    const float node_d2 = d_aabb_distance2(p, node.bbox_min, node.bbox_max);
    if (node_d2 > best_d2) {
      continue;
    }

    if (node.count > 0) {
      for (int i = 0; i < node.count; ++i) {
        const int tri_id = triangle_indices[node.first + i];
        const DeviceTriangle tri = triangles[tri_id];
        const DeviceVec3 cp = closest_point_triangle_device(p, tri.v0, tri.v1, tri.v2);
        const float d2 = d_length2(d_sub(cp, p));
        if (d2 < best_d2) {
          best_d2 = d2;
          best_p = cp;
          best_tri = tri_id;
        }
      }
    } else {
      const int left = node.left;
      const int right = node.right;
      if (left >= 0 && right >= 0) {
        const DeviceBvhNode left_node = nodes[left];
        const DeviceBvhNode right_node = nodes[right];
        const float left_d2 = d_aabb_distance2(p, left_node.bbox_min, left_node.bbox_max);
        const float right_d2 = d_aabb_distance2(p, right_node.bbox_min, right_node.bbox_max);

        int near_id = left;
        int far_id = right;
        float near_d2 = left_d2;
        float far_d2 = right_d2;
        if (right_d2 < left_d2) {
          near_id = right;
          far_id = left;
          near_d2 = right_d2;
          far_d2 = left_d2;
        }

        if (far_d2 <= best_d2) {
          if (sp < kStackCapacity) {
            stack[sp++] = far_id;
          } else {
            overflow = 1;
          }
        }
        if (near_d2 <= best_d2) {
          if (sp < kStackCapacity) {
            stack[sp++] = near_id;
          } else {
            overflow = 1;
          }
        }
      }
    }
  }

  out_distance2[qid] = best_d2;
  out_closest[qid] = best_p;
  out_triangle_id[qid] = best_tri;
  out_overflow[qid] = overflow;
}

void copy_to_device(void** dst, const void* src, std::size_t bytes) {
  *dst = nullptr;
  if (bytes == 0) return;
  N2WOS_CUDA_CHECK(cudaMalloc(dst, bytes));
  N2WOS_CUDA_CHECK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
}

}  // namespace

CudaBvh::CudaBvh(const Mesh& mesh, int leaf_size) {
  HostBvhData host = build_host_bvh(mesh, leaf_size);
  triangle_count_ = host.triangles.size();
  node_count_ = host.nodes.size();
  index_count_ = host.triangle_indices.size();
  leaf_size_ = host.leaf_size;
  max_depth_ = host.max_depth;

  copy_to_device(reinterpret_cast<void**>(&d_triangles_), host.triangles.data(), host.triangles.size() * sizeof(DeviceTriangle));
  copy_to_device(reinterpret_cast<void**>(&d_nodes_), host.nodes.data(), host.nodes.size() * sizeof(DeviceBvhNode));
  copy_to_device(reinterpret_cast<void**>(&d_triangle_indices_), host.triangle_indices.data(),
                 host.triangle_indices.size() * sizeof(int));
}

CudaBvh::~CudaBvh() {
  release();
}

CudaBvh::CudaBvh(CudaBvh&& other) noexcept {
  *this = std::move(other);
}

CudaBvh& CudaBvh::operator=(CudaBvh&& other) noexcept {
  if (this == &other) return *this;
  release();
  d_triangles_ = other.d_triangles_;
  d_nodes_ = other.d_nodes_;
  d_triangle_indices_ = other.d_triangle_indices_;
  triangle_count_ = other.triangle_count_;
  node_count_ = other.node_count_;
  index_count_ = other.index_count_;
  leaf_size_ = other.leaf_size_;
  max_depth_ = other.max_depth_;

  other.d_triangles_ = nullptr;
  other.d_nodes_ = nullptr;
  other.d_triangle_indices_ = nullptr;
  other.triangle_count_ = 0;
  other.node_count_ = 0;
  other.index_count_ = 0;
  other.leaf_size_ = 8;
  other.max_depth_ = 0;
  return *this;
}

void CudaBvh::release() {
  if (d_triangles_) cudaFree(d_triangles_);
  if (d_nodes_) cudaFree(d_nodes_);
  if (d_triangle_indices_) cudaFree(d_triangle_indices_);
  d_triangles_ = nullptr;
  d_nodes_ = nullptr;
  d_triangle_indices_ = nullptr;
  triangle_count_ = 0;
  node_count_ = 0;
  index_count_ = 0;
}

CudaBvhQueryResult CudaBvh::query(const std::vector<Vec3f>& points, int block_size) const {
  if (!d_triangles_ || !d_nodes_ || !d_triangle_indices_) {
    throw std::runtime_error("CudaBvh::query called on empty BVH");
  }
  if (points.empty()) {
    return {};
  }
  if (block_size <= 0 || block_size > 1024) {
    throw std::runtime_error("CUDA block size must be in [1, 1024]");
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

    const int query_count = static_cast<int>(points.size());
    const int grid_size = (query_count + block_size - 1) / block_size;

    N2WOS_CUDA_CHECK(cudaEventRecord(start));
    closest_point_bvh_kernel<<<grid_size, block_size>>>(d_points,
                                                        query_count,
                                                        d_triangles_,
                                                        d_nodes_,
                                                        d_triangle_indices_,
                                                        d_distance2,
                                                        d_closest,
                                                        d_triangle_id,
                                                        d_overflow);
    N2WOS_CUDA_CHECK(cudaEventRecord(stop));
    N2WOS_CUDA_CHECK(cudaGetLastError());
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

std::string cuda_runtime_version_string() {
  int runtime_version = 0;
  int driver_version = 0;
  N2WOS_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  N2WOS_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  std::ostringstream out;
  out << "runtime=" << runtime_version << ",driver=" << driver_version;
  return out.str();
}

std::string cuda_device_summary() {
  int device = 0;
  N2WOS_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  N2WOS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::ostringstream out;
  out << prop.name << " sm_" << prop.major << prop.minor
      << " global_mem=" << static_cast<unsigned long long>(prop.totalGlobalMem);
  return out.str();
}

}  // namespace n2wos
