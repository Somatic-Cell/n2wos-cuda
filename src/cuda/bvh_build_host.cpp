#include "n2wos/cuda_bvh.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace n2wos {
namespace {

DeviceVec3 to_device_vec3(const Vec3f& p) {
  return DeviceVec3{p.x, p.y, p.z};
}

Vec3f from_device_vec3(const DeviceVec3& p) {
  return Vec3f{p.x, p.y, p.z};
}

struct BuildPrimitive {
  int triangle_id = -1;
  Aabb bounds;
  Vec3f centroid;
};

Aabb triangle_bounds(const DeviceTriangle& tri) {
  Aabb box = empty_aabb();
  expand(box, from_device_vec3(tri.v0));
  expand(box, from_device_vec3(tri.v1));
  expand(box, from_device_vec3(tri.v2));
  return box;
}

int largest_axis(const Vec3f& extent) {
  if (extent.x >= extent.y && extent.x >= extent.z) return 0;
  if (extent.y >= extent.x && extent.y >= extent.z) return 1;
  return 2;
}

float axis_value(const Vec3f& p, int axis) {
  if (axis == 0) return p.x;
  if (axis == 1) return p.y;
  return p.z;
}

class BvhBuilder {
 public:
  BvhBuilder(const std::vector<BuildPrimitive>& primitives, int leaf_size)
      : primitives_(primitives), leaf_size_(leaf_size) {}

  HostBvhData build(std::vector<DeviceTriangle> triangles) {
    if (primitives_.empty()) {
      throw std::runtime_error("cannot build BVH with no primitives");
    }
    if (leaf_size_ < 1) {
      throw std::runtime_error("leaf size must be positive");
    }

    HostBvhData out;
    out.triangles = std::move(triangles);
    out.leaf_size = leaf_size_;

    std::vector<int> primitive_indices(primitives_.size());
    std::iota(primitive_indices.begin(), primitive_indices.end(), 0);
    build_recursive(out, primitive_indices.begin(), primitive_indices.end(), 0);
    out.max_depth = max_depth_;
    return out;
  }

 private:
  using It = std::vector<int>::iterator;

  int build_recursive(HostBvhData& out, It begin, It end, int depth) {
    max_depth_ = std::max(max_depth_, depth);

    Aabb bounds = empty_aabb();
    Aabb centroid_bounds = empty_aabb();
    for (It it = begin; it != end; ++it) {
      const BuildPrimitive& prim = primitives_[static_cast<std::size_t>(*it)];
      expand(bounds, prim.bounds);
      expand(centroid_bounds, prim.centroid);
    }

    const int node_index = static_cast<int>(out.nodes.size());
    DeviceBvhNode node{};
    node.bbox_min = to_device_vec3(bounds.min);
    node.bbox_max = to_device_vec3(bounds.max);
    node.left = -1;
    node.right = -1;
    node.first = -1;
    node.count = 0;
    out.nodes.push_back(node);

    const int count = static_cast<int>(std::distance(begin, end));
    if (count <= leaf_size_) {
      const int first = static_cast<int>(out.triangle_indices.size());
      for (It it = begin; it != end; ++it) {
        out.triangle_indices.push_back(primitives_[static_cast<std::size_t>(*it)].triangle_id);
      }
      out.nodes[static_cast<std::size_t>(node_index)].first = first;
      out.nodes[static_cast<std::size_t>(node_index)].count = count;
      return node_index;
    }

    const Vec3f extent = centroid_bounds.max - centroid_bounds.min;
    const int axis = largest_axis(extent);
    It mid = begin + count / 2;
    std::nth_element(begin, mid, end, [&](int a, int b) {
      return axis_value(primitives_[static_cast<std::size_t>(a)].centroid, axis) <
             axis_value(primitives_[static_cast<std::size_t>(b)].centroid, axis);
    });

    // Degenerate centroid splits can still be divided by index position due to nth_element.
    const int left = build_recursive(out, begin, mid, depth + 1);
    const int right = build_recursive(out, mid, end, depth + 1);
    out.nodes[static_cast<std::size_t>(node_index)].left = left;
    out.nodes[static_cast<std::size_t>(node_index)].right = right;
    return node_index;
  }

  const std::vector<BuildPrimitive>& primitives_;
  int leaf_size_ = 8;
  int max_depth_ = 0;
};

}  // namespace

HostBvhData build_host_bvh(const Mesh& mesh, int leaf_size) {
  require_mesh_valid(mesh);
  if (leaf_size < 1) {
    throw std::runtime_error("leaf size must be positive");
  }

  std::vector<DeviceTriangle> device_triangles;
  device_triangles.reserve(mesh.triangles.size());
  std::vector<BuildPrimitive> primitives;
  primitives.reserve(mesh.triangles.size());

  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    const Triangle& tri = mesh.triangles[i];
    const DeviceTriangle dtri{to_device_vec3(mesh.vertices[tri.v0]),
                              to_device_vec3(mesh.vertices[tri.v1]),
                              to_device_vec3(mesh.vertices[tri.v2])};
    device_triangles.push_back(dtri);

    const Aabb bounds = triangle_bounds(dtri);
    const Vec3f centroid = (from_device_vec3(dtri.v0) + from_device_vec3(dtri.v1) + from_device_vec3(dtri.v2)) / 3.0f;
    primitives.push_back(BuildPrimitive{static_cast<int>(i), bounds, centroid});
  }

  BvhBuilder builder(primitives, leaf_size);
  HostBvhData bvh = builder.build(std::move(device_triangles));
  if (bvh.nodes.empty() || bvh.triangle_indices.size() != mesh.triangles.size()) {
    throw std::runtime_error("internal BVH build error");
  }
  return bvh;
}

}  // namespace n2wos
