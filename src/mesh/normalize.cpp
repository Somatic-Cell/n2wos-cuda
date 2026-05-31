#include "n2wos/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace n2wos {

Aabb compute_bounds(const Mesh& mesh) {
  require_mesh_valid(mesh);
  Aabb bounds = empty_aabb();
  for (const Vec3f& p : mesh.vertices) {
    expand(bounds, p);
  }
  return bounds;
}

NormalizeTransform normalize_to_unit_radius(Mesh& mesh) {
  require_mesh_valid(mesh);

  Aabb bounds = compute_bounds(mesh);
  Vec3f center = (bounds.min + bounds.max) * 0.5f;

  float radius = 0.0f;
  for (const Vec3f& p : mesh.vertices) {
    radius = std::max(radius, length(p - center));
  }
  if (!(radius > 0.0f)) {
    throw std::runtime_error("cannot normalize mesh with zero radius");
  }

  for (Vec3f& p : mesh.vertices) {
    p = (p - center) / radius;
  }

  return NormalizeTransform{center, 1.0f / radius};
}

}  // namespace n2wos
