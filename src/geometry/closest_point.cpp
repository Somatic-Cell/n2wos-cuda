#include "n2wos/closest_point.hpp"

#include <limits>

namespace n2wos {

Vec3f closest_point_on_triangle(const Vec3f& p, const Vec3f& a, const Vec3f& b, const Vec3f& c) {
  // Christer Ericson, Real-Time Collision Detection, section 5.1.5.
  const Vec3f ab = b - a;
  const Vec3f ac = c - a;
  const Vec3f ap = p - a;
  const float d1 = dot(ab, ap);
  const float d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return a;

  const Vec3f bp = p - b;
  const float d3 = dot(ab, bp);
  const float d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return b;

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = d1 / (d1 - d3);
    return a + ab * v;
  }

  const Vec3f cp = p - c;
  const float d5 = dot(ab, cp);
  const float d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return c;

  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = d2 / (d2 - d6);
    return a + ac * w;
  }

  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + (c - b) * w;
  }

  const float denom = 1.0f / (va + vb + vc);
  const float v = vb * denom;
  const float w = vc * denom;
  return a + ab * v + ac * w;
}

ClosestPointResult closest_point_bruteforce(const Mesh& mesh, const Vec3f& p) {
  require_mesh_valid(mesh);

  ClosestPointResult best;
  best.distance2 = std::numeric_limits<float>::infinity();

  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    const Triangle& tri = mesh.triangles[i];
    const Vec3f cp = closest_point_on_triangle(p, mesh.vertices[tri.v0], mesh.vertices[tri.v1], mesh.vertices[tri.v2]);
    const float d2 = length2(cp - p);
    if (d2 < best.distance2) {
      best.distance2 = d2;
      best.closest = cp;
      best.triangle_id = static_cast<int>(i);
    }
  }

  return best;
}

std::size_t count_degenerate_triangles(const Mesh& mesh, float area2_epsilon) {
  require_mesh_valid(mesh);
  std::size_t count = 0;
  for (const Triangle& tri : mesh.triangles) {
    const Vec3f a = mesh.vertices[tri.v0];
    const Vec3f b = mesh.vertices[tri.v1];
    const Vec3f c = mesh.vertices[tri.v2];
    const float area2x4 = length2(cross(b - a, c - a));
    if (area2x4 <= area2_epsilon) {
      ++count;
    }
  }
  return count;
}

}  // namespace n2wos
