#pragma once

#include <cstddef>

#include "n2wos/types.hpp"

namespace n2wos {

struct ClosestPointResult {
  float distance2 = 0.0f;
  Vec3f closest;
  int triangle_id = -1;
};

Vec3f closest_point_on_triangle(const Vec3f& p, const Vec3f& a, const Vec3f& b, const Vec3f& c);
ClosestPointResult closest_point_bruteforce(const Mesh& mesh, const Vec3f& p);
std::size_t count_degenerate_triangles(const Mesh& mesh, float area2_epsilon = 1.0e-20f);

}  // namespace n2wos
