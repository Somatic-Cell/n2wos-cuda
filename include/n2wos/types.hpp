#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace n2wos {

struct Vec3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

inline Vec3f make_vec3(float x, float y, float z) { return Vec3f{x, y, z}; }
inline Vec3f operator+(const Vec3f& a, const Vec3f& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3f operator-(const Vec3f& a, const Vec3f& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3f operator*(const Vec3f& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3f operator*(float s, const Vec3f& a) { return a * s; }
inline Vec3f operator/(const Vec3f& a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3f& operator+=(Vec3f& a, const Vec3f& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
inline Vec3f& operator-=(Vec3f& a, const Vec3f& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

inline float dot(const Vec3f& a, const Vec3f& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3f cross(const Vec3f& a, const Vec3f& b) {
  return {a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

inline float length2(const Vec3f& a) { return dot(a, a); }
inline float length(const Vec3f& a) { return std::sqrt(length2(a)); }

inline Vec3f min_vec(const Vec3f& a, const Vec3f& b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3f max_vec(const Vec3f& a, const Vec3f& b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

struct Triangle {
  std::uint32_t v0 = 0;
  std::uint32_t v1 = 0;
  std::uint32_t v2 = 0;
};

struct Aabb {
  Vec3f min;
  Vec3f max;
};

inline Aabb empty_aabb() {
  const float inf = std::numeric_limits<float>::infinity();
  return {{inf, inf, inf}, {-inf, -inf, -inf}};
}

inline void expand(Aabb& box, const Vec3f& p) {
  box.min = min_vec(box.min, p);
  box.max = max_vec(box.max, p);
}

inline void expand(Aabb& box, const Aabb& other) {
  expand(box, other.min);
  expand(box, other.max);
}

struct Mesh {
  std::vector<Vec3f> vertices;
  std::vector<Triangle> triangles;
  std::string name;
};

inline void require_mesh_valid(const Mesh& mesh) {
  if (mesh.vertices.empty()) {
    throw std::runtime_error("mesh has no vertices");
  }
  if (mesh.triangles.empty()) {
    throw std::runtime_error("mesh has no triangles");
  }
  for (const Triangle& tri : mesh.triangles) {
    if (tri.v0 >= mesh.vertices.size() || tri.v1 >= mesh.vertices.size() || tri.v2 >= mesh.vertices.size()) {
      throw std::runtime_error("mesh triangle contains out-of-range vertex index");
    }
  }
}

}  // namespace n2wos
