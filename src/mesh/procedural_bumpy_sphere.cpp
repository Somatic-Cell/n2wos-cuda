#include "n2wos/mesh.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace n2wos {
namespace {

constexpr float kPi = 3.14159265358979323846f;

float bump_radius(float theta, float phi, float amplitude) {
  const float low = std::sin(3.0f * theta + 0.35f * std::cos(5.0f * phi));
  const float high = std::sin(7.0f * phi + 0.5f * std::cos(2.0f * theta));
  const float mid = std::cos(5.0f * theta - 2.0f * phi);
  return 1.0f + amplitude * (0.45f * low * high + 0.35f * mid * std::sin(theta) + 0.20f * std::sin(4.0f * theta));
}

Vec3f sphere_point(float theta, float phi, float amplitude) {
  const float r = bump_radius(theta, phi, amplitude);
  const float st = std::sin(theta);
  return {r * st * std::cos(phi), r * st * std::sin(phi), r * std::cos(theta)};
}

std::uint32_t ring_index(int i, int j, int slices) {
  // i is 1..stacks-1 for interior rings.
  return static_cast<std::uint32_t>(1 + (i - 1) * slices + (j % slices));
}

}  // namespace

Mesh make_procedural_bumpy_sphere(int stacks, int slices, float bump_amplitude) {
  if (stacks < 4) {
    throw std::runtime_error("bumpy sphere requires stacks >= 4");
  }
  if (slices < 8) {
    throw std::runtime_error("bumpy sphere requires slices >= 8");
  }
  if (bump_amplitude < 0.0f || bump_amplitude > 0.5f) {
    throw std::runtime_error("bumpy sphere bump amplitude must be in [0, 0.5]");
  }

  Mesh mesh;
  mesh.name = "procedural_bumpy_sphere";
  mesh.vertices.reserve(2 + static_cast<std::size_t>(stacks - 1) * static_cast<std::size_t>(slices));

  const std::uint32_t north = 0;
  mesh.vertices.push_back(sphere_point(0.0f, 0.0f, bump_amplitude));

  for (int i = 1; i <= stacks - 1; ++i) {
    const float theta = kPi * static_cast<float>(i) / static_cast<float>(stacks);
    for (int j = 0; j < slices; ++j) {
      const float phi = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
      mesh.vertices.push_back(sphere_point(theta, phi, bump_amplitude));
    }
  }

  const std::uint32_t south = static_cast<std::uint32_t>(mesh.vertices.size());
  mesh.vertices.push_back(sphere_point(kPi, 0.0f, bump_amplitude));

  mesh.triangles.reserve(static_cast<std::size_t>(2 * slices * stacks));

  // North cap.
  for (int j = 0; j < slices; ++j) {
    const std::uint32_t a = ring_index(1, j, slices);
    const std::uint32_t b = ring_index(1, j + 1, slices);
    mesh.triangles.push_back({north, a, b});
  }

  // Interior bands.
  for (int i = 1; i < stacks - 1; ++i) {
    for (int j = 0; j < slices; ++j) {
      const std::uint32_t a = ring_index(i, j, slices);
      const std::uint32_t b = ring_index(i, j + 1, slices);
      const std::uint32_t c = ring_index(i + 1, j, slices);
      const std::uint32_t d = ring_index(i + 1, j + 1, slices);
      mesh.triangles.push_back({a, c, b});
      mesh.triangles.push_back({b, c, d});
    }
  }

  // South cap.
  for (int j = 0; j < slices; ++j) {
    const std::uint32_t a = ring_index(stacks - 1, j, slices);
    const std::uint32_t b = ring_index(stacks - 1, j + 1, slices);
    mesh.triangles.push_back({a, south, b});
  }

  return mesh;
}

}  // namespace n2wos
