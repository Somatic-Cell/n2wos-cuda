#pragma once

#include <string>

#include "n2wos/types.hpp"

namespace n2wos {

Mesh load_obj_mesh(const std::string& path);
Mesh make_procedural_bumpy_sphere(int stacks, int slices, float bump_amplitude = 0.15f);

struct NormalizeTransform {
  Vec3f center;
  float scale = 1.0f;
};

NormalizeTransform normalize_to_unit_radius(Mesh& mesh);
Aabb compute_bounds(const Mesh& mesh);

}  // namespace n2wos
