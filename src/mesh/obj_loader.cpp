#include "n2wos/mesh.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace n2wos {
namespace {

std::string trim_copy(const std::string& s) {
  std::size_t first = 0;
  while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
    ++first;
  }
  std::size_t last = s.size();
  while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
    --last;
  }
  return s.substr(first, last - first);
}

std::int64_t parse_integer_prefix(const std::string& token) {
  const std::size_t slash = token.find('/');
  const std::string head = slash == std::string::npos ? token : token.substr(0, slash);
  if (head.empty()) {
    throw std::runtime_error("OBJ face token has empty vertex index: " + token);
  }
  std::size_t consumed = 0;
  const long long value = std::stoll(head, &consumed, 10);
  if (consumed != head.size()) {
    throw std::runtime_error("OBJ face token has invalid vertex index: " + token);
  }
  return static_cast<std::int64_t>(value);
}

std::uint32_t parse_obj_vertex_index(const std::string& token, std::size_t vertex_count) {
  const std::int64_t raw = parse_integer_prefix(token);
  std::int64_t zero_based = 0;
  if (raw > 0) {
    zero_based = raw - 1;
  } else if (raw < 0) {
    zero_based = static_cast<std::int64_t>(vertex_count) + raw;
  } else {
    throw std::runtime_error("OBJ indices are 1-based; index 0 is invalid");
  }
  if (zero_based < 0 || zero_based >= static_cast<std::int64_t>(vertex_count)) {
    throw std::runtime_error("OBJ face index out of range: " + token);
  }
  return static_cast<std::uint32_t>(zero_based);
}

std::string basename(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

Mesh load_obj_mesh(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open OBJ file: " + path);
  }

  Mesh mesh;
  mesh.name = basename(path);

  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    line = trim_copy(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "v") {
      Vec3f p;
      if (!(ss >> p.x >> p.y >> p.z)) {
        throw std::runtime_error("invalid OBJ vertex at line " + std::to_string(line_number));
      }
      mesh.vertices.push_back(p);
    } else if (tag == "f") {
      std::vector<std::uint32_t> face;
      std::string token;
      while (ss >> token) {
        face.push_back(parse_obj_vertex_index(token, mesh.vertices.size()));
      }
      if (face.size() < 3) {
        throw std::runtime_error("OBJ face has fewer than 3 vertices at line " + std::to_string(line_number));
      }
      for (std::size_t i = 1; i + 1 < face.size(); ++i) {
        mesh.triangles.push_back({face[0], face[i], face[i + 1]});
      }
    }
  }

  require_mesh_valid(mesh);
  return mesh;
}

}  // namespace n2wos
