#include "n2wos/mesh.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace n2wos {
namespace {

enum class PlyFormat {
  Ascii,
  BinaryLittleEndian,
};

struct PlyProperty {
  bool is_list = false;
  std::string name;
  std::string scalar_type;
  std::string count_type;
  std::string item_type;
};

struct PlyElement {
  std::string name;
  std::size_t count = 0;
  std::vector<PlyProperty> properties;
};

struct PlyHeader {
  PlyFormat format = PlyFormat::Ascii;
  std::vector<PlyElement> elements;
};

std::string trim_copy(const std::string& s) {
  std::size_t first = 0;
  while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) ++first;
  std::size_t last = s.size();
  while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) --last;
  return s.substr(first, last - first);
}

std::string basename(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string lowercase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool is_float_type(const std::string& type) {
  const std::string t = lowercase(type);
  return t == "float" || t == "float32" || t == "double" || t == "float64";
}

bool is_supported_scalar_type(const std::string& type) {
  const std::string t = lowercase(type);
  return t == "char" || t == "int8" || t == "uchar" || t == "uint8" ||
         t == "short" || t == "int16" || t == "ushort" || t == "uint16" ||
         t == "int" || t == "int32" || t == "uint" || t == "uint32" ||
         t == "float" || t == "float32" || t == "double" || t == "float64";
}

std::size_t scalar_size(const std::string& type) {
  const std::string t = lowercase(type);
  if (t == "char" || t == "int8" || t == "uchar" || t == "uint8") return 1;
  if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
  if (t == "int" || t == "int32" || t == "uint" || t == "uint32" || t == "float" || t == "float32") return 4;
  if (t == "double" || t == "float64") return 8;
  throw std::runtime_error("unsupported PLY scalar type: " + type);
}

template <typename T>
T read_pod(std::istream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("unexpected end of binary PLY data");
  return value;
}

double read_binary_scalar_as_double(std::istream& in, const std::string& type) {
  const std::string t = lowercase(type);
  if (t == "char" || t == "int8") return static_cast<double>(read_pod<std::int8_t>(in));
  if (t == "uchar" || t == "uint8") return static_cast<double>(read_pod<std::uint8_t>(in));
  if (t == "short" || t == "int16") return static_cast<double>(read_pod<std::int16_t>(in));
  if (t == "ushort" || t == "uint16") return static_cast<double>(read_pod<std::uint16_t>(in));
  if (t == "int" || t == "int32") return static_cast<double>(read_pod<std::int32_t>(in));
  if (t == "uint" || t == "uint32") return static_cast<double>(read_pod<std::uint32_t>(in));
  if (t == "float" || t == "float32") return static_cast<double>(read_pod<float>(in));
  if (t == "double" || t == "float64") return read_pod<double>(in);
  throw std::runtime_error("unsupported PLY scalar type: " + type);
}

std::int64_t read_binary_scalar_as_int64(std::istream& in, const std::string& type) {
  const std::string t = lowercase(type);
  if (t == "char" || t == "int8") return static_cast<std::int64_t>(read_pod<std::int8_t>(in));
  if (t == "uchar" || t == "uint8") return static_cast<std::int64_t>(read_pod<std::uint8_t>(in));
  if (t == "short" || t == "int16") return static_cast<std::int64_t>(read_pod<std::int16_t>(in));
  if (t == "ushort" || t == "uint16") return static_cast<std::int64_t>(read_pod<std::uint16_t>(in));
  if (t == "int" || t == "int32") return static_cast<std::int64_t>(read_pod<std::int32_t>(in));
  if (t == "uint" || t == "uint32") return static_cast<std::int64_t>(read_pod<std::uint32_t>(in));
  throw std::runtime_error("PLY integer field uses non-integer type: " + type);
}

void skip_binary_scalar(std::istream& in, const std::string& type) {
  const std::size_t n = scalar_size(type);
  char buffer[8];
  if (n > sizeof(buffer)) throw std::runtime_error("internal PLY scalar skip size error");
  in.read(buffer, static_cast<std::streamsize>(n));
  if (!in) throw std::runtime_error("unexpected end of binary PLY data while skipping scalar");
}

std::uint32_t checked_vertex_index(std::int64_t index, std::size_t vertex_count, const std::string& context) {
  if (index < 0 || index >= static_cast<std::int64_t>(vertex_count)) {
    throw std::runtime_error("PLY face vertex index out of range in " + context + ": " + std::to_string(index));
  }
  return static_cast<std::uint32_t>(index);
}

void triangulate_face(Mesh& mesh, const std::vector<std::uint32_t>& face, const std::string& context) {
  if (face.size() < 3) {
    throw std::runtime_error("PLY face has fewer than 3 vertices in " + context);
  }
  for (std::size_t i = 1; i + 1 < face.size(); ++i) {
    mesh.triangles.push_back({face[0], face[i], face[i + 1]});
  }
}

const PlyElement* find_element(const PlyHeader& header, const std::string& name) {
  for (const PlyElement& element : header.elements) {
    if (element.name == name) return &element;
  }
  return nullptr;
}

int find_property_index(const PlyElement& element, const std::string& name) {
  for (std::size_t i = 0; i < element.properties.size(); ++i) {
    if (element.properties[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

int find_face_index_property(const PlyElement& face_element) {
  for (std::size_t i = 0; i < face_element.properties.size(); ++i) {
    const PlyProperty& prop = face_element.properties[i];
    if (!prop.is_list) continue;
    if (prop.name == "vertex_indices" || prop.name == "vertex_index" || prop.name == "vertices") {
      return static_cast<int>(i);
    }
  }
  for (std::size_t i = 0; i < face_element.properties.size(); ++i) {
    if (face_element.properties[i].is_list) return static_cast<int>(i);
  }
  return -1;
}

PlyHeader parse_header(std::istream& in) {
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty PLY file");
  if (trim_copy(line) != "ply") throw std::runtime_error("file does not start with PLY magic header");

  PlyHeader header;
  PlyElement* current_element = nullptr;
  std::size_t line_number = 1;
  bool seen_format = false;

  while (std::getline(in, line)) {
    ++line_number;
    line = trim_copy(line);
    if (line.empty()) continue;
    if (line == "end_header") break;

    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "comment" || tag == "obj_info") {
      continue;
    } else if (tag == "format") {
      std::string format_name;
      std::string version;
      ss >> format_name >> version;
      if (format_name == "ascii") {
        header.format = PlyFormat::Ascii;
      } else if (format_name == "binary_little_endian") {
        header.format = PlyFormat::BinaryLittleEndian;
      } else if (format_name == "binary_big_endian") {
        throw std::runtime_error("binary_big_endian PLY is not supported; convert to ascii or binary_little_endian");
      } else {
        throw std::runtime_error("unsupported PLY format at line " + std::to_string(line_number) + ": " + format_name);
      }
      seen_format = true;
    } else if (tag == "element") {
      std::string name;
      std::size_t count = 0;
      if (!(ss >> name >> count)) {
        throw std::runtime_error("invalid PLY element declaration at line " + std::to_string(line_number));
      }
      header.elements.push_back(PlyElement{name, count, {}});
      current_element = &header.elements.back();
    } else if (tag == "property") {
      if (!current_element) {
        throw std::runtime_error("PLY property appears before any element at line " + std::to_string(line_number));
      }
      std::string maybe_list;
      ss >> maybe_list;
      if (maybe_list == "list") {
        PlyProperty prop;
        prop.is_list = true;
        if (!(ss >> prop.count_type >> prop.item_type >> prop.name)) {
          throw std::runtime_error("invalid PLY list property at line " + std::to_string(line_number));
        }
        if (!is_supported_scalar_type(prop.count_type) || !is_supported_scalar_type(prop.item_type)) {
          throw std::runtime_error("unsupported PLY list property type at line " + std::to_string(line_number));
        }
        current_element->properties.push_back(prop);
      } else {
        PlyProperty prop;
        prop.is_list = false;
        prop.scalar_type = maybe_list;
        if (!(ss >> prop.name)) {
          throw std::runtime_error("invalid PLY scalar property at line " + std::to_string(line_number));
        }
        if (!is_supported_scalar_type(prop.scalar_type)) {
          throw std::runtime_error("unsupported PLY scalar property type at line " + std::to_string(line_number));
        }
        current_element->properties.push_back(prop);
      }
    }
  }

  if (!seen_format) throw std::runtime_error("PLY header has no format declaration");
  if (!in) throw std::runtime_error("PLY header ended before end_header");
  return header;
}

void parse_ascii_vertices(std::istream& in, const PlyElement& vertex_element, Mesh& mesh) {
  const int x_index = find_property_index(vertex_element, "x");
  const int y_index = find_property_index(vertex_element, "y");
  const int z_index = find_property_index(vertex_element, "z");
  if (x_index < 0 || y_index < 0 || z_index < 0) {
    throw std::runtime_error("PLY vertex element must contain scalar x, y, z properties");
  }
  for (std::size_t i = 0; i < vertex_element.count; ++i) {
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("unexpected end of ASCII PLY vertex data");
    std::istringstream ss(line);
    Vec3f p{};
    for (std::size_t pi = 0; pi < vertex_element.properties.size(); ++pi) {
      const PlyProperty& prop = vertex_element.properties[pi];
      if (prop.is_list) {
        int count = 0;
        if (!(ss >> count) || count < 0) throw std::runtime_error("invalid list property in ASCII PLY vertex element");
        for (int k = 0; k < count; ++k) {
          std::string ignored;
          if (!(ss >> ignored)) throw std::runtime_error("truncated list property in ASCII PLY vertex element");
        }
      } else {
        std::string token;
        if (!(ss >> token)) throw std::runtime_error("truncated ASCII PLY vertex line");
        const float value = static_cast<float>(std::stod(token));
        if (static_cast<int>(pi) == x_index) p.x = value;
        if (static_cast<int>(pi) == y_index) p.y = value;
        if (static_cast<int>(pi) == z_index) p.z = value;
      }
    }
    mesh.vertices.push_back(p);
  }
}

void parse_ascii_faces(std::istream& in, const PlyElement& face_element, Mesh& mesh) {
  const int index_prop = find_face_index_property(face_element);
  if (index_prop < 0) {
    throw std::runtime_error("PLY face element has no list property for vertex indices");
  }
  for (std::size_t face_id = 0; face_id < face_element.count; ++face_id) {
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("unexpected end of ASCII PLY face data");
    std::istringstream ss(line);
    std::vector<std::uint32_t> face;
    for (std::size_t pi = 0; pi < face_element.properties.size(); ++pi) {
      const PlyProperty& prop = face_element.properties[pi];
      if (prop.is_list) {
        int count = 0;
        if (!(ss >> count) || count < 0) throw std::runtime_error("invalid list count in ASCII PLY face element");
        std::vector<std::int64_t> values(static_cast<std::size_t>(count));
        for (int k = 0; k < count; ++k) {
          if (!(ss >> values[static_cast<std::size_t>(k)])) throw std::runtime_error("truncated list in ASCII PLY face element");
        }
        if (static_cast<int>(pi) == index_prop) {
          face.reserve(values.size());
          for (std::int64_t index : values) {
            face.push_back(checked_vertex_index(index, mesh.vertices.size(), "ASCII face " + std::to_string(face_id)));
          }
        }
      } else {
        std::string ignored;
        if (!(ss >> ignored)) throw std::runtime_error("truncated scalar property in ASCII PLY face element");
      }
    }
    triangulate_face(mesh, face, "ASCII face " + std::to_string(face_id));
  }
}

void parse_binary_vertices(std::istream& in, const PlyElement& vertex_element, Mesh& mesh) {
  const int x_index = find_property_index(vertex_element, "x");
  const int y_index = find_property_index(vertex_element, "y");
  const int z_index = find_property_index(vertex_element, "z");
  if (x_index < 0 || y_index < 0 || z_index < 0) {
    throw std::runtime_error("PLY vertex element must contain scalar x, y, z properties");
  }
  for (std::size_t i = 0; i < vertex_element.count; ++i) {
    Vec3f p{};
    for (std::size_t pi = 0; pi < vertex_element.properties.size(); ++pi) {
      const PlyProperty& prop = vertex_element.properties[pi];
      if (prop.is_list) {
        const std::int64_t count = read_binary_scalar_as_int64(in, prop.count_type);
        if (count < 0) throw std::runtime_error("negative list count in binary PLY vertex element");
        for (std::int64_t k = 0; k < count; ++k) skip_binary_scalar(in, prop.item_type);
      } else {
        if (!is_float_type(prop.scalar_type) && static_cast<int>(pi) == x_index) {
          throw std::runtime_error("PLY x property must be floating point");
        }
        const double value = read_binary_scalar_as_double(in, prop.scalar_type);
        if (static_cast<int>(pi) == x_index) p.x = static_cast<float>(value);
        if (static_cast<int>(pi) == y_index) p.y = static_cast<float>(value);
        if (static_cast<int>(pi) == z_index) p.z = static_cast<float>(value);
      }
    }
    mesh.vertices.push_back(p);
  }
}

void parse_binary_faces(std::istream& in, const PlyElement& face_element, Mesh& mesh) {
  const int index_prop = find_face_index_property(face_element);
  if (index_prop < 0) {
    throw std::runtime_error("PLY face element has no list property for vertex indices");
  }
  for (std::size_t face_id = 0; face_id < face_element.count; ++face_id) {
    std::vector<std::uint32_t> face;
    for (std::size_t pi = 0; pi < face_element.properties.size(); ++pi) {
      const PlyProperty& prop = face_element.properties[pi];
      if (prop.is_list) {
        const std::int64_t count = read_binary_scalar_as_int64(in, prop.count_type);
        if (count < 0) throw std::runtime_error("negative list count in binary PLY face element");
        std::vector<std::int64_t> values;
        if (static_cast<int>(pi) == index_prop) values.reserve(static_cast<std::size_t>(count));
        for (std::int64_t k = 0; k < count; ++k) {
          const std::int64_t index = read_binary_scalar_as_int64(in, prop.item_type);
          if (static_cast<int>(pi) == index_prop) values.push_back(index);
        }
        if (static_cast<int>(pi) == index_prop) {
          face.reserve(values.size());
          for (std::int64_t index : values) {
            face.push_back(checked_vertex_index(index, mesh.vertices.size(), "binary face " + std::to_string(face_id)));
          }
        }
      } else {
        skip_binary_scalar(in, prop.scalar_type);
      }
    }
    triangulate_face(mesh, face, "binary face " + std::to_string(face_id));
  }
}

void skip_ascii_element(std::istream& in, const PlyElement& element) {
  std::string line;
  for (std::size_t i = 0; i < element.count; ++i) {
    if (!std::getline(in, line)) throw std::runtime_error("unexpected end of ASCII PLY while skipping element " + element.name);
  }
}

void skip_binary_element(std::istream& in, const PlyElement& element) {
  for (std::size_t i = 0; i < element.count; ++i) {
    for (const PlyProperty& prop : element.properties) {
      if (prop.is_list) {
        const std::int64_t count = read_binary_scalar_as_int64(in, prop.count_type);
        if (count < 0) throw std::runtime_error("negative list count while skipping binary PLY element " + element.name);
        for (std::int64_t k = 0; k < count; ++k) skip_binary_scalar(in, prop.item_type);
      } else {
        skip_binary_scalar(in, prop.scalar_type);
      }
    }
  }
}

}  // namespace

Mesh load_ply_mesh(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open PLY file: " + path);
  }

  const PlyHeader header = parse_header(in);
  const PlyElement* vertex_element = find_element(header, "vertex");
  const PlyElement* face_element = find_element(header, "face");
  if (!vertex_element) throw std::runtime_error("PLY file has no vertex element: " + path);
  if (!face_element) throw std::runtime_error("PLY file has no face element: " + path);

  Mesh mesh;
  mesh.name = basename(path);
  mesh.vertices.reserve(vertex_element->count);
  mesh.triangles.reserve(face_element->count);

  for (const PlyElement& element : header.elements) {
    if (header.format == PlyFormat::Ascii) {
      if (element.name == "vertex") {
        parse_ascii_vertices(in, element, mesh);
      } else if (element.name == "face") {
        parse_ascii_faces(in, element, mesh);
      } else {
        skip_ascii_element(in, element);
      }
    } else {
      if (element.name == "vertex") {
        parse_binary_vertices(in, element, mesh);
      } else if (element.name == "face") {
        parse_binary_faces(in, element, mesh);
      } else {
        skip_binary_element(in, element);
      }
    }
  }

  require_mesh_valid(mesh);
  return mesh;
}

}  // namespace n2wos
