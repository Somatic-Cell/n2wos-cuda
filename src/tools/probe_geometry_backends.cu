#include "n2wos/closest_point.hpp"
#include "n2wos/cuda_bvh.hpp"
#ifdef N2WOS_HAS_CUBQL
#include "n2wos/cubql_bvh.hpp"
#endif
#include "n2wos/json.hpp"
#include "n2wos/mesh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

#define N2WOS_CUDA_CHECK(expr) \
  do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                               ": " + cudaGetErrorString(err__)); \
    } \
  } while (false)

struct Options {
  std::string mesh = "procedural_bumpy_sphere";
  std::string mesh_path;
  int normalize = 1;
  int bumpy_stacks = 128;
  int bumpy_slices = 256;
  float bumpy_amplitude = 0.15f;
  int leaf_size = 8;
  std::vector<int> cubql_leaf_sizes = {8};
  bool cubql_leaf_sizes_explicit = false;
  std::vector<std::string> cubql_build_methods = {"spatial_median", "radix", "sah", "elh"};
  int queries = 262144;
  int validate = 2048;
  int repeat = 10;
  int block_size = 128;
  unsigned int seed = 12345;
  float query_extent = 1.35f;
  std::string query_mode = "uniform_box";
  float shell_min = 1.0e-4f;
  float shell_max = 2.5e-1f;
  float plane_z = 0.0f;
  int run_custom = 1;
  int run_cubql = 1;
  std::string output = "results/probe_geometry_backends.json";
};

struct ValidationStats {
  int checked = 0;
  int bad_distance_count = 0;
  float max_abs_distance_error = 0.0f;
  float rms_distance_error = 0.0f;
  float max_closest_point_error = 0.0f;
  float rms_closest_point_error = 0.0f;
  float max_backend_distance_delta = 0.0f;
  int triangle_id_disagreement_count = 0;
};

struct BackendRun {
  std::string name;
  std::string builder;
  std::string error;
  bool enabled = false;
  bool built = false;
  bool device_resident_query_api = true;
  bool host_transfer_in_timing = false;
  bool production_candidate = false;
  std::size_t triangles = 0;
  std::size_t nodes = 0;
  std::size_t prim_ids = 0;
  int leaf_size = 0;
  float build_milliseconds = 0.0f;
  std::vector<float> kernel_ms;
  n2wos::CudaBvhQueryResult result;
  ValidationStats validation;
};

std::string trim(const std::string& s) {
  const std::size_t first = s.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return std::string();
  const std::size_t last = s.find_last_not_of(" \t\n\r");
  return s.substr(first, last - first + 1);
}

std::string canonical_build_method(std::string method) {
  method = trim(method);
  if (method == "sm" || method == "spatial-median") return "spatial_median";
  if (method == "surface_area_heuristic") return "sah";
  if (method == "morton" || method == "radix_morton") return "radix";
  if (method == "rebin" || method == "robust_radix" || method == "modified_radix" ||
      method == "rebin_radix" || method == "rebin-radix") {
    throw std::runtime_error("cuBQL build method rebin_radix is disabled: current cuBQL main declares rebinRadixBuilder but does not provide a linkable implementation header");
  }
  return method;
}

std::vector<std::string> parse_string_list(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> out;
  for (const std::string& item : parse_string_list(text)) {
    out.push_back(std::stoi(item));
  }
  return out;
}

void validate_build_method_name(const std::string& method) {
  if (method != "spatial_median" && method != "sah" && method != "elh" && method != "radix") {
    throw std::runtime_error("cuBQL build method must be one of: spatial_median, sah, elh, radix; rebin_radix is disabled for current cuBQL main");
  }
}

std::string canonical_query_mode(std::string mode) {
  mode = trim(mode);
  if (mode == "plane_slice") return "interior_slice";
  if (mode == "near_surface_shell") return "near_boundary_shell";
  if (mode == "mixed_wos_like") return "wos_like_prefix";
  return mode;
}

void validate_query_mode_name(const std::string& mode) {
  if (mode != "uniform_box" && mode != "interior_slice" && mode != "surface_exact" &&
      mode != "near_boundary_shell" && mode != "wos_like_prefix") {
    throw std::runtime_error("--query-mode must be uniform_box, interior_slice, surface_exact, near_boundary_shell, or wos_like_prefix");
  }
}

void print_usage(std::ostream& out, const char* argv0) {
  out << "Usage: " << argv0 << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --mesh procedural_bumpy_sphere|obj\n"
      << "  --mesh-path <path>                 Required when --mesh obj\n"
      << "  --normalize 0|1                    Normalize OBJ mesh to centered unit radius [1]\n"
      << "  --bumpy-stacks <int>               Procedural sphere stacks [128]\n"
      << "  --bumpy-slices <int>               Procedural sphere slices [256]\n"
      << "  --bumpy-amplitude <float>          Procedural bump amplitude [0.15]\n"
      << "  --leaf-size <int>                  In-tree custom BVH leaf triangle count [8]\n"
      << "  --cubql-leaf-sizes <csv>           cuBQL leaf thresholds, e.g. 1,4,8,16 [8]\n"
      << "  --queries <int>                    Number of device-resident query points [262144]\n"
      << "  --validate <int>                   Number of points checked by CPU brute force [2048]\n"
      << "  --repeat <int>                     Repeated timed device-query launches [10]\n"
      << "  --block-size <int>                 CUDA block size [128]\n"
      << "  --seed <int>                       Host query generation seed [12345]\n"
      << "  --query-extent <float>             Uniform cube/slice half-width [1.35]\n"
      << "  --query-mode <mode>                uniform_box|interior_slice|surface_exact|near_boundary_shell|wos_like_prefix [uniform_box]\n"
      << "  --shell-min <float>                near_boundary_shell minimum offset [1e-4]\n"
      << "  --shell-max <float>                near_boundary_shell maximum offset [0.25]\n"
      << "  --plane-z <float>                  z coordinate for interior_slice [0]\n"
      << "  --cubql-build-method <name>        Single method; kept for backward compatibility\n"
      << "  --cubql-build-methods <csv>        spatial_median,radix,sah,elh or sweep/all [spatial_median,radix,sah,elh]; rebin_radix disabled\n"
      << "  --run-custom 0|1                   Run in-tree median BVH backend [1]\n"
      << "  --run-cubql 0|1                    Run cuBQL backends if compiled [1]\n"
      << "  --output <path>                    JSON output [results/probe_geometry_backends.json]\n"
      << "  --help\n";
}

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[i]);
  }
  return argv[++i];
}

Options parse_options(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    } else if (arg == "--mesh") {
      opt.mesh = require_value(i, argc, argv);
    } else if (arg == "--mesh-path") {
      opt.mesh_path = require_value(i, argc, argv);
    } else if (arg == "--normalize") {
      opt.normalize = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--bumpy-stacks") {
      opt.bumpy_stacks = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--bumpy-slices") {
      opt.bumpy_slices = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--bumpy-amplitude") {
      opt.bumpy_amplitude = std::stof(require_value(i, argc, argv));
    } else if (arg == "--leaf-size") {
      opt.leaf_size = std::stoi(require_value(i, argc, argv));
      if (!opt.cubql_leaf_sizes_explicit) opt.cubql_leaf_sizes = {opt.leaf_size};
    } else if (arg == "--cubql-leaf-sizes") {
      opt.cubql_leaf_sizes = parse_int_list(require_value(i, argc, argv));
      opt.cubql_leaf_sizes_explicit = true;
    } else if (arg == "--queries") {
      opt.queries = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--validate") {
      opt.validate = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--repeat") {
      opt.repeat = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--block-size") {
      opt.block_size = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--seed") {
      opt.seed = static_cast<unsigned int>(std::stoul(require_value(i, argc, argv)));
    } else if (arg == "--query-extent") {
      opt.query_extent = std::stof(require_value(i, argc, argv));
    } else if (arg == "--query-mode") {
      opt.query_mode = canonical_query_mode(require_value(i, argc, argv));
    } else if (arg == "--shell-min") {
      opt.shell_min = std::stof(require_value(i, argc, argv));
    } else if (arg == "--shell-max") {
      opt.shell_max = std::stof(require_value(i, argc, argv));
    } else if (arg == "--plane-z") {
      opt.plane_z = std::stof(require_value(i, argc, argv));
    } else if (arg == "--cubql-build-method") {
      opt.cubql_build_methods = {canonical_build_method(require_value(i, argc, argv))};
    } else if (arg == "--cubql-build-methods") {
      const std::string value = trim(require_value(i, argc, argv));
      if (value == "sweep" || value == "all") {
        opt.cubql_build_methods = {"spatial_median", "sah", "elh", "radix"};
      } else {
        opt.cubql_build_methods.clear();
        for (const std::string& method : parse_string_list(value)) {
          opt.cubql_build_methods.push_back(canonical_build_method(method));
        }
      }
    } else if (arg == "--run-custom") {
      opt.run_custom = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--run-cubql") {
      opt.run_cubql = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--output") {
      opt.output = require_value(i, argc, argv);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }

  if (opt.mesh != "procedural_bumpy_sphere" && opt.mesh != "obj") {
    throw std::runtime_error("--mesh must be procedural_bumpy_sphere or obj");
  }
  if (opt.mesh == "obj" && opt.mesh_path.empty()) {
    throw std::runtime_error("--mesh-path is required when --mesh obj");
  }
  if (opt.cubql_build_methods.empty()) {
    throw std::runtime_error("--cubql-build-methods cannot be empty");
  }
  for (const std::string& method : opt.cubql_build_methods) validate_build_method_name(method);
  if (opt.cubql_leaf_sizes.empty()) {
    throw std::runtime_error("--cubql-leaf-sizes cannot be empty");
  }
  for (int leaf : opt.cubql_leaf_sizes) {
    if (leaf <= 0) throw std::runtime_error("--cubql-leaf-sizes entries must be positive");
  }
  validate_query_mode_name(opt.query_mode);
  if (opt.queries <= 0) throw std::runtime_error("--queries must be positive");
  if (opt.validate < 0) throw std::runtime_error("--validate must be non-negative");
  if (opt.repeat <= 0) throw std::runtime_error("--repeat must be positive");
  if (opt.leaf_size <= 0) throw std::runtime_error("--leaf-size must be positive");
  if (!(opt.query_extent > 0.0f)) throw std::runtime_error("--query-extent must be positive");
  if (!(opt.shell_min >= 0.0f) || !(opt.shell_max >= opt.shell_min)) {
    throw std::runtime_error("require 0 <= --shell-min <= --shell-max");
  }
  return opt;
}

n2wos::DeviceVec3 to_device_vec3(const n2wos::Vec3f& p) {
  return n2wos::DeviceVec3{p.x, p.y, p.z};
}

n2wos::Vec3f from_device_vec3(const n2wos::DeviceVec3& p) {
  return n2wos::Vec3f{p.x, p.y, p.z};
}

n2wos::Vec3f normalize_or_fallback(const n2wos::Vec3f& v) {
  const float len = n2wos::length(v);
  if (!(len > 0.0f)) return n2wos::Vec3f{1.0f, 0.0f, 0.0f};
  return v / len;
}

n2wos::Vec3f sample_point_on_triangle(const n2wos::Mesh& mesh,
                                      const n2wos::Triangle& tri,
                                      std::mt19937& rng) {
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  const float su = std::sqrt(unit(rng));
  const float v = unit(rng);
  const float a = 1.0f - su;
  const float b = su * (1.0f - v);
  const float c = su * v;
  return mesh.vertices[tri.v0] * a + mesh.vertices[tri.v1] * b + mesh.vertices[tri.v2] * c;
}

float triangle_area(const n2wos::Mesh& mesh, const n2wos::Triangle& tri) {
  const n2wos::Vec3f a = mesh.vertices[tri.v0];
  const n2wos::Vec3f b = mesh.vertices[tri.v1];
  const n2wos::Vec3f c = mesh.vertices[tri.v2];
  return 0.5f * n2wos::length(n2wos::cross(b - a, c - a));
}

std::vector<n2wos::Vec3f> make_uniform_box_points(int count, unsigned int seed, float extent) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-extent, extent);
  std::vector<n2wos::Vec3f> points(static_cast<std::size_t>(count));
  for (n2wos::Vec3f& p : points) {
    p = {uniform(rng), uniform(rng), uniform(rng)};
  }
  return points;
}

std::vector<n2wos::Vec3f> make_plane_slice_points(int count, unsigned int seed, float extent, float z) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-extent, extent);
  std::vector<n2wos::Vec3f> points(static_cast<std::size_t>(count));
  for (n2wos::Vec3f& p : points) {
    p = {uniform(rng), uniform(rng), z};
  }
  return points;
}

std::vector<n2wos::Vec3f> make_surface_shell_points(const n2wos::Mesh& mesh,
                                                    int count,
                                                    unsigned int seed,
                                                    float shell_min,
                                                    float shell_max,
                                                    bool zero_offset) {
  n2wos::require_mesh_valid(mesh);
  std::mt19937 rng(seed);
  std::vector<double> triangle_weights;
  triangle_weights.reserve(mesh.triangles.size());
  for (const n2wos::Triangle& tri : mesh.triangles) {
    triangle_weights.push_back(std::max<double>(triangle_area(mesh, tri), 1.0e-30));
  }
  std::discrete_distribution<std::size_t> tri_pick(triangle_weights.begin(), triangle_weights.end());
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::vector<n2wos::Vec3f> points(static_cast<std::size_t>(count));

  const float min_positive = std::max(shell_min, 1.0e-8f);
  const bool log_sample = shell_max > min_positive;
  for (n2wos::Vec3f& p : points) {
    const n2wos::Triangle& tri = mesh.triangles[tri_pick(rng)];
    const n2wos::Vec3f a = mesh.vertices[tri.v0];
    const n2wos::Vec3f b = mesh.vertices[tri.v1];
    const n2wos::Vec3f c = mesh.vertices[tri.v2];
    const n2wos::Vec3f q = sample_point_on_triangle(mesh, tri, rng);
    const n2wos::Vec3f n = normalize_or_fallback(n2wos::cross(b - a, c - a));
    float offset = 0.0f;
    if (!zero_offset) {
      if (log_sample) {
        const float t = unit(rng);
        offset = min_positive * std::exp(std::log(shell_max / min_positive) * t);
      } else {
        offset = shell_max;
      }
      if (unit(rng) < 0.5f) offset = -offset;
    }
    p = q + n * offset;
  }
  return points;
}

std::vector<n2wos::Vec3f> make_mixed_wos_like_points(const n2wos::Mesh& mesh,
                                                     int count,
                                                     unsigned int seed,
                                                     float extent,
                                                     float shell_min,
                                                     float shell_max,
                                                     float plane_z) {
  std::vector<n2wos::Vec3f> shell = make_surface_shell_points(mesh, count, seed, shell_min, shell_max, false);
  std::vector<n2wos::Vec3f> slice = make_plane_slice_points(count, seed ^ 0x9e3779b9u, extent, plane_z);
  for (int i = 0; i < count; ++i) {
    if ((i % 4) == 0) shell[static_cast<std::size_t>(i)] = slice[static_cast<std::size_t>(i)];
  }
  return shell;
}

std::vector<n2wos::Vec3f> make_query_points(const Options& opt, const n2wos::Mesh& mesh) {
  if (opt.query_mode == "uniform_box") {
    return make_uniform_box_points(opt.queries, opt.seed, opt.query_extent);
  }
  if (opt.query_mode == "interior_slice") {
    return make_plane_slice_points(opt.queries, opt.seed, opt.query_extent, opt.plane_z);
  }
  if (opt.query_mode == "surface_exact") {
    return make_surface_shell_points(mesh, opt.queries, opt.seed, 0.0f, 0.0f, true);
  }
  if (opt.query_mode == "near_boundary_shell") {
    return make_surface_shell_points(mesh, opt.queries, opt.seed, opt.shell_min, opt.shell_max, false);
  }
  if (opt.query_mode == "wos_like_prefix") {
    return make_mixed_wos_like_points(mesh, opt.queries, opt.seed, opt.query_extent, opt.shell_min, opt.shell_max, opt.plane_z);
  }
  throw std::runtime_error("unsupported query mode: " + opt.query_mode);
}

float median(std::vector<float> values) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5f * (values[mid - 1] + values[mid]);
}

float mean(const std::vector<float>& values) {
  if (values.empty()) return 0.0f;
  const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
  return sum / static_cast<float>(values.size());
}

double median_us_per_query(const BackendRun& run, int queries) {
  if (run.kernel_ms.empty() || queries <= 0) return std::numeric_limits<double>::infinity();
  return static_cast<double>(median(run.kernel_ms)) * 1000.0 / static_cast<double>(queries);
}

bool backend_passed(const BackendRun& run) {
  return run.enabled && run.built && run.device_resident_query_api && !run.host_transfer_in_timing &&
         run.validation.bad_distance_count == 0 && run.result.overflow_count == 0 && !run.kernel_ms.empty();
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
#if defined(_WIN32)
  std::tm tm{};
  gmtime_s(&tm, &t);
#else
  std::tm tm{};
  gmtime_r(&t, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::string join_argv(int argc, char** argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i) out << ' ';
    out << argv[i];
  }
  return out.str();
}

struct DeviceQueryStorage {
  n2wos::DeviceVec3* d_points = nullptr;
  float* d_distance2 = nullptr;
  n2wos::DeviceVec3* d_closest = nullptr;
  int* d_triangle_id = nullptr;
  int* d_overflow = nullptr;
  std::size_t count = 0;

  explicit DeviceQueryStorage(const std::vector<n2wos::Vec3f>& points) : count(points.size()) {
    std::vector<n2wos::DeviceVec3> h_points(count);
    for (std::size_t i = 0; i < count; ++i) h_points[i] = to_device_vec3(points[i]);
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points), count * sizeof(n2wos::DeviceVec3)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distance2), count * sizeof(float)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_closest), count * sizeof(n2wos::DeviceVec3)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_id), count * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_overflow), count * sizeof(int)));
    N2WOS_CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), count * sizeof(n2wos::DeviceVec3), cudaMemcpyHostToDevice));
  }

  ~DeviceQueryStorage() {
    if (d_points) cudaFree(d_points);
    if (d_distance2) cudaFree(d_distance2);
    if (d_closest) cudaFree(d_closest);
    if (d_triangle_id) cudaFree(d_triangle_id);
    if (d_overflow) cudaFree(d_overflow);
  }

  DeviceQueryStorage(const DeviceQueryStorage&) = delete;
  DeviceQueryStorage& operator=(const DeviceQueryStorage&) = delete;
};

template <typename Backend>
BackendRun run_backend_device(const std::string& name,
                              const std::string& builder,
                              const Backend& backend,
                              DeviceQueryStorage& storage,
                              int repeat,
                              int block_size,
                              bool production_candidate,
                              std::size_t triangles,
                              std::size_t nodes,
                              std::size_t prim_ids,
                              int leaf_size,
                              float build_milliseconds) {
  BackendRun run;
  run.name = name;
  run.builder = builder;
  run.enabled = true;
  run.built = true;
  run.production_candidate = production_candidate;
  run.triangles = triangles;
  run.nodes = nodes;
  run.prim_ids = prim_ids;
  run.leaf_size = leaf_size;
  run.build_milliseconds = build_milliseconds;
  run.kernel_ms.reserve(static_cast<std::size_t>(repeat));

  const int query_count = static_cast<int>(storage.count);

  // Warmup. Inputs and outputs remain on device.
  backend.query_device(storage.d_points,
                       query_count,
                       storage.d_distance2,
                       storage.d_closest,
                       storage.d_triangle_id,
                       storage.d_overflow,
                       block_size,
                       0);
  N2WOS_CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < repeat; ++r) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    N2WOS_CUDA_CHECK(cudaEventCreate(&start));
    N2WOS_CUDA_CHECK(cudaEventCreate(&stop));
    N2WOS_CUDA_CHECK(cudaEventRecord(start));
    backend.query_device(storage.d_points,
                         query_count,
                         storage.d_distance2,
                         storage.d_closest,
                         storage.d_triangle_id,
                         storage.d_overflow,
                         block_size,
                         0);
    N2WOS_CUDA_CHECK(cudaEventRecord(stop));
    N2WOS_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    N2WOS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    N2WOS_CUDA_CHECK(cudaEventDestroy(start));
    N2WOS_CUDA_CHECK(cudaEventDestroy(stop));
    run.kernel_ms.push_back(ms);
    std::cerr << name << " repeat " << (r + 1) << "/" << repeat << ": kernel_ms=" << ms << "\n";
  }

  run.result.distance2.resize(storage.count);
  run.result.closest.resize(storage.count);
  run.result.triangle_id.resize(storage.count);
  std::vector<n2wos::DeviceVec3> h_closest(storage.count);
  std::vector<int> h_overflow(storage.count);
  N2WOS_CUDA_CHECK(cudaMemcpy(run.result.distance2.data(), storage.d_distance2, storage.count * sizeof(float), cudaMemcpyDeviceToHost));
  N2WOS_CUDA_CHECK(cudaMemcpy(h_closest.data(), storage.d_closest, storage.count * sizeof(n2wos::DeviceVec3), cudaMemcpyDeviceToHost));
  N2WOS_CUDA_CHECK(cudaMemcpy(run.result.triangle_id.data(), storage.d_triangle_id, storage.count * sizeof(int), cudaMemcpyDeviceToHost));
  N2WOS_CUDA_CHECK(cudaMemcpy(h_overflow.data(), storage.d_overflow, storage.count * sizeof(int), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < storage.count; ++i) {
    run.result.closest[i] = from_device_vec3(h_closest[i]);
    run.result.overflow_count += h_overflow[i] != 0 ? 1 : 0;
  }
  run.result.kernel_milliseconds = median(run.kernel_ms);
  return run;
}

ValidationStats validate_against_cpu(const n2wos::Mesh& mesh,
                                     const std::vector<n2wos::Vec3f>& points,
                                     const n2wos::CudaBvhQueryResult& result,
                                     int requested_count,
                                     const n2wos::CudaBvhQueryResult* comparison = nullptr) {
  ValidationStats stats;
  const int count = std::min<int>(requested_count, static_cast<int>(points.size()));
  stats.checked = count;
  double sum_d2 = 0.0;
  double sum_cp2 = 0.0;

  for (int i = 0; i < count; ++i) {
    const n2wos::ClosestPointResult cpu = n2wos::closest_point_bruteforce(mesh, points[static_cast<std::size_t>(i)]);
    const float cpu_d = std::sqrt(std::max(cpu.distance2, 0.0f));
    const float backend_d = std::sqrt(std::max(result.distance2[static_cast<std::size_t>(i)], 0.0f));
    const float abs_distance_error = std::fabs(cpu_d - backend_d);
    const float closest_error = n2wos::length(cpu.closest - result.closest[static_cast<std::size_t>(i)]);
    stats.max_abs_distance_error = std::max(stats.max_abs_distance_error, abs_distance_error);
    stats.max_closest_point_error = std::max(stats.max_closest_point_error, closest_error);
    sum_d2 += static_cast<double>(abs_distance_error) * static_cast<double>(abs_distance_error);
    sum_cp2 += static_cast<double>(closest_error) * static_cast<double>(closest_error);
    if (abs_distance_error > 2.0e-4f) {
      ++stats.bad_distance_count;
    }
    if (cpu.triangle_id != result.triangle_id[static_cast<std::size_t>(i)]) {
      ++stats.triangle_id_disagreement_count;
    }
    if (comparison) {
      const float other_d = std::sqrt(std::max(comparison->distance2[static_cast<std::size_t>(i)], 0.0f));
      stats.max_backend_distance_delta = std::max(stats.max_backend_distance_delta, std::fabs(other_d - backend_d));
    }
  }

  if (count > 0) {
    stats.rms_distance_error = static_cast<float>(std::sqrt(sum_d2 / static_cast<double>(count)));
    stats.rms_closest_point_error = static_cast<float>(std::sqrt(sum_cp2 / static_cast<double>(count)));
  }
  return stats;
}

void write_timing_json(std::ostream& out, const std::vector<float>& timings_ms, int queries, int indent_spaces) {
  const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
  const float med_ms = median(timings_ms);
  const float mean_ms = mean(timings_ms);
  const double median_us = timings_ms.empty() ? 0.0 : static_cast<double>(med_ms) * 1000.0 / static_cast<double>(queries);
  const double mean_us = timings_ms.empty() ? 0.0 : static_cast<double>(mean_ms) * 1000.0 / static_cast<double>(queries);
  const double median_mqueries_per_second = timings_ms.empty() ? 0.0 : static_cast<double>(queries) / (static_cast<double>(med_ms) * 1000.0);
  out << indent << "\"kernel_ms\": [";
  for (std::size_t i = 0; i < timings_ms.size(); ++i) {
    if (i) out << ", ";
    out << timings_ms[i];
  }
  out << "],\n";
  out << indent << "\"kernel_ms_median\": " << med_ms << ",\n";
  out << indent << "\"kernel_ms_mean\": " << mean_ms << ",\n";
  out << indent << "\"median_us_per_query\": " << median_us << ",\n";
  out << indent << "\"mean_us_per_query\": " << mean_us << ",\n";
  out << indent << "\"median_mqueries_per_second\": " << median_mqueries_per_second << "\n";
}

void write_validation_json(std::ostream& out, const ValidationStats& v, int indent_spaces) {
  const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
  out << indent << "\"checked\": " << v.checked << ",\n";
  out << indent << "\"bad_distance_count_threshold_2e_4\": " << v.bad_distance_count << ",\n";
  out << indent << "\"max_abs_distance_error\": " << v.max_abs_distance_error << ",\n";
  out << indent << "\"rms_distance_error\": " << v.rms_distance_error << ",\n";
  out << indent << "\"max_closest_point_error\": " << v.max_closest_point_error << ",\n";
  out << indent << "\"rms_closest_point_error\": " << v.rms_closest_point_error << ",\n";
  out << indent << "\"max_backend_distance_delta\": " << v.max_backend_distance_delta << ",\n";
  out << indent << "\"triangle_id_disagreement_count\": " << v.triangle_id_disagreement_count << "\n";
}

void write_string_array_json(std::ostream& out, const std::vector<std::string>& values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ", ";
    out << n2wos::json_quote(values[i]);
  }
  out << "]";
}

void write_int_array_json(std::ostream& out, const std::vector<int>& values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ", ";
    out << values[i];
  }
  out << "]";
}

void write_backend_json(std::ostream& out, const BackendRun& run, int queries) {
  out << "    " << n2wos::json_quote(run.name) << ": {\n";
  out << "      \"enabled\": " << (run.enabled ? "true" : "false") << ",\n";
  out << "      \"built\": " << (run.built ? "true" : "false") << ",\n";
  out << "      \"builder\": " << n2wos::json_quote(run.builder) << ",\n";
  if (!run.error.empty()) out << "      \"error\": " << n2wos::json_quote(run.error) << ",\n";
  out << "      \"production_candidate\": " << (run.production_candidate ? "true" : "false") << ",\n";
  out << "      \"device_resident_query_api\": " << (run.device_resident_query_api ? "true" : "false") << ",\n";
  out << "      \"host_transfer_in_timing\": " << (run.host_transfer_in_timing ? "true" : "false") << ",\n";
  out << "      \"bvh_stats\": {\n";
  out << "        \"triangles\": " << run.triangles << ",\n";
  out << "        \"nodes\": " << run.nodes << ",\n";
  out << "        \"prim_ids\": " << run.prim_ids << ",\n";
  out << "        \"leaf_size\": " << run.leaf_size << ",\n";
  out << "        \"build_milliseconds\": " << run.build_milliseconds << "\n";
  out << "      },\n";
  out << "      \"timing\": {\n";
  write_timing_json(out, run.kernel_ms, queries, 8);
  out << "      },\n";
  out << "      \"validation\": {\n";
  write_validation_json(out, run.validation, 8);
  out << "      },\n";
  out << "      \"validation_passed\": " << (backend_passed(run) ? "true" : "false") << ",\n";
  out << "      \"overflow_count\": " << run.result.overflow_count << "\n";
  out << "    }";
}

void write_disabled_backend_json(std::ostream& out, const std::string& name, const std::string& reason) {
  out << "    " << n2wos::json_quote(name) << ": {\n";
  out << "      \"enabled\": false,\n";
  out << "      \"built\": false,\n";
  out << "      \"reason\": " << n2wos::json_quote(reason) << ",\n";
  out << "      \"production_candidate\": false,\n";
  out << "      \"device_resident_query_api\": true,\n";
  out << "      \"host_transfer_in_timing\": false\n";
  out << "    }";
}

const BackendRun* fastest_passing(const std::vector<BackendRun>& runs, int queries, bool production_only) {
  const BackendRun* best = nullptr;
  double best_us = std::numeric_limits<double>::infinity();
  for (const BackendRun& run : runs) {
    if (!backend_passed(run)) continue;
    if (production_only && !run.production_candidate) continue;
    const double us = median_us_per_query(run, queries);
    if (us < best_us) {
      best_us = us;
      best = &run;
    }
  }
  return best;
}

void write_recommendation_json(std::ostream& out, const std::vector<BackendRun>& runs, int queries) {
  const BackendRun* best_prod = fastest_passing(runs, queries, true);
  const BackendRun* best_all = fastest_passing(runs, queries, false);
  out << "  \"recommendation\": {\n";
  out << "    \"selection_policy\": \"choose the fastest validated device-resident production candidate for the tested mesh/query mode; custom_cuda_bvh is excluded until separately revalidated\",\n";
  out << "    \"preferred_cubql_sweep_order\": [\"sah\", \"rebin_radix\", \"radix\", \"spatial_median\", \"elh_optional\"],\n";
  out << "    \"default_before_measurement\": \"sah for static meshes when build cost is amortized; rebin_radix as robust fast-builder fallback; do not hard-code spatial_median based only on patch 0002\",\n";
  if (best_prod) {
    out << "    \"selected_production_candidate\": " << n2wos::json_quote(best_prod->name) << ",\n";
    out << "    \"selected_median_us_per_query\": " << median_us_per_query(*best_prod, queries) << ",\n";
  } else {
    out << "    \"selected_production_candidate\": null,\n";
    out << "    \"selected_median_us_per_query\": null,\n";
  }
  if (best_all) {
    out << "    \"fastest_validated_overall\": " << n2wos::json_quote(best_all->name) << ",\n";
    out << "    \"fastest_validated_overall_median_us_per_query\": " << median_us_per_query(*best_all, queries) << ",\n";
  } else {
    out << "    \"fastest_validated_overall\": null,\n";
    out << "    \"fastest_validated_overall_median_us_per_query\": null,\n";
  }
  out << "    \"custom_cuda_bvh_excluded_from_production_selection\": true\n";
  out << "  },\n";
}

void write_json(const std::string& path,
                const Options& opt,
                const n2wos::Mesh& mesh,
                const n2wos::NormalizeTransform* transform,
                const n2wos::Aabb& bounds,
                std::size_t degenerate_triangles,
                const std::vector<BackendRun>& runs,
                const std::string& command_line) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output JSON: " + path);
  }

  out << std::setprecision(9);
  out << "{\n";
  out << "  \"schema\": \"n2wos_geometry_backend_probe_v2\",\n";
  out << "  \"patch\": \"0002a-geometry-backend-stress-and-cubql-builder-sweep\",\n";
  out << "  \"generated_at_utc\": " << n2wos::json_quote(utc_timestamp()) << ",\n";
  out << "  \"command_line\": " << n2wos::json_quote(command_line) << ",\n";
  out << "  \"cuda\": {\n";
  out << "    \"runtime\": " << n2wos::json_quote(n2wos::cuda_runtime_version_string()) << ",\n";
  out << "    \"device\": " << n2wos::json_quote(n2wos::cuda_device_summary()) << "\n";
  out << "  },\n";
  out << "  \"backend_policy\": {\n";
  out << "    \"production_requires_gpu_resident_geometry_query\": true,\n";
  out << "    \"production_forbids_per_step_host_transfer\": true,\n";
  out << "    \"production_forbids_fcpw_public_host_vector_loop\": true,\n";
  out << "    \"probe_timing_scope\": \"device_query_kernel_only_excludes_query_upload_result_readback; build time is separately reported\",\n";
  out << "    \"custom_cuda_bvh_role\": \"comparison_only_until_replaced_or_revalidated\",\n";
  out << "    \"cubql_cuda_role\": \"production_candidate_when_enabled_and_validates\"\n";
  out << "  },\n";
  out << "  \"options\": {\n";
  out << "    \"mesh\": " << n2wos::json_quote(opt.mesh) << ",\n";
  out << "    \"mesh_path\": " << n2wos::json_quote(opt.mesh_path) << ",\n";
  out << "    \"normalize\": " << opt.normalize << ",\n";
  out << "    \"bumpy_stacks\": " << opt.bumpy_stacks << ",\n";
  out << "    \"bumpy_slices\": " << opt.bumpy_slices << ",\n";
  out << "    \"bumpy_amplitude\": " << opt.bumpy_amplitude << ",\n";
  out << "    \"leaf_size\": " << opt.leaf_size << ",\n";
  out << "    \"cubql_leaf_sizes\": "; write_int_array_json(out, opt.cubql_leaf_sizes); out << ",\n";
  out << "    \"queries\": " << opt.queries << ",\n";
  out << "    \"validate\": " << opt.validate << ",\n";
  out << "    \"repeat\": " << opt.repeat << ",\n";
  out << "    \"block_size\": " << opt.block_size << ",\n";
  out << "    \"seed\": " << opt.seed << ",\n";
  out << "    \"query_extent\": " << opt.query_extent << ",\n";
  out << "    \"query_mode\": " << n2wos::json_quote(opt.query_mode) << ",\n";
  out << "    \"shell_min\": " << opt.shell_min << ",\n";
  out << "    \"shell_max\": " << opt.shell_max << ",\n";
  out << "    \"plane_z\": " << opt.plane_z << ",\n";
  out << "    \"cubql_build_methods\": "; write_string_array_json(out, opt.cubql_build_methods); out << "\n";
  out << "  },\n";
  out << "  \"mesh_stats\": {\n";
  out << "    \"name\": " << n2wos::json_quote(mesh.name) << ",\n";
  out << "    \"vertices\": " << mesh.vertices.size() << ",\n";
  out << "    \"triangles\": " << mesh.triangles.size() << ",\n";
  out << "    \"degenerate_triangles\": " << degenerate_triangles << ",\n";
  out << "    \"bounds_min\": [" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << "],\n";
  out << "    \"bounds_max\": [" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << "]";
  if (transform) {
    out << ",\n    \"normalization\": {\n";
    out << "      \"center\": [" << transform->center.x << ", " << transform->center.y << ", " << transform->center.z << "],\n";
    out << "      \"scale\": " << transform->scale << "\n";
    out << "    }\n";
  } else {
    out << "\n";
  }
  out << "  },\n";
  out << "  \"build_features\": {\n";
#ifdef N2WOS_HAS_CUBQL
  out << "    \"N2WOS_HAS_CUBQL\": true\n";
#else
  out << "    \"N2WOS_HAS_CUBQL\": false\n";
#endif
  out << "  },\n";
  write_recommendation_json(out, runs, opt.queries);
  out << "  \"backends\": {\n";
  bool emitted_backend = false;
  auto emit_backend_separator = [&]() {
    if (emitted_backend) out << ",\n";
    emitted_backend = true;
  };
  for (const BackendRun& run : runs) {
    emit_backend_separator();
    write_backend_json(out, run, opt.queries);
  }
#ifdef N2WOS_HAS_CUBQL
  const bool any_cubql = std::any_of(runs.begin(), runs.end(), [](const BackendRun& run) {
    return run.name.find("cubql_cuda") == 0;
  });
  if (!opt.run_cubql) {
    emit_backend_separator();
    write_disabled_backend_json(out, "cubql_cuda", "disabled by --run-cubql 0");
  } else if (!any_cubql) {
    emit_backend_separator();
    write_disabled_backend_json(out, "cubql_cuda", "not run");
  }
#else
  emit_backend_separator();
  write_disabled_backend_json(out, "cubql_cuda", "not compiled; configure with -DN2WOS_ENABLE_CUBQL=ON after fetching external/cuBQL");
#endif
  out << "\n  }\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_options(argc, argv);

    n2wos::Mesh mesh;
    n2wos::NormalizeTransform transform{};
    n2wos::NormalizeTransform* transform_ptr = nullptr;

    if (opt.mesh == "procedural_bumpy_sphere") {
      mesh = n2wos::make_procedural_bumpy_sphere(opt.bumpy_stacks, opt.bumpy_slices, opt.bumpy_amplitude);
      transform = n2wos::normalize_to_unit_radius(mesh);
      transform_ptr = &transform;
    } else {
      mesh = n2wos::load_obj_mesh(opt.mesh_path);
      if (opt.normalize != 0) {
        transform = n2wos::normalize_to_unit_radius(mesh);
        transform_ptr = &transform;
      }
    }

    const n2wos::Aabb bounds = n2wos::compute_bounds(mesh);
    const std::size_t degenerate_triangles = n2wos::count_degenerate_triangles(mesh);
    const std::vector<n2wos::Vec3f> query_points = make_query_points(opt, mesh);
    DeviceQueryStorage storage(query_points);

    std::cerr << "mesh: " << mesh.name << " vertices=" << mesh.vertices.size()
              << " triangles=" << mesh.triangles.size() << "\n";
    std::cerr << "query_mode=" << opt.query_mode << " query points uploaded once; timed backend queries do not include H2D/D2H transfers\n";

    std::vector<BackendRun> runs;

    if (opt.run_custom) {
      std::cerr << "building comparison custom CUDA BVH...\n";
      const auto t0 = std::chrono::high_resolution_clock::now();
      n2wos::CudaBvh custom(mesh, opt.leaf_size);
      const auto t1 = std::chrono::high_resolution_clock::now();
      const float build_ms = static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count());
      BackendRun custom_run = run_backend_device("custom_cuda_bvh_leaf" + std::to_string(custom.leaf_size()),
                                                 "in_tree_cpu_largest_axis_centroid_median_split",
                                                 custom,
                                                 storage,
                                                 opt.repeat,
                                                 opt.block_size,
                                                 false,
                                                 custom.triangle_count(),
                                                 custom.node_count(),
                                                 custom.index_count(),
                                                 custom.leaf_size(),
                                                 build_ms);
      custom_run.validation = validate_against_cpu(mesh, query_points, custom_run.result, opt.validate);
      runs.push_back(std::move(custom_run));
    }

#ifdef N2WOS_HAS_CUBQL
    if (opt.run_cubql) {
      const n2wos::CudaBvhQueryResult* comparison = nullptr;
      if (!runs.empty() && runs.front().name.find("custom_cuda_bvh") == 0) comparison = &runs.front().result;
      for (int leaf_size : opt.cubql_leaf_sizes) {
        for (const std::string& method : opt.cubql_build_methods) {
          const std::string backend_name = "cubql_cuda_" + method + "_leaf" + std::to_string(leaf_size);
          std::cerr << "building " << backend_name << "...\n";
          try {
            n2wos::CuBqlBvh cubql(mesh, leaf_size, method);
            BackendRun cubql_run = run_backend_device(backend_name,
                                                      std::string("cuBQL_") + cubql.build_method(),
                                                      cubql,
                                                      storage,
                                                      opt.repeat,
                                                      opt.block_size,
                                                      true,
                                                      cubql.triangle_count(),
                                                      cubql.node_count(),
                                                      cubql.prim_id_count(),
                                                      cubql.leaf_size(),
                                                      cubql.build_milliseconds());
            cubql_run.validation = validate_against_cpu(mesh, query_points, cubql_run.result, opt.validate, comparison);
            runs.push_back(std::move(cubql_run));
          } catch (const std::exception& e) {
            BackendRun failed;
            failed.name = backend_name;
            failed.builder = std::string("cuBQL_") + method;
            failed.enabled = true;
            failed.built = false;
            failed.production_candidate = false;
            failed.device_resident_query_api = true;
            failed.host_transfer_in_timing = false;
            failed.triangles = mesh.triangles.size();
            failed.leaf_size = leaf_size;
            failed.error = e.what();
            runs.push_back(std::move(failed));
            std::cerr << "backend failed: " << backend_name << ": " << e.what() << "\n";
          }
        }
      }
    }
#endif

    write_json(opt.output,
               opt,
               mesh,
               transform_ptr,
               bounds,
               degenerate_triangles,
               runs,
               join_argv(argc, argv));

    int total_bad = 0;
    int total_overflow = 0;
    int built_runs = 0;
    for (const BackendRun& run : runs) {
      if (!run.built) continue;
      ++built_runs;
      total_bad += run.validation.bad_distance_count;
      total_overflow += run.result.overflow_count;
      std::cout << run.name
                << " median_us_per_query=" << median_us_per_query(run, opt.queries)
                << " build_ms=" << run.build_milliseconds
                << " validation_max_abs_distance_error=" << run.validation.max_abs_distance_error
                << " overflow=" << run.result.overflow_count << "\n";
    }
    const BackendRun* selected = fastest_passing(runs, opt.queries, true);
    if (selected) {
      std::cout << "selected_production_candidate=" << selected->name
                << " median_us_per_query=" << median_us_per_query(*selected, opt.queries) << "\n";
    }
    std::cout << "wrote " << opt.output << "\n";
    if (built_runs == 0) return 1;
    return total_bad == 0 && total_overflow == 0 ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
