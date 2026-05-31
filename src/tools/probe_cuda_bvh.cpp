#include "n2wos/closest_point.hpp"
#include "n2wos/cuda_bvh.hpp"
#include "n2wos/json.hpp"
#include "n2wos/mesh.hpp"

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
#include <vector>

namespace {

struct Options {
  std::string mesh = "procedural_bumpy_sphere";
  std::string mesh_path;
  int normalize = 1;
  int bumpy_stacks = 128;
  int bumpy_slices = 256;
  float bumpy_amplitude = 0.15f;
  int leaf_size = 8;
  int queries = 262144;
  int validate = 512;
  int repeat = 10;
  int block_size = 128;
  unsigned int seed = 12345;
  float query_extent = 1.35f;
  std::string output = "results/probe_cuda_bvh.json";
};

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
      << "  --leaf-size <int>                  BVH leaf triangle count [8]\n"
      << "  --queries <int>                    Number of CUDA query points [262144]\n"
      << "  --validate <int>                   Number of points checked by CPU brute force [512]\n"
      << "  --repeat <int>                     Repeated timed kernel launches [10]\n"
      << "  --block-size <int>                 CUDA block size [128]\n"
      << "  --seed <int>                       Host query generation seed [12345]\n"
      << "  --query-extent <float>             Uniform query cube half-width [1.35]\n"
      << "  --output <path>                    JSON output [results/probe_cuda_bvh.json]\n"
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
    } else if (arg == "--output") {
      opt.output = require_value(i, argc, argv);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }

  if (opt.mesh != "procedural_bumpy_sphere" && opt.mesh != "obj" && opt.mesh != "ply") {
    throw std::runtime_error("--mesh must be procedural_bumpy_sphere, obj, or ply");
  }
  if ((opt.mesh == "obj" || opt.mesh == "ply") && opt.mesh_path.empty()) {
    throw std::runtime_error("--mesh-path is required when --mesh obj or ply");
  }
  if (opt.queries <= 0) throw std::runtime_error("--queries must be positive");
  if (opt.validate < 0) throw std::runtime_error("--validate must be non-negative");
  if (opt.repeat <= 0) throw std::runtime_error("--repeat must be positive");
  if (opt.leaf_size <= 0) throw std::runtime_error("--leaf-size must be positive");
  if (!(opt.query_extent > 0.0f)) throw std::runtime_error("--query-extent must be positive");
  return opt;
}

std::vector<n2wos::Vec3f> make_query_points(int count, unsigned int seed, float extent) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-extent, extent);
  std::vector<n2wos::Vec3f> points(static_cast<std::size_t>(count));
  for (n2wos::Vec3f& p : points) {
    p = {uniform(rng), uniform(rng), uniform(rng)};
  }
  return points;
}

struct ValidationStats {
  int checked = 0;
  int bad_distance_count = 0;
  float max_abs_distance_error = 0.0f;
  float rms_distance_error = 0.0f;
  float max_closest_point_error = 0.0f;
  float rms_closest_point_error = 0.0f;
};

ValidationStats validate_against_cpu(const n2wos::Mesh& mesh,
                                     const std::vector<n2wos::Vec3f>& points,
                                     const n2wos::CudaBvhQueryResult& cuda_result,
                                     int requested_count) {
  ValidationStats stats;
  const int count = std::min<int>(requested_count, static_cast<int>(points.size()));
  stats.checked = count;
  double sum_d2 = 0.0;
  double sum_cp2 = 0.0;

  for (int i = 0; i < count; ++i) {
    const n2wos::ClosestPointResult cpu = n2wos::closest_point_bruteforce(mesh, points[static_cast<std::size_t>(i)]);
    const float cpu_d = std::sqrt(std::max(cpu.distance2, 0.0f));
    const float gpu_d = std::sqrt(std::max(cuda_result.distance2[static_cast<std::size_t>(i)], 0.0f));
    const float abs_distance_error = std::fabs(cpu_d - gpu_d);
    const float closest_error = n2wos::length(cpu.closest - cuda_result.closest[static_cast<std::size_t>(i)]);
    stats.max_abs_distance_error = std::max(stats.max_abs_distance_error, abs_distance_error);
    stats.max_closest_point_error = std::max(stats.max_closest_point_error, closest_error);
    sum_d2 += static_cast<double>(abs_distance_error) * static_cast<double>(abs_distance_error);
    sum_cp2 += static_cast<double>(closest_error) * static_cast<double>(closest_error);
    if (abs_distance_error > 2.0e-4f) {
      ++stats.bad_distance_count;
    }
  }

  if (count > 0) {
    stats.rms_distance_error = static_cast<float>(std::sqrt(sum_d2 / static_cast<double>(count)));
    stats.rms_closest_point_error = static_cast<float>(std::sqrt(sum_cp2 / static_cast<double>(count)));
  }
  return stats;
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

void write_json(const std::string& path,
                const Options& opt,
                const n2wos::Mesh& mesh,
                const n2wos::NormalizeTransform* transform,
                const n2wos::CudaBvh& bvh,
                const n2wos::Aabb& bounds,
                std::size_t degenerate_triangles,
                const std::vector<float>& timings_ms,
                const n2wos::CudaBvhQueryResult& result,
                const ValidationStats& validation,
                const std::string& command_line) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output JSON: " + path);
  }

  const float med_ms = median(timings_ms);
  const float mean_ms = mean(timings_ms);
  const double median_us_per_query = static_cast<double>(med_ms) * 1000.0 / static_cast<double>(opt.queries);
  const double mean_us_per_query = static_cast<double>(mean_ms) * 1000.0 / static_cast<double>(opt.queries);
  const double median_mqueries_per_second = static_cast<double>(opt.queries) / (static_cast<double>(med_ms) * 1000.0);

  out << std::setprecision(9);
  out << "{\n";
  out << "  \"schema\": \"n2wos_cuda_probe_bvh_v1\",\n";
  out << "  \"patch\": \"0001-bootstrap-cuda-resident-geometry\",\n";
  out << "  \"generated_at_utc\": " << n2wos::json_quote(utc_timestamp()) << ",\n";
  out << "  \"command_line\": " << n2wos::json_quote(command_line) << ",\n";
  out << "  \"cuda\": {\n";
  out << "    \"runtime\": " << n2wos::json_quote(n2wos::cuda_runtime_version_string()) << ",\n";
  out << "    \"device\": " << n2wos::json_quote(n2wos::cuda_device_summary()) << "\n";
  out << "  },\n";
  out << "  \"implementation_mode\": {\n";
  out << "    \"gpu_resident_geometry_query\": true,\n";
  out << "    \"geometry_query\": \"custom_cuda_bvh\",\n";
  out << "    \"fcpw_gpu_public_api\": false,\n";
  out << "    \"host_driven_walker_loop\": false,\n";
  out << "    \"cache_inference\": \"none\",\n";
  out << "    \"rng\": \"host_query_generation_only\",\n";
  out << "    \"accumulation\": \"not_applicable_patch_0001\",\n";
  out << "    \"timing_scope\": \"cuda_kernel_only_excludes_h2d_d2h_and_allocation\"\n";
  out << "  },\n";
  out << "  \"options\": {\n";
  out << "    \"mesh\": " << n2wos::json_quote(opt.mesh) << ",\n";
  out << "    \"mesh_path\": " << n2wos::json_quote(opt.mesh_path) << ",\n";
  out << "    \"normalize\": " << opt.normalize << ",\n";
  out << "    \"bumpy_stacks\": " << opt.bumpy_stacks << ",\n";
  out << "    \"bumpy_slices\": " << opt.bumpy_slices << ",\n";
  out << "    \"bumpy_amplitude\": " << opt.bumpy_amplitude << ",\n";
  out << "    \"leaf_size\": " << opt.leaf_size << ",\n";
  out << "    \"queries\": " << opt.queries << ",\n";
  out << "    \"validate\": " << opt.validate << ",\n";
  out << "    \"repeat\": " << opt.repeat << ",\n";
  out << "    \"block_size\": " << opt.block_size << ",\n";
  out << "    \"seed\": " << opt.seed << ",\n";
  out << "    \"query_extent\": " << opt.query_extent << "\n";
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
  out << "  \"bvh_stats\": {\n";
  out << "    \"nodes\": " << bvh.node_count() << ",\n";
  out << "    \"triangle_indices\": " << bvh.index_count() << ",\n";
  out << "    \"leaf_size\": " << bvh.leaf_size() << ",\n";
  out << "    \"max_depth\": " << bvh.max_depth() << "\n";
  out << "  },\n";
  out << "  \"timing\": {\n";
  out << "    \"kernel_ms\": [";
  for (std::size_t i = 0; i < timings_ms.size(); ++i) {
    if (i) out << ", ";
    out << timings_ms[i];
  }
  out << "],\n";
  out << "    \"kernel_ms_median\": " << med_ms << ",\n";
  out << "    \"kernel_ms_mean\": " << mean_ms << ",\n";
  out << "    \"median_us_per_query\": " << median_us_per_query << ",\n";
  out << "    \"mean_us_per_query\": " << mean_us_per_query << ",\n";
  out << "    \"median_mqueries_per_second\": " << median_mqueries_per_second << "\n";
  out << "  },\n";
  out << "  \"validation\": {\n";
  out << "    \"checked\": " << validation.checked << ",\n";
  out << "    \"bad_distance_count_threshold_2e_4\": " << validation.bad_distance_count << ",\n";
  out << "    \"max_abs_distance_error\": " << validation.max_abs_distance_error << ",\n";
  out << "    \"rms_distance_error\": " << validation.rms_distance_error << ",\n";
  out << "    \"max_closest_point_error\": " << validation.max_closest_point_error << ",\n";
  out << "    \"rms_closest_point_error\": " << validation.rms_closest_point_error << ",\n";
  out << "    \"cuda_stack_overflow_count\": " << result.overflow_count << "\n";
  out << "  }\n";
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
      if (opt.mesh == "obj") {
        mesh = n2wos::load_obj_mesh(opt.mesh_path);
      } else if (opt.mesh == "ply") {
        mesh = n2wos::load_ply_mesh(opt.mesh_path);
      } else {
        throw std::runtime_error("unsupported mesh type after validation: " + opt.mesh);
      }
      if (opt.normalize != 0) {
        transform = n2wos::normalize_to_unit_radius(mesh);
        transform_ptr = &transform;
      }
    }

    const n2wos::Aabb bounds = n2wos::compute_bounds(mesh);
    const std::size_t degenerate_triangles = n2wos::count_degenerate_triangles(mesh);

    std::cerr << "mesh: " << mesh.name << " vertices=" << mesh.vertices.size()
              << " triangles=" << mesh.triangles.size() << "\n";
    std::cerr << "building host BVH and copying to GPU...\n";
    n2wos::CudaBvh bvh(mesh, opt.leaf_size);
    std::cerr << "BVH: nodes=" << bvh.node_count() << " max_depth=" << bvh.max_depth() << "\n";

    const std::vector<n2wos::Vec3f> query_points = make_query_points(opt.queries, opt.seed, opt.query_extent);

    // Warmup.
    (void)bvh.query(query_points, opt.block_size);

    std::vector<float> timings_ms;
    timings_ms.reserve(static_cast<std::size_t>(opt.repeat));
    n2wos::CudaBvhQueryResult result;
    for (int r = 0; r < opt.repeat; ++r) {
      result = bvh.query(query_points, opt.block_size);
      timings_ms.push_back(result.kernel_milliseconds);
      std::cerr << "repeat " << (r + 1) << "/" << opt.repeat
                << ": kernel_ms=" << result.kernel_milliseconds
                << " overflow=" << result.overflow_count << "\n";
    }

    std::cerr << "validating " << std::min(opt.validate, opt.queries) << " points by CPU brute force...\n";
    const ValidationStats validation = validate_against_cpu(mesh, query_points, result, opt.validate);

    write_json(opt.output,
               opt,
               mesh,
               transform_ptr,
               bvh,
               bounds,
               degenerate_triangles,
               timings_ms,
               result,
               validation,
               join_argv(argc, argv));

    std::cout << "wrote " << opt.output << "\n";
    std::cout << "median_us_per_query="
              << (static_cast<double>(median(timings_ms)) * 1000.0 / static_cast<double>(opt.queries))
              << " validation_max_abs_distance_error=" << validation.max_abs_distance_error
              << " overflow=" << result.overflow_count << "\n";

    return validation.bad_distance_count == 0 && result.overflow_count == 0 ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
