#include "n2wos/cubql_bvh.hpp"
#include "n2wos/cuda_bvh.hpp"
#include "n2wos/json.hpp"
#include "n2wos/mesh.hpp"
#include "n2wos/wos_wavefront.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string mesh = "procedural_bumpy_sphere";
  std::string mesh_path;
  bool normalize = true;
  int bumpy_stacks = 128;
  int bumpy_slices = 256;
  float bumpy_amplitude = 0.15f;

  std::string method = "pure_wos";
  int samples = 65536;
  int coarse_samples = 0;
  int residual_samples = 0;
  int depth_m = 8;
  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  n2wos::Vec3f x0 = {0.1f, 0.05f, 0.0f};
  std::uint64_t seed = 12345;
  int block_size = 128;

  int cubql_leaf_size = 8;
  std::string cubql_build_method = "sah";
  std::string output = "results/eval_wavefront_wos.json";
};

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + argv[i]);
  return argv[++i];
}

int parse_int(const std::string& s, const std::string& name) {
  try {
    std::size_t pos = 0;
    int value = std::stoi(s, &pos);
    if (pos != s.size()) throw std::runtime_error("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse integer for " + name + ": " + s);
  }
}

float parse_float(const std::string& s, const std::string& name) {
  try {
    std::size_t pos = 0;
    float value = std::stof(s, &pos);
    if (pos != s.size()) throw std::runtime_error("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse float for " + name + ": " + s);
  }
}

std::uint64_t parse_u64(const std::string& s, const std::string& name) {
  try {
    std::size_t pos = 0;
    std::uint64_t value = static_cast<std::uint64_t>(std::stoull(s, &pos));
    if (pos != s.size()) throw std::runtime_error("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse uint64 for " + name + ": " + s);
  }
}

n2wos::Vec3f parse_vec3(const std::string& s) {
  std::stringstream ss(s);
  std::string token;
  std::vector<float> values;
  while (std::getline(ss, token, ',')) values.push_back(parse_float(token, "--x0"));
  if (values.size() != 3) throw std::runtime_error("--x0 expects x,y,z");
  return {values[0], values[1], values[2]};
}

void print_usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0 << " [options]\n"
      << "  --mesh procedural_bumpy_sphere|obj|ply\n"
      << "  --mesh-path PATH\n"
      << "  --method pure_wos|oracle_coarse|oracle_residual|oracle_2lmc\n"
      << "  --samples N\n"
      << "  --coarse-samples N       for oracle_2lmc; default samples\n"
      << "  --residual-samples N     for oracle_2lmc; default max(1, samples/8)\n"
      << "  --depth-m M\n"
      << "  --max-steps N\n"
      << "  --epsilon EPS\n"
      << "  --step-scale S\n"
      << "  --x0 x,y,z\n"
      << "  --cubql-build-method sah|spatial_median|radix|elh\n"
      << "  --cubql-leaf-size N\n"
      << "  --output PATH\n";
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--mesh") {
      opt.mesh = require_value(i, argc, argv);
    } else if (arg == "--mesh-path") {
      opt.mesh_path = require_value(i, argc, argv);
    } else if (arg == "--normalize") {
      opt.normalize = parse_int(require_value(i, argc, argv), "--normalize") != 0;
    } else if (arg == "--bumpy-stacks") {
      opt.bumpy_stacks = parse_int(require_value(i, argc, argv), "--bumpy-stacks");
    } else if (arg == "--bumpy-slices") {
      opt.bumpy_slices = parse_int(require_value(i, argc, argv), "--bumpy-slices");
    } else if (arg == "--bumpy-amplitude") {
      opt.bumpy_amplitude = parse_float(require_value(i, argc, argv), "--bumpy-amplitude");
    } else if (arg == "--method") {
      opt.method = require_value(i, argc, argv);
    } else if (arg == "--samples") {
      opt.samples = parse_int(require_value(i, argc, argv), "--samples");
    } else if (arg == "--coarse-samples") {
      opt.coarse_samples = parse_int(require_value(i, argc, argv), "--coarse-samples");
    } else if (arg == "--residual-samples") {
      opt.residual_samples = parse_int(require_value(i, argc, argv), "--residual-samples");
    } else if (arg == "--depth-m") {
      opt.depth_m = parse_int(require_value(i, argc, argv), "--depth-m");
    } else if (arg == "--max-steps") {
      opt.max_steps = parse_int(require_value(i, argc, argv), "--max-steps");
    } else if (arg == "--epsilon") {
      opt.epsilon = parse_float(require_value(i, argc, argv), "--epsilon");
    } else if (arg == "--step-scale") {
      opt.step_scale = parse_float(require_value(i, argc, argv), "--step-scale");
    } else if (arg == "--x0") {
      opt.x0 = parse_vec3(require_value(i, argc, argv));
    } else if (arg == "--seed") {
      opt.seed = parse_u64(require_value(i, argc, argv), "--seed");
    } else if (arg == "--block-size") {
      opt.block_size = parse_int(require_value(i, argc, argv), "--block-size");
    } else if (arg == "--cubql-leaf-size") {
      opt.cubql_leaf_size = parse_int(require_value(i, argc, argv), "--cubql-leaf-size");
    } else if (arg == "--cubql-build-method") {
      opt.cubql_build_method = require_value(i, argc, argv);
    } else if (arg == "--output") {
      opt.output = require_value(i, argc, argv);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.method != "pure_wos" && opt.method != "oracle_coarse" &&
      opt.method != "oracle_residual" && opt.method != "oracle_2lmc") {
    throw std::runtime_error("--method must be pure_wos, oracle_coarse, oracle_residual, or oracle_2lmc");
  }
  if (opt.samples <= 0 || opt.max_steps <= 0 || opt.block_size <= 0) {
    throw std::runtime_error("samples, max-steps, and block-size must be positive");
  }
  return opt;
}

n2wos::Mesh load_mesh_for_options(const Options& opt) {
  if (opt.mesh == "procedural_bumpy_sphere") {
    return n2wos::make_procedural_bumpy_sphere(opt.bumpy_stacks, opt.bumpy_slices, opt.bumpy_amplitude);
  }
  if (opt.mesh == "obj") {
    if (opt.mesh_path.empty()) throw std::runtime_error("--mesh obj requires --mesh-path");
    return n2wos::load_obj_mesh(opt.mesh_path);
  }
  if (opt.mesh == "ply") {
    if (opt.mesh_path.empty()) throw std::runtime_error("--mesh ply requires --mesh-path");
    return n2wos::load_ply_mesh(opt.mesh_path);
  }
  throw std::runtime_error("unknown mesh type: " + opt.mesh);
}

int count_degenerate_triangles(const n2wos::Mesh& mesh) {
  int count = 0;
  for (const n2wos::Triangle& tri : mesh.triangles) {
    const n2wos::Vec3f a = mesh.vertices[tri.v0];
    const n2wos::Vec3f b = mesh.vertices[tri.v1];
    const n2wos::Vec3f c = mesh.vertices[tri.v2];
    if (n2wos::length2(n2wos::cross(b - a, c - a)) <= 1.0e-24f) ++count;
  }
  return count;
}

std::string now_utc_iso8601() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const std::time_t t = clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string join_command_line(int argc, char** argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i) out << ' ';
    out << argv[i];
  }
  return out.str();
}

std::string vec3_json(const n2wos::Vec3f& p) {
  std::ostringstream out;
  out << "[" << p.x << ", " << p.y << ", " << p.z << "]";
  return out.str();
}

void write_run_json(std::ostream& out, const std::string& name, const n2wos::WavefrontRunStats& s, double exact_target) {
  const double error = s.mean - exact_target;
  out << "    " << n2wos::json_quote(name) << ": {\n";
  out << "      \"samples\": " << s.samples << ",\n";
  out << "      \"mean\": " << s.mean << ",\n";
  out << "      \"exact_target\": " << exact_target << ",\n";
  out << "      \"error\": " << error << ",\n";
  out << "      \"abs_error\": " << std::abs(error) << ",\n";
  out << "      \"sample_variance\": " << s.sample_variance << ",\n";
  out << "      \"estimator_variance\": " << s.estimator_variance << ",\n";
  out << "      \"stderr\": " << s.stderr << ",\n";
  out << "      \"mean_steps\": " << s.mean_steps << ",\n";
  out << "      \"elapsed_ms\": " << s.elapsed_ms << ",\n";
  out << "      \"us_per_sample\": " << s.us_per_sample << ",\n";
  out << "      \"launched_query_slots\": " << s.launched_query_slots << ",\n";
  out << "      \"mean_launched_query_slots_per_sample\": " << s.mean_launched_query_slots_per_sample << ",\n";
  out << "      \"scheduled_query_rounds\": " << s.scheduled_query_rounds << ",\n";
  out << "      \"forced_max_steps\": " << s.forced_max_steps << ",\n";
  out << "      \"overflow_count\": " << s.overflow_count << "\n";
  out << "    }";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    n2wos::Mesh mesh = load_mesh_for_options(opt);
    n2wos::NormalizeTransform normalization;
    if (opt.normalize) normalization = n2wos::normalize_to_unit_radius(mesh);
    const n2wos::Aabb bounds = n2wos::compute_bounds(mesh);
    const int degenerate = count_degenerate_triangles(mesh);

    std::cerr << "building cuBQL BVH method=" << opt.cubql_build_method
              << " leaf=" << opt.cubql_leaf_size << "...\n";
    n2wos::CuBqlBvh bvh(mesh, opt.cubql_leaf_size, opt.cubql_build_method);

    n2wos::WavefrontRunOptions run_opt;
    run_opt.samples = opt.samples;
    run_opt.x0 = opt.x0;
    run_opt.max_steps = opt.max_steps;
    run_opt.depth_m = opt.depth_m;
    run_opt.epsilon = opt.epsilon;
    run_opt.step_scale = opt.step_scale;
    run_opt.seed = opt.seed;
    run_opt.block_size = opt.block_size;

    const double exact_u_x0 = n2wos::harmonic_x2_minus_y2(opt.x0);
    std::ostringstream runs_json;
    double estimator_mean = 0.0;
    double estimator_variance = 0.0;
    double estimator_stderr = 0.0;
    double elapsed_ms_total = 0.0;

    if (opt.method == "pure_wos") {
      const n2wos::WavefrontRunStats s = n2wos::run_wavefront_harmonic(bvh, n2wos::WavefrontMethod::PureWos, run_opt);
      estimator_mean = s.mean;
      estimator_variance = s.estimator_variance;
      estimator_stderr = s.stderr;
      elapsed_ms_total = s.elapsed_ms;
      write_run_json(runs_json, "pure_wos", s, exact_u_x0);
    } else if (opt.method == "oracle_coarse") {
      const n2wos::WavefrontRunStats s = n2wos::run_wavefront_harmonic(bvh, n2wos::WavefrontMethod::OracleCoarse, run_opt);
      estimator_mean = s.mean;
      estimator_variance = s.estimator_variance;
      estimator_stderr = s.stderr;
      elapsed_ms_total = s.elapsed_ms;
      write_run_json(runs_json, "oracle_coarse", s, exact_u_x0);
    } else if (opt.method == "oracle_residual") {
      const n2wos::WavefrontRunStats s = n2wos::run_wavefront_harmonic(bvh, n2wos::WavefrontMethod::OracleResidual, run_opt);
      estimator_mean = s.mean;
      estimator_variance = s.estimator_variance;
      estimator_stderr = s.stderr;
      elapsed_ms_total = s.elapsed_ms;
      write_run_json(runs_json, "oracle_residual", s, 0.0);
    } else if (opt.method == "oracle_2lmc") {
      n2wos::WavefrontRunOptions coarse_opt = run_opt;
      n2wos::WavefrontRunOptions residual_opt = run_opt;
      coarse_opt.samples = opt.coarse_samples > 0 ? opt.coarse_samples : opt.samples;
      residual_opt.samples = opt.residual_samples > 0 ? opt.residual_samples : std::max(1, opt.samples / 8);
      residual_opt.seed = opt.seed ^ 0x517cc1b727220a95ull;
      const n2wos::WavefrontRunStats coarse = n2wos::run_wavefront_harmonic(bvh, n2wos::WavefrontMethod::OracleCoarse, coarse_opt);
      const n2wos::WavefrontRunStats residual = n2wos::run_wavefront_harmonic(bvh, n2wos::WavefrontMethod::OracleResidual, residual_opt);
      estimator_mean = coarse.mean + residual.mean;
      estimator_variance = coarse.estimator_variance + residual.estimator_variance;
      estimator_stderr = std::sqrt(estimator_variance);
      elapsed_ms_total = coarse.elapsed_ms + residual.elapsed_ms;
      write_run_json(runs_json, "coarse", coarse, exact_u_x0);
      runs_json << ",\n";
      write_run_json(runs_json, "residual", residual, 0.0);
    }

    const double estimator_error = estimator_mean - exact_u_x0;
    std::ostringstream json;
    json.precision(10);
    json << "{\n";
    json << "  \"schema\": \"n2wos_wavefront_wos_eval_v1\",\n";
    json << "  \"patch\": \"0004-add-common-wavefront-wos-engine\",\n";
    json << "  \"generated_at_utc\": " << n2wos::json_quote(now_utc_iso8601()) << ",\n";
    json << "  \"command_line\": " << n2wos::json_quote(join_command_line(argc, argv)) << ",\n";
    json << "  \"cuda\": {\n";
    json << "    \"runtime\": " << n2wos::json_quote(n2wos::cuda_runtime_version_string()) << ",\n";
    json << "    \"device\": " << n2wos::json_quote(n2wos::cuda_device_summary()) << "\n";
    json << "  },\n";
    json << "  \"implementation_mode\": {\n";
    json << "    \"engine\": \"batched_wavefront_global_step_loop\",\n";
    json << "    \"geometry_backend\": \"cubql_cuda\",\n";
    json << "    \"geometry_build_method\": " << n2wos::json_quote(opt.cubql_build_method) << ",\n";
    json << "    \"gpu_resident_walker_state\": true,\n";
    json << "    \"gpu_rng\": \"pcg32_per_sample_state\",\n";
    json << "    \"cpu_gpu_transfer_inside_sampling_loop\": false,\n";
    json << "    \"host_controls_global_step_loop\": true,\n";
    json << "    \"per_walk_kernel_launch\": false,\n";
    json << "    \"active_compaction\": false,\n";
    json << "    \"inactive_slots_skip_bvh_traversal\": true,\n";
    json << "    \"coarse_query_rounds_capped_at_depth_m\": true,\n";
    json << "    \"cache_backend\": \"analytic_oracle_debug_only\",\n";
    json << "    \"tcnn_in_solver\": false,\n";
    json << "    \"timing_scope\": \"cuda_events_include_wavefront_query_update_reduction_exclude_bvh_build_allocation_final_readback\",\n";
    json << "    \"production_status\": \"scaffold_for_common_method_engine_not_final_wall_clock\"\n";
    json << "  },\n";
    json << "  \"options\": {\n";
    json << "    \"mesh\": " << n2wos::json_quote(opt.mesh) << ",\n";
    json << "    \"mesh_path\": " << n2wos::json_quote(opt.mesh_path) << ",\n";
    json << "    \"normalize\": " << (opt.normalize ? "true" : "false") << ",\n";
    json << "    \"method\": " << n2wos::json_quote(opt.method) << ",\n";
    json << "    \"samples\": " << opt.samples << ",\n";
    json << "    \"coarse_samples\": " << opt.coarse_samples << ",\n";
    json << "    \"residual_samples\": " << opt.residual_samples << ",\n";
    json << "    \"depth_m\": " << opt.depth_m << ",\n";
    json << "    \"max_steps\": " << opt.max_steps << ",\n";
    json << "    \"epsilon\": " << opt.epsilon << ",\n";
    json << "    \"step_scale\": " << opt.step_scale << ",\n";
    json << "    \"x0\": " << vec3_json(opt.x0) << ",\n";
    json << "    \"seed\": " << opt.seed << ",\n";
    json << "    \"block_size\": " << opt.block_size << ",\n";
    json << "    \"cubql_leaf_size\": " << opt.cubql_leaf_size << ",\n";
    json << "    \"cubql_build_method\": " << n2wos::json_quote(opt.cubql_build_method) << "\n";
    json << "  },\n";
    json << "  \"mesh_stats\": {\n";
    json << "    \"name\": " << n2wos::json_quote(mesh.name) << ",\n";
    json << "    \"vertices\": " << mesh.vertices.size() << ",\n";
    json << "    \"triangles\": " << mesh.triangles.size() << ",\n";
    json << "    \"degenerate_triangles\": " << degenerate << ",\n";
    json << "    \"bounds_min\": " << vec3_json(bounds.min) << ",\n";
    json << "    \"bounds_max\": " << vec3_json(bounds.max) << ",\n";
    json << "    \"normalization\": {\"center\": " << vec3_json(normalization.center) << ", \"scale\": " << normalization.scale << "}\n";
    json << "  },\n";
    json << "  \"bvh_stats\": {\n";
    json << "    \"triangles\": " << bvh.triangle_count() << ",\n";
    json << "    \"nodes\": " << bvh.node_count() << ",\n";
    json << "    \"prim_ids\": " << bvh.prim_id_count() << ",\n";
    json << "    \"leaf_size\": " << bvh.leaf_size() << ",\n";
    json << "    \"build_method\": " << n2wos::json_quote(bvh.build_method()) << ",\n";
    json << "    \"build_milliseconds\": " << bvh.build_milliseconds() << "\n";
    json << "  },\n";
    json << "  \"target\": {\"boundary_condition\": \"harmonic_x2_minus_y2\", \"exact_u_x0\": " << exact_u_x0 << "},\n";
    json << "  \"runs\": {\n" << runs_json.str() << "\n  },\n";
    json << "  \"estimator\": {\n";
    json << "    \"method\": " << n2wos::json_quote(opt.method) << ",\n";
    json << "    \"mean\": " << estimator_mean << ",\n";
    json << "    \"exact\": " << exact_u_x0 << ",\n";
    json << "    \"error\": " << estimator_error << ",\n";
    json << "    \"abs_error\": " << std::abs(estimator_error) << ",\n";
    json << "    \"estimator_variance\": " << estimator_variance << ",\n";
    json << "    \"stderr\": " << estimator_stderr << ",\n";
    json << "    \"elapsed_ms_total\": " << elapsed_ms_total << "\n";
    json << "  }\n";
    json << "}\n";

    std::filesystem::path output_path(opt.output);
    if (output_path.has_parent_path()) std::filesystem::create_directories(output_path.parent_path());
    std::ofstream fout(opt.output);
    if (!fout) throw std::runtime_error("failed to open output: " + opt.output);
    fout << json.str();
    std::cout << json.str();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
