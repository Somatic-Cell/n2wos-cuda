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
#include <iomanip>
#include <iostream>
#include <limits>
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

  int width = 128;
  int height = 128;
  int samples_per_pixel = 256;
  std::string slice_view = "xy";
  float plane_z = 0.0f;
  bool use_mesh_bounds = true;
  bool preserve_world_aspect = true;
  float x_min = -1.0f;
  float x_max = 1.0f;
  float y_min = -1.0f;
  float y_max = 1.0f;
  float padding_fraction = 0.02f;
  bool mask_outside_mesh = true;

  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  std::uint64_t seed = 12345;
  int block_size = 128;

  int cubql_leaf_size = 8;
  std::string cubql_build_method = "sah";
  std::string output_prefix = "results/harmonic_slice";
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
    auto value = static_cast<std::uint64_t>(std::stoull(s, &pos));
    if (pos != s.size()) throw std::runtime_error("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse uint64 for " + name + ": " + s);
  }
}

std::vector<float> parse_frame(const std::string& s) {
  std::stringstream ss(s);
  std::string token;
  std::vector<float> values;
  while (std::getline(ss, token, ',')) values.push_back(parse_float(token, "--frame"));
  if (values.size() != 4) throw std::runtime_error("--frame expects u_min,u_max,v_min,v_max in the selected slice view");
  return values;
}

void print_usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0 << " [options]\n"
      << "  --mesh procedural_bumpy_sphere|obj|ply\n"
      << "  --mesh-path PATH\n"
      << "  --width N --height N\n"
      << "  --samples-per-pixel N\n"
      << "  --view xy|xz|yz\n"
      << "  --plane-z Z      fixed coordinate: z for xy, y for xz, x for yz\n"
      << "  --frame u_min,u_max,v_min,v_max   overrides mesh bounds for selected view\n"
      << "  --preserve-world-aspect 0|1\n"
      << "  --mask-outside-mesh 0|1\n"
      << "  --max-steps N --epsilon EPS --step-scale S\n"
      << "  --cubql-build-method sah|spatial_median|radix|elh\n"
      << "  --output-prefix PATH_PREFIX\n";
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
    } else if (arg == "--width") {
      opt.width = parse_int(require_value(i, argc, argv), "--width");
    } else if (arg == "--height") {
      opt.height = parse_int(require_value(i, argc, argv), "--height");
    } else if (arg == "--samples-per-pixel" || arg == "--spp") {
      opt.samples_per_pixel = parse_int(require_value(i, argc, argv), "--samples-per-pixel");
    } else if (arg == "--view") {
      opt.slice_view = require_value(i, argc, argv);
      if (opt.slice_view != "xy" && opt.slice_view != "xz" && opt.slice_view != "yz") {
        throw std::runtime_error("--view must be one of: xy, xz, yz");
      }
    } else if (arg == "--plane-z" || arg == "--plane") {
      opt.plane_z = parse_float(require_value(i, argc, argv), arg);
    } else if (arg == "--frame") {
      auto f = parse_frame(require_value(i, argc, argv));
      opt.x_min = f[0]; opt.x_max = f[1]; opt.y_min = f[2]; opt.y_max = f[3];
      opt.use_mesh_bounds = false;
    } else if (arg == "--padding-fraction") {
      opt.padding_fraction = parse_float(require_value(i, argc, argv), "--padding-fraction");
    } else if (arg == "--preserve-world-aspect") {
      opt.preserve_world_aspect = parse_int(require_value(i, argc, argv), "--preserve-world-aspect") != 0;
    } else if (arg == "--mask-outside-mesh") {
      opt.mask_outside_mesh = parse_int(require_value(i, argc, argv), "--mask-outside-mesh") != 0;
    } else if (arg == "--max-steps") {
      opt.max_steps = parse_int(require_value(i, argc, argv), "--max-steps");
    } else if (arg == "--epsilon") {
      opt.epsilon = parse_float(require_value(i, argc, argv), "--epsilon");
    } else if (arg == "--step-scale") {
      opt.step_scale = parse_float(require_value(i, argc, argv), "--step-scale");
    } else if (arg == "--seed") {
      opt.seed = parse_u64(require_value(i, argc, argv), "--seed");
    } else if (arg == "--block-size") {
      opt.block_size = parse_int(require_value(i, argc, argv), "--block-size");
    } else if (arg == "--cubql-leaf-size") {
      opt.cubql_leaf_size = parse_int(require_value(i, argc, argv), "--cubql-leaf-size");
    } else if (arg == "--cubql-build-method") {
      opt.cubql_build_method = require_value(i, argc, argv);
    } else if (arg == "--output-prefix") {
      opt.output_prefix = require_value(i, argc, argv);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
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

void ensure_parent_dir(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

void write_csv(const std::string& path, const n2wos::SliceRenderResult& result) {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to open CSV for writing: " + path);
  out << "ix,iy,x,y,z,inside,samples,mean,exact,error,stderr,sample_variance,mean_steps,forced_max_steps,overflow_count\n";
  out << std::setprecision(10);
  for (int iy = 0; iy < result.height; ++iy) {
    for (int ix = 0; ix < result.width; ++ix) {
      const int pid = iy * result.width + ix;
      const auto& p = result.pixels[pid];
      out << ix << ',' << iy << ',' << p.x << ',' << p.y << ',' << p.z << ','
          << static_cast<int>(p.inside) << ',' << p.samples << ','
          << p.mean << ',' << p.exact << ',' << p.error << ',' << p.stderr << ','
          << p.sample_variance << ',' << p.mean_steps << ',' << p.forced_max_steps << ',' << p.overflow_count << '\n';
    }
  }
}

struct Rgb { unsigned char r, g, b; };

unsigned char to_byte(double x) {
  x = std::max(0.0, std::min(1.0, x));
  return static_cast<unsigned char>(std::lround(255.0 * x));
}

Rgb scalar_map(double value, double vmin, double vmax) {
  if (!std::isfinite(value)) return {30, 30, 30};
  if (!(vmax > vmin)) return {128, 128, 128};
  const double t = std::max(0.0, std::min(1.0, (value - vmin) / (vmax - vmin)));
  return {to_byte(t), to_byte(t), to_byte(t)};
}

Rgb diverging_map(double value, double max_abs) {
  if (!std::isfinite(value)) return {30, 30, 30};
  if (!(max_abs > 0.0)) return {255, 255, 255};
  const double t = std::max(-1.0, std::min(1.0, value / max_abs));
  if (t < 0.0) {
    const double a = -t;
    return {to_byte(1.0 - a), to_byte(1.0 - a), 255};
  }
  const double a = t;
  return {255, to_byte(1.0 - a), to_byte(1.0 - a)};
}

void write_ppm(const std::string& path, int width, int height, const std::vector<Rgb>& pixels) {
  ensure_parent_dir(path);
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open PPM for writing: " + path);
  out << "P6\n" << width << " " << height << "\n255\n";
  for (const Rgb& c : pixels) {
    out.put(static_cast<char>(c.r));
    out.put(static_cast<char>(c.g));
    out.put(static_cast<char>(c.b));
  }
}

void write_heatmaps(const std::string& prefix, const n2wos::SliceRenderResult& result) {
  double exact_min = std::numeric_limits<double>::infinity();
  double exact_max = -std::numeric_limits<double>::infinity();
  double estimate_min = std::numeric_limits<double>::infinity();
  double estimate_max = -std::numeric_limits<double>::infinity();
  double stderr_max = 0.0;
  double error_max_abs = 0.0;
  for (const auto& p : result.pixels) {
    if (!p.inside) continue;
    exact_min = std::min(exact_min, p.exact);
    exact_max = std::max(exact_max, p.exact);
    estimate_min = std::min(estimate_min, p.mean);
    estimate_max = std::max(estimate_max, p.mean);
    stderr_max = std::max(stderr_max, p.stderr);
    error_max_abs = std::max(error_max_abs, std::fabs(p.error));
  }

  const double value_min = std::min(exact_min, estimate_min);
  const double value_max = std::max(exact_max, estimate_max);

  std::vector<Rgb> mask_img(result.width * result.height);
  std::vector<Rgb> exact_img(result.width * result.height);
  std::vector<Rgb> estimate_img(result.width * result.height);
  std::vector<Rgb> error_img(result.width * result.height);
  std::vector<Rgb> stderr_img(result.width * result.height);
  for (int iy = 0; iy < result.height; ++iy) {
    const int flipped_y = result.height - 1 - iy;
    for (int ix = 0; ix < result.width; ++ix) {
      const int src = iy * result.width + ix;
      const int dst = flipped_y * result.width + ix;
      const auto& p = result.pixels[src];
      if (!p.inside) {
        mask_img[dst] = {20, 20, 20};
        exact_img[dst] = {20, 20, 20};
        estimate_img[dst] = {20, 20, 20};
        error_img[dst] = {20, 20, 20};
        stderr_img[dst] = {20, 20, 20};
      } else {
        mask_img[dst] = {235, 235, 235};
        exact_img[dst] = scalar_map(p.exact, value_min, value_max);
        estimate_img[dst] = scalar_map(p.mean, value_min, value_max);
        error_img[dst] = diverging_map(p.error, error_max_abs);
        stderr_img[dst] = scalar_map(p.stderr, 0.0, stderr_max);
      }
    }
  }
  write_ppm(prefix + "_mask.ppm", result.width, result.height, mask_img);
  write_ppm(prefix + "_exact.ppm", result.width, result.height, exact_img);
  write_ppm(prefix + "_estimate.ppm", result.width, result.height, estimate_img);
  write_ppm(prefix + "_error.ppm", result.width, result.height, error_img);
  write_ppm(prefix + "_stderr.ppm", result.width, result.height, stderr_img);
}

void write_json(const std::string& path,
                const Options& opt,
                int argc,
                char** argv,
                const n2wos::Mesh& mesh,
                const n2wos::NormalizeTransform& norm,
                const n2wos::Aabb& bounds,
                const n2wos::CuBqlBvh& bvh,
                const n2wos::SliceRenderResult& result) {
  ensure_parent_dir(path);
  std::ofstream json(path);
  if (!json) throw std::runtime_error("failed to open JSON for writing: " + path);
  json << std::setprecision(10);
  json << "{\n";
  json << "  \"schema\": \"n2wos_harmonic_slice_render_v2\",\n";
  json << "  \"patch\": \"0004h-fix-slice-visualization-aspect-and-mask\",\n";
  json << "  \"generated_at_utc\": " << n2wos::json_quote(now_utc_iso8601()) << ",\n";
  json << "  \"command_line\": " << n2wos::json_quote(join_command_line(argc, argv)) << ",\n";
  json << "  \"cuda\": {\n";
  json << "    \"runtime\": " << n2wos::json_quote(n2wos::cuda_runtime_version_string()) << ",\n";
  json << "    \"device\": " << n2wos::json_quote(n2wos::cuda_device_summary()) << "\n";
  json << "  },\n";
  json << "  \"implementation_mode\": {\n";
  json << "    \"engine\": \"persistent_per_sample_kernel_for_each_pixel_walk\",\n";
  json << "    \"geometry_backend\": \"cubql_cuda\",\n";
  json << "    \"geometry_build_method\": " << n2wos::json_quote(opt.cubql_build_method) << ",\n";
  json << "    \"cpu_gpu_transfer_inside_sampling_loop\": false,\n";
  json << "    \"host_controls_global_step_loop\": false,\n";
  json << "    \"field_rendering_role\": \"visual_diagnostic_not_final_time_to_mse_benchmark\"\n";
  json << "  },\n";
  json << "  \"options\": {\n";
  json << "    \"mesh\": " << n2wos::json_quote(opt.mesh) << ",\n";
  json << "    \"mesh_path\": " << n2wos::json_quote(opt.mesh_path) << ",\n";
  json << "    \"normalize\": " << (opt.normalize ? "true" : "false") << ",\n";
  json << "    \"width\": " << opt.width << ",\n";
  json << "    \"height\": " << opt.height << ",\n";
  json << "    \"samples_per_pixel\": " << opt.samples_per_pixel << ",\n";
  json << "    \"slice_view\": " << n2wos::json_quote(opt.slice_view) << ",\n";
  json << "    \"plane_z\": " << opt.plane_z << ",\n";
  json << "    \"plane_z_note\": \"fixed coordinate: z for xy, y for xz, x for yz\",\n";
  json << "    \"preserve_world_aspect\": " << (opt.preserve_world_aspect ? "true" : "false") << ",\n";
  json << "    \"mask_outside_mesh\": " << (opt.mask_outside_mesh ? "true" : "false") << ",\n";
  json << "    \"max_steps\": " << opt.max_steps << ",\n";
  json << "    \"epsilon\": " << opt.epsilon << ",\n";
  json << "    \"step_scale\": " << opt.step_scale << ",\n";
  json << "    \"seed\": " << opt.seed << "\n";
  json << "  },\n";
  json << "  \"mesh_stats\": {\n";
  json << "    \"name\": " << n2wos::json_quote(mesh.name) << ",\n";
  json << "    \"vertices\": " << mesh.vertices.size() << ",\n";
  json << "    \"triangles\": " << mesh.triangles.size() << ",\n";
  json << "    \"degenerate_triangles\": " << count_degenerate_triangles(mesh) << ",\n";
  json << "    \"bounds_min\": [" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << "],\n";
  json << "    \"bounds_max\": [" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << "],\n";
  json << "    \"normalization\": {\"center\": [" << norm.center.x << ", " << norm.center.y << ", " << norm.center.z << "], \"scale\": " << norm.scale << "}\n";
  json << "  },\n";
  json << "  \"bvh_stats\": {\n";
  json << "    \"triangles\": " << bvh.triangle_count() << ",\n";
  json << "    \"nodes\": " << bvh.node_count() << ",\n";
  json << "    \"prim_ids\": " << bvh.prim_id_count() << ",\n";
  json << "    \"leaf_size\": " << bvh.leaf_size() << ",\n";
  json << "    \"build_method\": " << n2wos::json_quote(bvh.build_method()) << ",\n";
  json << "    \"build_milliseconds\": " << bvh.build_milliseconds() << "\n";
  json << "  },\n";
  json << "  \"outputs\": {\n";
  json << "    \"csv\": " << n2wos::json_quote(opt.output_prefix + ".csv") << ",\n";
  json << "    \"mask_ppm\": " << n2wos::json_quote(opt.output_prefix + "_mask.ppm") << ",\n";
  json << "    \"exact_ppm\": " << n2wos::json_quote(opt.output_prefix + "_exact.ppm") << ",\n";
  json << "    \"estimate_ppm\": " << n2wos::json_quote(opt.output_prefix + "_estimate.ppm") << ",\n";
  json << "    \"error_ppm\": " << n2wos::json_quote(opt.output_prefix + "_error.ppm") << ",\n";
  json << "    \"stderr_ppm\": " << n2wos::json_quote(opt.output_prefix + "_stderr.ppm") << "\n";
  json << "  },\n";
  json << "  \"slice_frame\": {\n";
  json << "    \"view\": " << n2wos::json_quote(result.slice_view) << ",\n";
  json << "    \"u_min\": " << result.frame_u_min << ",\n";
  json << "    \"u_max\": " << result.frame_u_max << ",\n";
  json << "    \"v_min\": " << result.frame_v_min << ",\n";
  json << "    \"v_max\": " << result.frame_v_max << ",\n";
  json << "    \"world_units_per_pixel_u\": " << result.world_units_per_pixel_u << ",\n";
  json << "    \"world_units_per_pixel_v\": " << result.world_units_per_pixel_v << "\n";
  json << "  },\n";
  json << "  \"summary\": {\n";
  json << "    \"inside_pixels\": " << result.inside_pixels << ",\n";
  json << "    \"elapsed_ms\": " << result.elapsed_ms << ",\n";
  json << "    \"us_per_active_pixel\": " << result.us_per_active_pixel << ",\n";
  json << "    \"us_per_launched_sample\": " << result.us_per_launched_sample << ",\n";
  json << "    \"rmse_inside\": " << result.rmse_inside << ",\n";
  json << "    \"mae_inside\": " << result.mae_inside << ",\n";
  json << "    \"max_abs_error_inside\": " << result.max_abs_error_inside << ",\n";
  json << "    \"forced_max_steps\": " << result.forced_max_steps << ",\n";
  json << "    \"overflow_count\": " << result.overflow_count << "\n";
  json << "  }\n";
  json << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt = parse_args(argc, argv);
    n2wos::Mesh mesh = load_mesh_for_options(opt);
    n2wos::NormalizeTransform norm;
    if (opt.normalize) norm = n2wos::normalize_to_unit_radius(mesh);
    const n2wos::Aabb bounds = n2wos::compute_bounds(mesh);
    n2wos::CuBqlBvh bvh(mesh, opt.cubql_leaf_size, opt.cubql_build_method);

    n2wos::SliceRenderOptions run_opt;
    run_opt.width = opt.width;
    run_opt.height = opt.height;
    run_opt.samples_per_pixel = opt.samples_per_pixel;
    run_opt.slice_view = opt.slice_view;
    run_opt.plane_z = opt.plane_z;
    run_opt.use_mesh_bounds = opt.use_mesh_bounds;
    run_opt.preserve_world_aspect = opt.preserve_world_aspect;
    run_opt.x_min = opt.x_min;
    run_opt.x_max = opt.x_max;
    run_opt.y_min = opt.y_min;
    run_opt.y_max = opt.y_max;
    run_opt.bounds_padding_fraction = opt.padding_fraction;
    run_opt.mask_outside_mesh = opt.mask_outside_mesh;
    run_opt.max_steps = opt.max_steps;
    run_opt.epsilon = opt.epsilon;
    run_opt.step_scale = opt.step_scale;
    run_opt.seed = opt.seed;
    run_opt.block_size = opt.block_size;

    n2wos::SliceRenderResult result = n2wos::render_persistent_harmonic_slice(bvh, mesh, run_opt);
    write_csv(opt.output_prefix + ".csv", result);
    write_heatmaps(opt.output_prefix, result);
    write_json(opt.output_prefix + ".json", opt, argc, argv, mesh, norm, bounds, bvh, result);

    std::cout << "wrote " << opt.output_prefix << ".json/.csv and PPM heatmaps\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
