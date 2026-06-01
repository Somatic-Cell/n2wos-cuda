#include "n2wos/cubql_bvh.hpp"
#include "n2wos/json.hpp"
#include "n2wos/mesh.hpp"
#include "n2wos/tcnn_nc_wos.hpp"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define N2WOS_CUDA_CHECK(expr) \
  do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err__)); \
    } \
  } while (false)

using precision_t = tcnn::network_precision_t;

struct Options {
  std::string mesh = "procedural_bumpy_sphere";
  std::string mesh_path;
  bool normalize = true;
  int bumpy_stacks = 128;
  int bumpy_slices = 256;
  float bumpy_amplitude = 0.15f;
  std::string cubql_build_method = "sah";
  int cubql_leaf_size = 8;
  n2wos::NcBoundaryMode boundary_mode = n2wos::NcBoundaryMode::ExternalChargesHigh;
  n2wos::NcLabelSource label_source = n2wos::NcLabelSource::WosSupervision;
  int train_points = 20000;
  int eval_points = 8192;
  int label_refreshes = 4;
  int walks_per_label_refresh = 16;
  int train_steps_per_refresh = 1000;
  int pure_walks_per_point = 64;
  int hybrid_walks_per_point = 4;
  int coarse_walks_per_point = 64;
  int residual_walks_per_point = 4;
  int enable_2lmc = 1;
  int depth_m = 1;
  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  int seed = 12345;
  int block_size = 128;
  std::string cache_preset = "baseline";
  int n_levels = 12;
  int n_features_per_level = 2;
  int log2_hashmap_size = 18;
  int base_resolution = 8;
  float per_level_scale = 1.5f;
  int n_neurons = 32;
  int n_hidden_layers = 2;
  float learning_rate = 1.0e-2f;
  std::string network = "FullyFusedMLP";
  int jit = 0;
  std::string output = "results/eval_tcnn_nc_wos.json";
};

struct Timer {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  Timer() { N2WOS_CUDA_CHECK(cudaEventCreate(&start)); N2WOS_CUDA_CHECK(cudaEventCreate(&stop)); }
  ~Timer() { if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop); }
  void begin(cudaStream_t s) { N2WOS_CUDA_CHECK(cudaEventRecord(start, s)); }
  float end(cudaStream_t s) { N2WOS_CUDA_CHECK(cudaEventRecord(stop, s)); N2WOS_CUDA_CHECK(cudaEventSynchronize(stop)); float ms=0; N2WOS_CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop)); return ms; }
};

struct Stats {
  double mse = 0.0;
  double rmse = 0.0;
  double mae = 0.0;
  double max_abs_error = 0.0;
  double mean_estimate = 0.0;
  double mean_exact = 0.0;
  double mean_sample_variance = 0.0;
  double mean_steps = 0.0;
  unsigned long long cache_queries = 0;
  unsigned long long forced = 0;
  unsigned long long overflow = 0;
};

struct TwoLevelStats {
  double mse = 0.0;
  double rmse = 0.0;
  double mae = 0.0;
  double max_abs_error = 0.0;
  double mean_estimate = 0.0;
  double mean_exact = 0.0;
  double mean_coarse = 0.0;
  double mean_residual = 0.0;
  double mean_coarse_sample_variance = 0.0;
  double mean_residual_sample_variance = 0.0;
  double mean_coarse_steps = 0.0;
  double mean_residual_steps = 0.0;
  unsigned long long coarse_cache_queries = 0;
  unsigned long long residual_cache_queries = 0;
  unsigned long long coarse_forced = 0;
  unsigned long long residual_forced = 0;
  unsigned long long coarse_overflow = 0;
  unsigned long long residual_overflow = 0;
};

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
  return argv[++i];
}

void apply_cache_preset(Options& o, const std::string& preset) {
  o.cache_preset = preset;
  if (preset == "baseline" || preset == "paper_like") {
    o.n_levels = 12; o.n_features_per_level = 2; o.log2_hashmap_size = 18; o.base_resolution = 8; o.per_level_scale = 1.5f; o.n_neurons = 32; o.n_hidden_layers = 2;
  } else if (preset == "light") {
    o.n_levels = 8; o.n_features_per_level = 2; o.log2_hashmap_size = 15; o.base_resolution = 8; o.per_level_scale = 1.5f; o.n_neurons = 16; o.n_hidden_layers = 1;
  } else if (preset == "nano") {
    o.n_levels = 6; o.n_features_per_level = 2; o.log2_hashmap_size = 13; o.base_resolution = 8; o.per_level_scale = 1.5f; o.n_neurons = 16; o.n_hidden_layers = 1;
  } else if (preset == "custom") {
    // Keep explicitly supplied architecture parameters.
  } else {
    throw std::runtime_error("unknown cache preset: " + preset + " (expected baseline, light, nano, custom)");
  }
}

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --mesh procedural_bumpy_sphere|obj|ply\n"
            << "  --mesh-path <path>\n"
            << "  --bc harmonic_x2_minus_y2|external_charges_medium|external_charges_high\n"
            << "  --label-source wos_supervision|exact_analytic\n"
            << "  --cache-preset baseline|light|nano|custom\n"
            << "  --train-points <int> --eval-points <int>\n"
            << "  --label-refreshes <int> --walks-per-label-refresh <int>\n"
            << "  --train-steps-per-refresh <int>\n"
            << "  --pure-walks-per-point <int> --hybrid-walks-per-point <int>\n"
            << "  --coarse-walks-per-point <int> --residual-walks-per-point <int>\n"
            << "  --depth-m <int> --enable-2lmc 0|1 --output <path>\n";
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else if (a == "--mesh") o.mesh = require_value(i,argc,argv);
    else if (a == "--mesh-path") o.mesh_path = require_value(i,argc,argv);
    else if (a == "--normalize") o.normalize = std::stoi(require_value(i,argc,argv)) != 0;
    else if (a == "--bumpy-stacks") o.bumpy_stacks = std::stoi(require_value(i,argc,argv));
    else if (a == "--bumpy-slices") o.bumpy_slices = std::stoi(require_value(i,argc,argv));
    else if (a == "--bumpy-amplitude") o.bumpy_amplitude = std::stof(require_value(i,argc,argv));
    else if (a == "--cubql-build-method") o.cubql_build_method = require_value(i,argc,argv);
    else if (a == "--cubql-leaf-size") o.cubql_leaf_size = std::stoi(require_value(i,argc,argv));
    else if (a == "--bc" || a == "--boundary") { const auto v=require_value(i,argc,argv); o.boundary_mode = n2wos::parse_nc_boundary_mode(v.c_str()); }
    else if (a == "--label-source") { const auto v=require_value(i,argc,argv); o.label_source = n2wos::parse_nc_label_source(v.c_str()); }
    else if (a == "--cache-preset") apply_cache_preset(o, require_value(i,argc,argv));
    else if (a == "--train-points") o.train_points = std::stoi(require_value(i,argc,argv));
    else if (a == "--eval-points") o.eval_points = std::stoi(require_value(i,argc,argv));
    else if (a == "--label-refreshes") o.label_refreshes = std::stoi(require_value(i,argc,argv));
    else if (a == "--walks-per-label-refresh") o.walks_per_label_refresh = std::stoi(require_value(i,argc,argv));
    else if (a == "--train-steps-per-refresh") o.train_steps_per_refresh = std::stoi(require_value(i,argc,argv));
    else if (a == "--pure-walks-per-point") o.pure_walks_per_point = std::stoi(require_value(i,argc,argv));
    else if (a == "--hybrid-walks-per-point") o.hybrid_walks_per_point = std::stoi(require_value(i,argc,argv));
    else if (a == "--coarse-walks-per-point") o.coarse_walks_per_point = std::stoi(require_value(i,argc,argv));
    else if (a == "--residual-walks-per-point") o.residual_walks_per_point = std::stoi(require_value(i,argc,argv));
    else if (a == "--enable-2lmc") o.enable_2lmc = std::stoi(require_value(i,argc,argv));
    else if (a == "--depth-m") o.depth_m = std::stoi(require_value(i,argc,argv));
    else if (a == "--max-steps") o.max_steps = std::stoi(require_value(i,argc,argv));
    else if (a == "--epsilon") o.epsilon = std::stof(require_value(i,argc,argv));
    else if (a == "--step-scale") o.step_scale = std::stof(require_value(i,argc,argv));
    else if (a == "--seed") o.seed = std::stoi(require_value(i,argc,argv));
    else if (a == "--block-size") o.block_size = std::stoi(require_value(i,argc,argv));
    else if (a == "--network") o.network = require_value(i,argc,argv);
    else if (a == "--n-levels") { o.n_levels = std::stoi(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--n-features-per-level") { o.n_features_per_level = std::stoi(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--log2-hashmap-size") { o.log2_hashmap_size = std::stoi(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--base-resolution") { o.base_resolution = std::stoi(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--per-level-scale") { o.per_level_scale = std::stof(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--n-neurons") { o.n_neurons = std::stoi(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--n-hidden-layers") { o.n_hidden_layers = std::stoi(require_value(i,argc,argv)); o.cache_preset = "custom"; }
    else if (a == "--learning-rate") o.learning_rate = std::stof(require_value(i,argc,argv));
    else if (a == "--jit") o.jit = std::stoi(require_value(i,argc,argv));
    else if (a == "--output") o.output = require_value(i,argc,argv);
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.train_points <= 0 || o.eval_points <= 0 || o.pure_walks_per_point <= 0 || o.hybrid_walks_per_point <= 0) throw std::runtime_error("invalid sample counts");
  if (o.coarse_walks_per_point <= 0 || o.residual_walks_per_point <= 0) throw std::runtime_error("invalid 2LMC sample counts");
  if (o.label_refreshes <= 0 || o.walks_per_label_refresh <= 0 || o.train_steps_per_refresh < 0) throw std::runtime_error("invalid training schedule");
  if (o.depth_m < 0 || o.max_steps <= 0 || !(o.epsilon > 0.0f)) throw std::runtime_error("invalid WoS options");
  return o;
}

std::string now_utc() {
  std::time_t t = std::time(nullptr); std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[64]; std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm); return buf;
}

std::string command_line(int argc, char** argv) { std::ostringstream o; for (int i=0;i<argc;++i){ if(i)o<<' '; o<<argv[i]; } return o.str(); }

std::string cuda_json() {
  int rt=0, dr=0, dev=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&dr); cudaGetDevice(&dev); cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
  std::ostringstream o; o << "{\"runtime\": \"runtime=" << rt << ",driver=" << dr << "\", \"device\": " << n2wos::json_quote(std::string(p.name) + " sm_" + std::to_string(p.major) + std::to_string(p.minor)) << "}"; return o.str();
}

n2wos::Mesh load_mesh(const Options& o) {
  if (o.mesh == "procedural_bumpy_sphere") return n2wos::make_procedural_bumpy_sphere(o.bumpy_stacks, o.bumpy_slices, o.bumpy_amplitude);
  if (o.mesh == "obj") { if (o.mesh_path.empty()) throw std::runtime_error("--mesh-path required for obj"); return n2wos::load_obj_mesh(o.mesh_path); }
  if (o.mesh == "ply") { if (o.mesh_path.empty()) throw std::runtime_error("--mesh-path required for ply"); return n2wos::load_ply_mesh(o.mesh_path); }
  throw std::runtime_error("unknown mesh: " + o.mesh);
}

int count_degenerate(const n2wos::Mesh& mesh) { int c=0; for (const auto& t:mesh.triangles){ const auto a=mesh.vertices[t.v0], b=mesh.vertices[t.v1], d=mesh.vertices[t.v2]; if(n2wos::length2(n2wos::cross(b-a,d-a)) <= 1e-24f) ++c; } return c; }

std::vector<n2wos::DeviceVec3> make_ball_points(int n, const n2wos::Aabb& bounds, std::uint64_t seed) {
  const n2wos::Vec3f center{0.5f*(bounds.min.x+bounds.max.x), 0.5f*(bounds.min.y+bounds.max.y), 0.5f*(bounds.min.z+bounds.max.z)};
  const float radius = 0.42f * std::min({bounds.max.x-bounds.min.x, bounds.max.y-bounds.min.y, bounds.max.z-bounds.min.z});
  std::mt19937_64 rng(seed); std::uniform_real_distribution<float> u(0.0f, 1.0f); std::vector<n2wos::DeviceVec3> pts; pts.reserve(n);
  for (int i=0;i<n;++i) { const float z=1.0f-2.0f*u(rng); const float phi=6.283185307179586f*u(rng); const float rr=radius*std::cbrt(std::max(0.0f,u(rng))); const float s=std::sqrt(std::max(0.0f,1.0f-z*z)); pts.push_back({center.x+rr*s*std::cos(phi), center.y+rr*s*std::sin(phi), center.z+rr*z}); }
  return pts;
}

n2wos::Vec3f input_min(const n2wos::Aabb& b) { const float pad=0.05f; return {b.min.x-pad*(b.max.x-b.min.x), b.min.y-pad*(b.max.y-b.min.y), b.min.z-pad*(b.max.z-b.min.z)}; }
n2wos::Vec3f input_extent(const n2wos::Aabb& b) { const float pad=0.05f; return {(b.max.x-b.min.x)*(1+2*pad), (b.max.y-b.min.y)*(1+2*pad), (b.max.z-b.min.z)*(1+2*pad)}; }

Stats summarize(const std::vector<float>& values,
                const std::vector<int>& steps,
                const std::vector<int>& forced,
                const std::vector<int>& overflow,
                const std::vector<unsigned char>* needs_cache,
                const std::vector<n2wos::DeviceVec3>& points,
                int wpp,
                n2wos::NcBoundaryMode bc) {
  Stats out; const int n=static_cast<int>(points.size()); double sum_mse=0, sum_abs=0, sum_var=0, sum_est=0, sum_exact=0, sum_steps=0;
  for (int p=0;p<n;++p) {
    double s=0, ss=0, st=0;
    for (int w=0;w<wpp;++w){ const int idx=p*wpp+w; const double v=values[idx]; s+=v; ss+=v*v; st+=steps[idx]; out.forced += forced[idx]?1ull:0ull; out.overflow += overflow[idx]?1ull:0ull; if(needs_cache && (*needs_cache)[idx]) ++out.cache_queries; }
    const double mean=s/static_cast<double>(wpp); const n2wos::Vec3f hp{points[p].x,points[p].y,points[p].z}; const double exact=n2wos::nc_boundary_value_host(hp,bc); const double err=mean-exact; const double centered=ss-static_cast<double>(wpp)*mean*mean; sum_var += wpp>1 ? std::max(0.0,centered)/static_cast<double>(wpp-1) : 0.0; sum_mse+=err*err; sum_abs+=std::fabs(err); out.max_abs_error=std::max(out.max_abs_error,std::fabs(err)); sum_est+=mean; sum_exact+=exact; sum_steps += st/static_cast<double>(wpp);
  }
  const double dn=static_cast<double>(n); out.mse=sum_mse/dn; out.rmse=std::sqrt(out.mse); out.mae=sum_abs/dn; out.mean_sample_variance=sum_var/dn; out.mean_estimate=sum_est/dn; out.mean_exact=sum_exact/dn; out.mean_steps=sum_steps/dn; return out;
}

TwoLevelStats summarize_two_level(const std::vector<float>& coarse_values,
                                  const std::vector<float>& residual_values,
                                  const std::vector<int>& coarse_steps,
                                  const std::vector<int>& residual_steps,
                                  const std::vector<int>& coarse_forced,
                                  const std::vector<int>& residual_forced,
                                  const std::vector<int>& coarse_overflow,
                                  const std::vector<int>& residual_overflow,
                                  const std::vector<unsigned char>& coarse_needs,
                                  const std::vector<unsigned char>& residual_needs,
                                  const std::vector<n2wos::DeviceVec3>& points,
                                  int coarse_wpp,
                                  int residual_wpp,
                                  n2wos::NcBoundaryMode bc) {
  TwoLevelStats out; const int n=static_cast<int>(points.size()); double sum_mse=0, sum_abs=0, sum_est=0, sum_exact=0, sum_c=0, sum_r=0, sum_cvar=0, sum_rvar=0, sum_csteps=0, sum_rsteps=0;
  for (int p=0; p<n; ++p) {
    double csum=0, css=0, cst=0;
    for (int w=0; w<coarse_wpp; ++w) { const int idx=p*coarse_wpp+w; const double v=coarse_values[idx]; csum+=v; css+=v*v; cst+=coarse_steps[idx]; out.coarse_forced += coarse_forced[idx]?1ull:0ull; out.coarse_overflow += coarse_overflow[idx]?1ull:0ull; if(coarse_needs[idx]) ++out.coarse_cache_queries; }
    double rsum=0, rss=0, rst=0;
    for (int w=0; w<residual_wpp; ++w) { const int idx=p*residual_wpp+w; const double v=residual_values[idx]; rsum+=v; rss+=v*v; rst+=residual_steps[idx]; out.residual_forced += residual_forced[idx]?1ull:0ull; out.residual_overflow += residual_overflow[idx]?1ull:0ull; if(residual_needs[idx]) ++out.residual_cache_queries; }
    const double cm=csum/static_cast<double>(coarse_wpp); const double rm=rsum/static_cast<double>(residual_wpp); const double est=cm+rm; const n2wos::Vec3f hp{points[p].x,points[p].y,points[p].z}; const double exact=n2wos::nc_boundary_value_host(hp,bc); const double err=est-exact;
    sum_c += cm; sum_r += rm; sum_est += est; sum_exact += exact; sum_mse += err*err; sum_abs += std::fabs(err); out.max_abs_error=std::max(out.max_abs_error,std::fabs(err));
    sum_cvar += coarse_wpp>1 ? std::max(0.0, css-static_cast<double>(coarse_wpp)*cm*cm)/static_cast<double>(coarse_wpp-1) : 0.0;
    sum_rvar += residual_wpp>1 ? std::max(0.0, rss-static_cast<double>(residual_wpp)*rm*rm)/static_cast<double>(residual_wpp-1) : 0.0;
    sum_csteps += cst/static_cast<double>(coarse_wpp); sum_rsteps += rst/static_cast<double>(residual_wpp);
  }
  const double dn=static_cast<double>(n); out.mse=sum_mse/dn; out.rmse=std::sqrt(out.mse); out.mae=sum_abs/dn; out.mean_estimate=sum_est/dn; out.mean_exact=sum_exact/dn; out.mean_coarse=sum_c/dn; out.mean_residual=sum_r/dn; out.mean_coarse_sample_variance=sum_cvar/dn; out.mean_residual_sample_variance=sum_rvar/dn; out.mean_coarse_steps=sum_csteps/dn; out.mean_residual_steps=sum_rsteps/dn; return out;
}

std::string stats_json(const Stats& s, int points, int wpp, float elapsed_ms, float training_ms=0.0f) {
  std::ostringstream o; o << std::setprecision(9) << "{\n"
    << "      \"eval_points\": " << points << ",\n"
    << "      \"walks_per_point\": " << wpp << ",\n"
    << "      \"samples\": " << points*wpp << ",\n"
    << "      \"rmse\": " << s.rmse << ",\n"
    << "      \"mae\": " << s.mae << ",\n"
    << "      \"max_abs_error\": " << s.max_abs_error << ",\n"
    << "      \"mean_estimate\": " << s.mean_estimate << ",\n"
    << "      \"mean_exact\": " << s.mean_exact << ",\n"
    << "      \"mean_bias\": " << (s.mean_estimate-s.mean_exact) << ",\n"
    << "      \"mean_sample_variance\": " << s.mean_sample_variance << ",\n"
    << "      \"mean_steps\": " << s.mean_steps << ",\n"
    << "      \"cache_queries\": " << s.cache_queries << ",\n"
    << "      \"cache_query_fraction\": " << (points*wpp>0?static_cast<double>(s.cache_queries)/static_cast<double>(points*wpp):0.0) << ",\n"
    << "      \"forced_max_steps\": " << s.forced << ",\n"
    << "      \"overflow_count\": " << s.overflow << ",\n"
    << "      \"elapsed_ms\": " << elapsed_ms << ",\n"
    << "      \"training_plus_elapsed_ms\": " << (training_ms+elapsed_ms) << ",\n"
    << "      \"us_per_point\": " << 1000.0*elapsed_ms/static_cast<double>(points) << ",\n"
    << "      \"us_per_sample\": " << 1000.0*elapsed_ms/static_cast<double>(points*wpp) << "\n"
    << "    }"; return o.str();
}

std::string two_level_json(const TwoLevelStats& s, int points, int coarse_wpp, int residual_wpp, float coarse_ms, float residual_ms, float training_ms=0.0f) {
  const float elapsed = coarse_ms + residual_ms;
  const int coarse_samples = points * coarse_wpp;
  const int residual_samples = points * residual_wpp;
  std::ostringstream o; o << std::setprecision(9) << "{\n"
    << "      \"eval_points\": " << points << ",\n"
    << "      \"coarse_walks_per_point\": " << coarse_wpp << ",\n"
    << "      \"residual_walks_per_point\": " << residual_wpp << ",\n"
    << "      \"coarse_samples\": " << coarse_samples << ",\n"
    << "      \"residual_samples\": " << residual_samples << ",\n"
    << "      \"rmse\": " << s.rmse << ",\n"
    << "      \"mae\": " << s.mae << ",\n"
    << "      \"max_abs_error\": " << s.max_abs_error << ",\n"
    << "      \"mean_estimate\": " << s.mean_estimate << ",\n"
    << "      \"mean_exact\": " << s.mean_exact << ",\n"
    << "      \"mean_bias\": " << (s.mean_estimate-s.mean_exact) << ",\n"
    << "      \"mean_coarse\": " << s.mean_coarse << ",\n"
    << "      \"mean_residual\": " << s.mean_residual << ",\n"
    << "      \"mean_coarse_sample_variance\": " << s.mean_coarse_sample_variance << ",\n"
    << "      \"mean_residual_sample_variance\": " << s.mean_residual_sample_variance << ",\n"
    << "      \"mean_coarse_steps\": " << s.mean_coarse_steps << ",\n"
    << "      \"mean_residual_steps\": " << s.mean_residual_steps << ",\n"
    << "      \"coarse_cache_queries\": " << s.coarse_cache_queries << ",\n"
    << "      \"residual_cache_queries\": " << s.residual_cache_queries << ",\n"
    << "      \"coarse_cache_query_fraction\": " << (coarse_samples>0?static_cast<double>(s.coarse_cache_queries)/static_cast<double>(coarse_samples):0.0) << ",\n"
    << "      \"residual_cache_query_fraction\": " << (residual_samples>0?static_cast<double>(s.residual_cache_queries)/static_cast<double>(residual_samples):0.0) << ",\n"
    << "      \"coarse_forced_max_steps\": " << s.coarse_forced << ",\n"
    << "      \"residual_forced_max_steps\": " << s.residual_forced << ",\n"
    << "      \"coarse_overflow_count\": " << s.coarse_overflow << ",\n"
    << "      \"residual_overflow_count\": " << s.residual_overflow << ",\n"
    << "      \"coarse_elapsed_ms\": " << coarse_ms << ",\n"
    << "      \"residual_elapsed_ms\": " << residual_ms << ",\n"
    << "      \"elapsed_ms\": " << elapsed << ",\n"
    << "      \"training_plus_elapsed_ms\": " << (training_ms+elapsed) << ",\n"
    << "      \"us_per_point\": " << 1000.0*elapsed/static_cast<double>(points) << ",\n"
    << "      \"us_per_effective_sample\": " << 1000.0*elapsed/static_cast<double>(coarse_samples+residual_samples) << "\n"
    << "    }"; return o.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt=parse(argc,argv); const std::string cmd=command_line(argc,argv);
    n2wos::Mesh mesh=load_mesh(opt); n2wos::NormalizeTransform norm{}; if(opt.normalize) norm=n2wos::normalize_to_unit_radius(mesh); const n2wos::Aabb bounds=n2wos::compute_bounds(mesh); const int deg=count_degenerate(mesh);
    n2wos::CuBqlBvh bvh(mesh,opt.cubql_leaf_size,opt.cubql_build_method);
    const int train_count=static_cast<int>(tcnn::next_multiple(static_cast<uint32_t>(opt.train_points), tcnn::BATCH_SIZE_GRANULARITY));
    const int hybrid_samples=opt.eval_points*opt.hybrid_walks_per_point; const int hybrid_padded=static_cast<int>(tcnn::next_multiple(static_cast<uint32_t>(hybrid_samples), tcnn::BATCH_SIZE_GRANULARITY));
    const int coarse_samples=opt.eval_points*opt.coarse_walks_per_point; const int coarse_padded=static_cast<int>(tcnn::next_multiple(static_cast<uint32_t>(coarse_samples), tcnn::BATCH_SIZE_GRANULARITY));
    const int residual_samples=opt.eval_points*opt.residual_walks_per_point; const int residual_padded=static_cast<int>(tcnn::next_multiple(static_cast<uint32_t>(residual_samples), tcnn::BATCH_SIZE_GRANULARITY));
    const std::vector<n2wos::DeviceVec3> train_points=make_ball_points(train_count,bounds,static_cast<std::uint64_t>(opt.seed));
    const std::vector<n2wos::DeviceVec3> eval_points=make_ball_points(opt.eval_points,bounds,static_cast<std::uint64_t>(opt.seed)^0x8da6b343ull);
    const n2wos::Vec3f in_min=input_min(bounds), in_extent=input_extent(bounds);

    cudaStream_t stream{}; N2WOS_CUDA_CHECK(cudaStreamCreate(&stream)); Timer timer;
    n2wos::DeviceVec3 *d_train=nullptr,*d_eval=nullptr; N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_train,sizeof(n2wos::DeviceVec3)*train_points.size())); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_eval,sizeof(n2wos::DeviceVec3)*eval_points.size()));
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(d_train,train_points.data(),sizeof(n2wos::DeviceVec3)*train_points.size(),cudaMemcpyHostToDevice,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(d_eval,eval_points.data(),sizeof(n2wos::DeviceVec3)*eval_points.size(),cudaMemcpyHostToDevice,stream));

    tcnn::json config={{"loss",{{"otype","L2"}}},{"optimizer",{{"otype","Adam"},{"learning_rate",opt.learning_rate},{"beta1",0.9f},{"beta2",0.99f},{"l2_reg",0.0f}}},{"encoding",{{"otype","HashGrid"},{"n_levels",opt.n_levels},{"n_features_per_level",opt.n_features_per_level},{"log2_hashmap_size",opt.log2_hashmap_size},{"base_resolution",opt.base_resolution},{"per_level_scale",opt.per_level_scale}}},{"network",{{"otype",opt.network},{"activation","ReLU"},{"output_activation","None"},{"n_neurons",opt.n_neurons},{"n_hidden_layers",opt.n_hidden_layers}}}};
    const tcnn::json enc=config.value("encoding",tcnn::json::object()), loss_opts=config.value("loss",tcnn::json::object()), opt_opts=config.value("optimizer",tcnn::json::object()), net_opts=config.value("network",tcnn::json::object());
    auto loss=std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(loss_opts)); auto optimizer=std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(opt_opts)); auto network=std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(3,1,enc,net_opts); const bool jit_supported=tcnn::supports_jit_fusion(); network->set_jit_fusion((opt.jit!=0)&&jit_supported); auto trainer=std::make_shared<tcnn::Trainer<float,precision_t,precision_t>>(network,optimizer,loss);
    tcnn::GPUMatrix<float> train_inputs(3,train_count), train_targets(1,train_count), prefix_inputs(3,hybrid_padded), cache_outputs(1,hybrid_padded), coarse_inputs(3,coarse_padded), coarse_cache_outputs(1,coarse_padded), residual_inputs(3,residual_padded), residual_cache_outputs(1,residual_padded);
    N2WOS_CUDA_CHECK(cudaMemsetAsync(train_targets.data(),0,sizeof(float)*train_count,stream));
    N2WOS_CUDA_CHECK(cudaMemsetAsync(prefix_inputs.data(),0,sizeof(float)*3*hybrid_padded,stream)); N2WOS_CUDA_CHECK(cudaMemsetAsync(cache_outputs.data(),0,sizeof(float)*hybrid_padded,stream));
    N2WOS_CUDA_CHECK(cudaMemsetAsync(coarse_inputs.data(),0,sizeof(float)*3*coarse_padded,stream)); N2WOS_CUDA_CHECK(cudaMemsetAsync(coarse_cache_outputs.data(),0,sizeof(float)*coarse_padded,stream));
    N2WOS_CUDA_CHECK(cudaMemsetAsync(residual_inputs.data(),0,sizeof(float)*3*residual_padded,stream)); N2WOS_CUDA_CHECK(cudaMemsetAsync(residual_cache_outputs.data(),0,sizeof(float)*residual_padded,stream));
    int* d_counts=nullptr; N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_counts,sizeof(int)*train_count)); N2WOS_CUDA_CHECK(cudaMemsetAsync(d_counts,0,sizeof(int)*train_count,stream));
    n2wos::NcDeviceDatasetOptions ds; ds.d_world_points=d_train; ds.point_count=train_count; ds.walks_per_point=opt.walks_per_label_refresh; ds.max_steps=opt.max_steps; ds.epsilon=opt.epsilon; ds.step_scale=opt.step_scale; ds.seed=opt.seed; ds.boundary_mode=opt.boundary_mode; ds.label_source=opt.label_source; ds.input_min=in_min; ds.input_extent=in_extent; ds.block_size=opt.block_size;
    float label_ms=0, train_ms=0; for(int r=0;r<opt.label_refreshes;++r){ ds.refresh_index=r; timer.begin(stream); n2wos::launch_nc_update_labels(bvh,ds,train_inputs.data(),train_targets.data(),d_counts,stream); label_ms += timer.end(stream); timer.begin(stream); for(int s=0;s<opt.train_steps_per_refresh;++s){ auto ctx=trainer->training_step(stream,train_inputs,train_targets); (void)ctx; } train_ms += timer.end(stream); }

    const int pure_samples=opt.eval_points*opt.pure_walks_per_point; float *d_pure=nullptr,*d_boundary=nullptr,*d_nc_values=nullptr,*d_dummy_residual=nullptr; int *d_pure_steps=nullptr,*d_pure_forced=nullptr,*d_pure_over=nullptr,*d_h_steps=nullptr,*d_h_forced=nullptr,*d_h_over=nullptr; unsigned char* d_need=nullptr;
    N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure,sizeof(float)*pure_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure_steps,sizeof(int)*pure_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure_forced,sizeof(int)*pure_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure_over,sizeof(int)*pure_samples));
    n2wos::NcDeviceSampleOptions so; so.d_eval_points=d_eval; so.eval_point_count=opt.eval_points; so.walks_per_point=opt.pure_walks_per_point; so.max_steps=opt.max_steps; so.epsilon=opt.epsilon; so.step_scale=opt.step_scale; so.seed=static_cast<std::uint64_t>(opt.seed)^0x44d2a123ull; so.boundary_mode=opt.boundary_mode; so.input_min=in_min; so.input_extent=in_extent; so.block_size=opt.block_size;
    timer.begin(stream); n2wos::launch_nc_pure_wos_samples(bvh,so,d_pure,d_pure_steps,d_pure_forced,d_pure_over,stream); const float pure_ms=timer.end(stream);

    N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_boundary,sizeof(float)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_need,sizeof(unsigned char)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_h_steps,sizeof(int)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_h_forced,sizeof(int)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_h_over,sizeof(int)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_nc_values,sizeof(float)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_dummy_residual,sizeof(float)*hybrid_samples));
    n2wos::NcDeviceSampleOptions ho=so; ho.walks_per_point=opt.hybrid_walks_per_point; ho.depth_m=opt.depth_m; ho.seed=static_cast<std::uint64_t>(opt.seed)^0x89abcdefull;
    timer.begin(stream); n2wos::launch_nc_hybrid_prefix(bvh,ho,prefix_inputs.data(),d_boundary,d_need,d_h_steps,d_h_forced,d_h_over,stream); network->inference(stream,prefix_inputs,cache_outputs); n2wos::launch_nc_combine_cache_and_residual(cache_outputs.data(),d_boundary,d_boundary,d_need,hybrid_samples,d_nc_values,d_dummy_residual,opt.block_size,stream); const float hybrid_ms=timer.end(stream);

    float coarse_ms=0.0f, residual_ms=0.0f;
    float *d_c_boundary=nullptr,*d_c_values=nullptr,*d_c_dummy_residual=nullptr,*d_r_boundary=nullptr,*d_r_continuation=nullptr,*d_r_nc_values=nullptr,*d_r_values=nullptr;
    int *d_c_steps=nullptr,*d_c_forced=nullptr,*d_c_over=nullptr,*d_r_steps=nullptr,*d_r_forced=nullptr,*d_r_over=nullptr;
    unsigned char *d_c_need=nullptr,*d_r_need=nullptr;
    if (opt.enable_2lmc) {
      N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_boundary,sizeof(float)*coarse_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_values,sizeof(float)*coarse_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_dummy_residual,sizeof(float)*coarse_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_need,sizeof(unsigned char)*coarse_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_steps,sizeof(int)*coarse_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_forced,sizeof(int)*coarse_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_c_over,sizeof(int)*coarse_samples));
      n2wos::NcDeviceSampleOptions co=so; co.walks_per_point=opt.coarse_walks_per_point; co.depth_m=opt.depth_m; co.seed=static_cast<std::uint64_t>(opt.seed)^0xf1357aea2e62a9c5ull;
      timer.begin(stream); n2wos::launch_nc_hybrid_prefix(bvh,co,coarse_inputs.data(),d_c_boundary,d_c_need,d_c_steps,d_c_forced,d_c_over,stream); network->inference(stream,coarse_inputs,coarse_cache_outputs); n2wos::launch_nc_combine_cache_and_residual(coarse_cache_outputs.data(),d_c_boundary,d_c_boundary,d_c_need,coarse_samples,d_c_values,d_c_dummy_residual,opt.block_size,stream); coarse_ms=timer.end(stream);

      N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_boundary,sizeof(float)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_continuation,sizeof(float)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_nc_values,sizeof(float)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_values,sizeof(float)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_need,sizeof(unsigned char)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_steps,sizeof(int)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_forced,sizeof(int)*residual_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_r_over,sizeof(int)*residual_samples));
      n2wos::NcDeviceSampleOptions ro=so; ro.walks_per_point=opt.residual_walks_per_point; ro.depth_m=opt.depth_m; ro.seed=static_cast<std::uint64_t>(opt.seed)^0x5f4dca4f0a2bbf27ull;
      timer.begin(stream); n2wos::launch_nc_2lmc_prefix_continue(bvh,ro,residual_inputs.data(),d_r_boundary,d_r_continuation,d_r_need,d_r_steps,d_r_forced,d_r_over,stream); network->inference(stream,residual_inputs,residual_cache_outputs); n2wos::launch_nc_combine_cache_and_residual(residual_cache_outputs.data(),d_r_boundary,d_r_continuation,d_r_need,residual_samples,d_r_nc_values,d_r_values,opt.block_size,stream); residual_ms=timer.end(stream);
    }

    std::vector<float> h_pure(pure_samples), h_hybrid(hybrid_samples); std::vector<int> h_ps(pure_samples),h_pf(pure_samples),h_po(pure_samples),h_hs(hybrid_samples),h_hf(hybrid_samples),h_ho(hybrid_samples); std::vector<unsigned char> h_need(hybrid_samples);
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_pure.data(),d_pure,sizeof(float)*pure_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_ps.data(),d_pure_steps,sizeof(int)*pure_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_pf.data(),d_pure_forced,sizeof(int)*pure_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_po.data(),d_pure_over,sizeof(int)*pure_samples,cudaMemcpyDeviceToHost,stream));
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_hybrid.data(),d_nc_values,sizeof(float)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_need.data(),d_need,sizeof(unsigned char)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_hs.data(),d_h_steps,sizeof(int)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_hf.data(),d_h_forced,sizeof(int)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_ho.data(),d_h_over,sizeof(int)*hybrid_samples,cudaMemcpyDeviceToHost,stream));
    std::vector<float> h_c_values, h_r_values; std::vector<int> h_cs,h_cf,h_co,h_rs,h_rf,h_ro; std::vector<unsigned char> h_c_need,h_r_need;
    if (opt.enable_2lmc) {
      h_c_values.resize(coarse_samples); h_r_values.resize(residual_samples); h_cs.resize(coarse_samples); h_cf.resize(coarse_samples); h_co.resize(coarse_samples); h_rs.resize(residual_samples); h_rf.resize(residual_samples); h_ro.resize(residual_samples); h_c_need.resize(coarse_samples); h_r_need.resize(residual_samples);
      N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_c_values.data(),d_c_values,sizeof(float)*coarse_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_cs.data(),d_c_steps,sizeof(int)*coarse_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_cf.data(),d_c_forced,sizeof(int)*coarse_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_co.data(),d_c_over,sizeof(int)*coarse_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_c_need.data(),d_c_need,sizeof(unsigned char)*coarse_samples,cudaMemcpyDeviceToHost,stream));
      N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_r_values.data(),d_r_values,sizeof(float)*residual_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_rs.data(),d_r_steps,sizeof(int)*residual_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_rf.data(),d_r_forced,sizeof(int)*residual_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_ro.data(),d_r_over,sizeof(int)*residual_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_r_need.data(),d_r_need,sizeof(unsigned char)*residual_samples,cudaMemcpyDeviceToHost,stream));
    }
    N2WOS_CUDA_CHECK(cudaStreamSynchronize(stream));
    const Stats pure=summarize(h_pure,h_ps,h_pf,h_po,nullptr,eval_points,opt.pure_walks_per_point,opt.boundary_mode); const Stats hybrid=summarize(h_hybrid,h_hs,h_hf,h_ho,&h_need,eval_points,opt.hybrid_walks_per_point,opt.boundary_mode);
    TwoLevelStats tl; if (opt.enable_2lmc) tl=summarize_two_level(h_c_values,h_r_values,h_cs,h_rs,h_cf,h_rf,h_co,h_ro,h_c_need,h_r_need,eval_points,opt.coarse_walks_per_point,opt.residual_walks_per_point,opt.boundary_mode);

    std::filesystem::path out_path(opt.output); if(!out_path.parent_path().empty()) std::filesystem::create_directories(out_path.parent_path()); std::ofstream out(opt.output); if(!out) throw std::runtime_error("failed to open output: "+opt.output); out << std::setprecision(9);
    out << "{\n  \"schema\": \"n2wos_tcnn_nc_wos_eval_v3\",\n  \"patch\": \"0008-extend-tcnn-nc-2lmc-depth-sweep\",\n  \"generated_at_utc\": " << n2wos::json_quote(now_utc()) << ",\n  \"command_line\": " << n2wos::json_quote(cmd) << ",\n  \"cuda\": " << cuda_json() << ",\n  \"implementation_mode\": {\n    \"solver\": \"neural_cache_wos_and_nc_2lmc_depth_sweep_screening\",\n    \"geometry_backend\": \"cubql_cuda\",\n    \"cache_backend\": \"tiny-cuda-nn_native_cpp\",\n    \"training_schedule\": \"fixed_refresh_count_and_fixed_train_steps\",\n    \"training_labels\": " << n2wos::json_quote(n2wos::nc_label_source_name(opt.label_source)) << ",\n    \"host_transfer_between_prefix_and_tcnn\": false,\n    \"host_transfer_between_tcnn_and_residual_consumer\": false,\n    \"csv_postprocess\": false,\n    \"python_bindings\": false,\n    \"nc_wos_biased\": true,\n    \"nc_2lmc_correction_enabled\": " << (opt.enable_2lmc?"true":"false") << ",\n    \"nc_and_2lmc_share_depth_m\": true,\n    \"point_sampler\": \"inscribed_ball_screening_sampler\",\n    \"production_status\": \"screening_diagnostic_not_final_wall_clock\"\n  },\n";
    out << "  \"options\": {\n    \"mesh\": " << n2wos::json_quote(opt.mesh) << ",\n    \"mesh_path\": " << n2wos::json_quote(opt.mesh_path) << ",\n    \"boundary_condition\": " << n2wos::json_quote(n2wos::nc_boundary_mode_name(opt.boundary_mode)) << ",\n    \"label_source\": " << n2wos::json_quote(n2wos::nc_label_source_name(opt.label_source)) << ",\n    \"train_points_requested\": " << opt.train_points << ",\n    \"train_points_padded\": " << train_count << ",\n    \"eval_points\": " << opt.eval_points << ",\n    \"label_refreshes\": " << opt.label_refreshes << ",\n    \"walks_per_label_refresh\": " << opt.walks_per_label_refresh << ",\n    \"train_steps_per_refresh\": " << opt.train_steps_per_refresh << ",\n    \"pure_walks_per_point\": " << opt.pure_walks_per_point << ",\n    \"hybrid_walks_per_point\": " << opt.hybrid_walks_per_point << ",\n    \"coarse_walks_per_point\": " << opt.coarse_walks_per_point << ",\n    \"residual_walks_per_point\": " << opt.residual_walks_per_point << ",\n    \"depth_m\": " << opt.depth_m << ",\n    \"max_steps\": " << opt.max_steps << ",\n    \"epsilon\": " << opt.epsilon << ",\n    \"cache_preset\": " << n2wos::json_quote(opt.cache_preset) << ",\n    \"network\": " << n2wos::json_quote(opt.network) << ",\n    \"n_levels\": " << opt.n_levels << ",\n    \"n_features_per_level\": " << opt.n_features_per_level << ",\n    \"log2_hashmap_size\": " << opt.log2_hashmap_size << ",\n    \"base_resolution\": " << opt.base_resolution << ",\n    \"per_level_scale\": " << opt.per_level_scale << ",\n    \"n_neurons\": " << opt.n_neurons << ",\n    \"n_hidden_layers\": " << opt.n_hidden_layers << ",\n    \"jit_requested\": " << (opt.jit?"true":"false") << ",\n    \"jit_supported\": " << (jit_supported?"true":"false") << ",\n    \"jit_enabled\": " << (network->jit_fusion()?"true":"false") << "\n  },\n";
    out << "  \"mesh_stats\": {\n    \"name\": " << n2wos::json_quote(mesh.name) << ",\n    \"vertices\": " << mesh.vertices.size() << ",\n    \"triangles\": " << mesh.triangles.size() << ",\n    \"degenerate_triangles\": " << deg << ",\n    \"bounds_min\": [" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << "],\n    \"bounds_max\": [" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << "],\n    \"normalization\": {\"center\": [" << norm.center.x << ", " << norm.center.y << ", " << norm.center.z << "], \"scale\": " << norm.scale << "}\n  },\n";
    out << "  \"bvh_stats\": {\n    \"triangles\": " << bvh.triangle_count() << ",\n    \"nodes\": " << bvh.node_count() << ",\n    \"prim_ids\": " << bvh.prim_id_count() << ",\n    \"leaf_size\": " << bvh.leaf_size() << ",\n    \"build_method\": " << n2wos::json_quote(bvh.build_method()) << ",\n    \"build_milliseconds\": " << bvh.build_milliseconds() << "\n  },\n";
    out << "  \"training\": {\n    \"label_update_ms\": " << label_ms << ",\n    \"tcnn_training_ms\": " << train_ms << ",\n    \"total_training_ms\": " << (label_ms+train_ms) << ",\n    \"host_readback_in_training_loop\": false\n  },\n";
    out << "  \"runs\": {\n    \"pure_wos\": " << stats_json(pure,opt.eval_points,opt.pure_walks_per_point,pure_ms) << ",\n    \"nc_wos\": " << stats_json(hybrid,opt.eval_points,opt.hybrid_walks_per_point,hybrid_ms,label_ms+train_ms);
    if (opt.enable_2lmc) {
      const std::string depth_key = "nc_2lmc_m" + std::to_string(opt.depth_m);
      const std::string two_json = two_level_json(tl,opt.eval_points,opt.coarse_walks_per_point,opt.residual_walks_per_point,coarse_ms,residual_ms,label_ms+train_ms);
      out << ",\n    \"nc_2lmc\": " << two_json << ",\n    \"" << depth_key << "\": " << two_json;
    }
    out << "\n  },\n";
    out << "  \"comparison\": {\n    \"nc_wos_rmse_div_pure_wos_rmse\": " << (pure.rmse>0?hybrid.rmse/pure.rmse:0) << ",\n";
    if (opt.enable_2lmc) out << "    \"nc_2lmc_rmse_div_pure_wos_rmse\": " << (pure.rmse>0?tl.rmse/pure.rmse:0) << ",\n    \"nc_2lmc_rmse_div_nc_wos_rmse\": " << (hybrid.rmse>0?tl.rmse/hybrid.rmse:0) << ",\n    \"nc_wos_mean_bias\": " << (hybrid.mean_estimate-hybrid.mean_exact) << ",\n    \"nc_2lmc_mean_bias\": " << (tl.mean_estimate-tl.mean_exact) << ",\n    \"pure_elapsed_div_nc_2lmc_inference_elapsed\": " << ((coarse_ms+residual_ms)>0?pure_ms/(coarse_ms+residual_ms):0) << ",\n    \"pure_elapsed_div_nc_2lmc_total_with_training\": " << ((coarse_ms+residual_ms+label_ms+train_ms)>0?pure_ms/(coarse_ms+residual_ms+label_ms+train_ms):0) << ",\n";
    out << "    \"pure_elapsed_div_nc_inference_elapsed\": " << (hybrid_ms>0?pure_ms/hybrid_ms:0) << ",\n    \"pure_elapsed_div_nc_total_with_training\": " << ((hybrid_ms+label_ms+train_ms)>0?pure_ms/(hybrid_ms+label_ms+train_ms):0) << ",\n    \"training_cost_counted_in_total_speedup\": true\n  }\n}\n"; out.close();

    cudaFree(d_train); cudaFree(d_eval); cudaFree(d_counts); cudaFree(d_pure); cudaFree(d_pure_steps); cudaFree(d_pure_forced); cudaFree(d_pure_over); cudaFree(d_boundary); cudaFree(d_need); cudaFree(d_h_steps); cudaFree(d_h_forced); cudaFree(d_h_over); cudaFree(d_nc_values); cudaFree(d_dummy_residual);
    if (opt.enable_2lmc) { cudaFree(d_c_boundary); cudaFree(d_c_values); cudaFree(d_c_dummy_residual); cudaFree(d_c_need); cudaFree(d_c_steps); cudaFree(d_c_forced); cudaFree(d_c_over); cudaFree(d_r_boundary); cudaFree(d_r_continuation); cudaFree(d_r_nc_values); cudaFree(d_r_values); cudaFree(d_r_need); cudaFree(d_r_steps); cudaFree(d_r_forced); cudaFree(d_r_over); }
    N2WOS_CUDA_CHECK(cudaStreamDestroy(stream)); tcnn::free_all_gpu_memory_arenas();
    std::cout << "wrote " << opt.output << "\n"; std::cout << "pure RMSE " << pure.rmse << " in " << pure_ms << " ms\n"; std::cout << "NC+WoS RMSE " << hybrid.rmse << " in " << hybrid_ms << " ms\n"; if (opt.enable_2lmc) std::cout << "NC+2LMC RMSE " << tl.rmse << " in " << (coarse_ms+residual_ms) << " ms\n"; return 0;
  } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
