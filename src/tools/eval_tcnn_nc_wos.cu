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
  int depth_m = 1;
  int max_steps = 256;
  float epsilon = 1.0e-4f;
  float step_scale = 0.999f;
  int seed = 12345;
  int block_size = 128;
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

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
  return argv[++i];
}

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --mesh procedural_bumpy_sphere|obj|ply\n"
            << "  --mesh-path <path>\n"
            << "  --bc harmonic_x2_minus_y2|external_charges_medium|external_charges_high\n"
            << "  --label-source wos_supervision|exact_analytic\n"
            << "  --train-points <int> --eval-points <int>\n"
            << "  --label-refreshes <int> --walks-per-label-refresh <int>\n"
            << "  --train-steps-per-refresh <int>\n"
            << "  --pure-walks-per-point <int> --hybrid-walks-per-point <int>\n"
            << "  --depth-m <int> --output <path>\n";
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
    else if (a == "--train-points") o.train_points = std::stoi(require_value(i,argc,argv));
    else if (a == "--eval-points") o.eval_points = std::stoi(require_value(i,argc,argv));
    else if (a == "--label-refreshes") o.label_refreshes = std::stoi(require_value(i,argc,argv));
    else if (a == "--walks-per-label-refresh") o.walks_per_label_refresh = std::stoi(require_value(i,argc,argv));
    else if (a == "--train-steps-per-refresh") o.train_steps_per_refresh = std::stoi(require_value(i,argc,argv));
    else if (a == "--pure-walks-per-point") o.pure_walks_per_point = std::stoi(require_value(i,argc,argv));
    else if (a == "--hybrid-walks-per-point") o.hybrid_walks_per_point = std::stoi(require_value(i,argc,argv));
    else if (a == "--depth-m") o.depth_m = std::stoi(require_value(i,argc,argv));
    else if (a == "--max-steps") o.max_steps = std::stoi(require_value(i,argc,argv));
    else if (a == "--epsilon") o.epsilon = std::stof(require_value(i,argc,argv));
    else if (a == "--step-scale") o.step_scale = std::stof(require_value(i,argc,argv));
    else if (a == "--seed") o.seed = std::stoi(require_value(i,argc,argv));
    else if (a == "--block-size") o.block_size = std::stoi(require_value(i,argc,argv));
    else if (a == "--network") o.network = require_value(i,argc,argv);
    else if (a == "--n-levels") o.n_levels = std::stoi(require_value(i,argc,argv));
    else if (a == "--n-features-per-level") o.n_features_per_level = std::stoi(require_value(i,argc,argv));
    else if (a == "--log2-hashmap-size") o.log2_hashmap_size = std::stoi(require_value(i,argc,argv));
    else if (a == "--base-resolution") o.base_resolution = std::stoi(require_value(i,argc,argv));
    else if (a == "--per-level-scale") o.per_level_scale = std::stof(require_value(i,argc,argv));
    else if (a == "--n-neurons") o.n_neurons = std::stoi(require_value(i,argc,argv));
    else if (a == "--n-hidden-layers") o.n_hidden_layers = std::stoi(require_value(i,argc,argv));
    else if (a == "--learning-rate") o.learning_rate = std::stof(require_value(i,argc,argv));
    else if (a == "--jit") o.jit = std::stoi(require_value(i,argc,argv));
    else if (a == "--output") o.output = require_value(i,argc,argv);
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.train_points <= 0 || o.eval_points <= 0 || o.pure_walks_per_point <= 0 || o.hybrid_walks_per_point <= 0) throw std::runtime_error("invalid sample counts");
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

Stats summarize(const std::vector<float>& values, const std::vector<int>& steps, const std::vector<int>& forced, const std::vector<int>& overflow, const std::vector<unsigned char>* needs_cache, const std::vector<n2wos::DeviceVec3>& points, int wpp, n2wos::NcBoundaryMode bc) {
  Stats out; const int n=static_cast<int>(points.size()); double sum_mse=0, sum_abs=0, sum_var=0, sum_est=0, sum_exact=0, sum_steps=0;
  for (int p=0;p<n;++p) { double s=0, ss=0, st=0; for (int w=0;w<wpp;++w){ const int idx=p*wpp+w; const double v=values[idx]; s+=v; ss+=v*v; st+=steps[idx]; out.forced += forced[idx]?1ull:0ull; out.overflow += overflow[idx]?1ull:0ull; if(needs_cache && (*needs_cache)[idx]) ++out.cache_queries; } const double mean=s/static_cast<double>(wpp); const n2wos::Vec3f hp{points[p].x,points[p].y,points[p].z}; const double exact=n2wos::nc_boundary_value_host(hp,bc); const double err=mean-exact; const double centered=ss-static_cast<double>(wpp)*mean*mean; sum_var += wpp>1 ? std::max(0.0,centered)/static_cast<double>(wpp-1) : 0.0; sum_mse+=err*err; sum_abs+=std::fabs(err); out.max_abs_error=std::max(out.max_abs_error,std::fabs(err)); sum_est+=mean; sum_exact+=exact; sum_steps += st/static_cast<double>(wpp); }
  const double dn=static_cast<double>(n); out.mse=sum_mse/dn; out.rmse=std::sqrt(out.mse); out.mae=sum_abs/dn; out.mean_sample_variance=sum_var/dn; out.mean_estimate=sum_est/dn; out.mean_exact=sum_exact/dn; out.mean_steps=sum_steps/dn; return out;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt=parse(argc,argv); const std::string cmd=command_line(argc,argv);
    n2wos::Mesh mesh=load_mesh(opt); n2wos::NormalizeTransform norm{}; if(opt.normalize) norm=n2wos::normalize_to_unit_radius(mesh); const n2wos::Aabb bounds=n2wos::compute_bounds(mesh); const int deg=count_degenerate(mesh);
    n2wos::CuBqlBvh bvh(mesh,opt.cubql_leaf_size,opt.cubql_build_method);
    const int train_count=static_cast<int>(tcnn::next_multiple(static_cast<uint32_t>(opt.train_points), tcnn::BATCH_SIZE_GRANULARITY));
    const int hybrid_samples=opt.eval_points*opt.hybrid_walks_per_point; const int hybrid_padded=static_cast<int>(tcnn::next_multiple(static_cast<uint32_t>(hybrid_samples), tcnn::BATCH_SIZE_GRANULARITY));
    const std::vector<n2wos::DeviceVec3> train_points=make_ball_points(train_count,bounds,static_cast<std::uint64_t>(opt.seed));
    const std::vector<n2wos::DeviceVec3> eval_points=make_ball_points(opt.eval_points,bounds,static_cast<std::uint64_t>(opt.seed)^0x8da6b343ull);
    const n2wos::Vec3f in_min=input_min(bounds), in_extent=input_extent(bounds);

    cudaStream_t stream{}; N2WOS_CUDA_CHECK(cudaStreamCreate(&stream)); Timer timer;
    n2wos::DeviceVec3 *d_train=nullptr,*d_eval=nullptr; N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_train,sizeof(n2wos::DeviceVec3)*train_points.size())); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_eval,sizeof(n2wos::DeviceVec3)*eval_points.size()));
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(d_train,train_points.data(),sizeof(n2wos::DeviceVec3)*train_points.size(),cudaMemcpyHostToDevice,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(d_eval,eval_points.data(),sizeof(n2wos::DeviceVec3)*eval_points.size(),cudaMemcpyHostToDevice,stream));

    tcnn::json config={{"loss",{{"otype","L2"}}},{"optimizer",{{"otype","Adam"},{"learning_rate",opt.learning_rate},{"beta1",0.9f},{"beta2",0.99f},{"l2_reg",0.0f}}},{"encoding",{{"otype","HashGrid"},{"n_levels",opt.n_levels},{"n_features_per_level",opt.n_features_per_level},{"log2_hashmap_size",opt.log2_hashmap_size},{"base_resolution",opt.base_resolution},{"per_level_scale",opt.per_level_scale}}},{"network",{{"otype",opt.network},{"activation","ReLU"},{"output_activation","None"},{"n_neurons",opt.n_neurons},{"n_hidden_layers",opt.n_hidden_layers}}}};
    const tcnn::json enc=config.value("encoding",tcnn::json::object()), loss_opts=config.value("loss",tcnn::json::object()), opt_opts=config.value("optimizer",tcnn::json::object()), net_opts=config.value("network",tcnn::json::object());
    auto loss=std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(loss_opts)); auto optimizer=std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(opt_opts)); auto network=std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(3,1,enc,net_opts); const bool jit_supported=tcnn::supports_jit_fusion(); network->set_jit_fusion((opt.jit!=0)&&jit_supported); auto trainer=std::make_shared<tcnn::Trainer<float,precision_t,precision_t>>(network,optimizer,loss);
    tcnn::GPUMatrix<float> train_inputs(3,train_count), train_targets(1,train_count), prefix_inputs(3,hybrid_padded), cache_outputs(1,hybrid_padded); N2WOS_CUDA_CHECK(cudaMemsetAsync(train_targets.data(),0,sizeof(float)*train_count,stream)); N2WOS_CUDA_CHECK(cudaMemsetAsync(prefix_inputs.data(),0,sizeof(float)*3*hybrid_padded,stream)); N2WOS_CUDA_CHECK(cudaMemsetAsync(cache_outputs.data(),0,sizeof(float)*hybrid_padded,stream));
    int* d_counts=nullptr; N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_counts,sizeof(int)*train_count)); N2WOS_CUDA_CHECK(cudaMemsetAsync(d_counts,0,sizeof(int)*train_count,stream));
    n2wos::NcDeviceDatasetOptions ds; ds.d_world_points=d_train; ds.point_count=train_count; ds.walks_per_point=opt.walks_per_label_refresh; ds.max_steps=opt.max_steps; ds.epsilon=opt.epsilon; ds.step_scale=opt.step_scale; ds.seed=opt.seed; ds.boundary_mode=opt.boundary_mode; ds.label_source=opt.label_source; ds.input_min=in_min; ds.input_extent=in_extent; ds.block_size=opt.block_size;
    float label_ms=0, train_ms=0; for(int r=0;r<opt.label_refreshes;++r){ ds.refresh_index=r; timer.begin(stream); n2wos::launch_nc_update_labels(bvh,ds,train_inputs.data(),train_targets.data(),d_counts,stream); label_ms += timer.end(stream); timer.begin(stream); for(int s=0;s<opt.train_steps_per_refresh;++s){ auto ctx=trainer->training_step(stream,train_inputs,train_targets); (void)ctx; } train_ms += timer.end(stream); }

    const int pure_samples=opt.eval_points*opt.pure_walks_per_point; float *d_pure=nullptr,*d_boundary=nullptr; int *d_pure_steps=nullptr,*d_pure_forced=nullptr,*d_pure_over=nullptr,*d_h_steps=nullptr,*d_h_forced=nullptr,*d_h_over=nullptr; unsigned char* d_need=nullptr;
    N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure,sizeof(float)*pure_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure_steps,sizeof(int)*pure_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure_forced,sizeof(int)*pure_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_pure_over,sizeof(int)*pure_samples));
    n2wos::NcDeviceSampleOptions so; so.d_eval_points=d_eval; so.eval_point_count=opt.eval_points; so.walks_per_point=opt.pure_walks_per_point; so.max_steps=opt.max_steps; so.epsilon=opt.epsilon; so.step_scale=opt.step_scale; so.seed=static_cast<std::uint64_t>(opt.seed)^0x44d2a123ull; so.boundary_mode=opt.boundary_mode; so.input_min=in_min; so.input_extent=in_extent; so.block_size=opt.block_size;
    timer.begin(stream); n2wos::launch_nc_pure_wos_samples(bvh,so,d_pure,d_pure_steps,d_pure_forced,d_pure_over,stream); const float pure_ms=timer.end(stream);
    N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_boundary,sizeof(float)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_need,sizeof(unsigned char)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_h_steps,sizeof(int)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_h_forced,sizeof(int)*hybrid_samples)); N2WOS_CUDA_CHECK(cudaMalloc((void**)&d_h_over,sizeof(int)*hybrid_samples));
    n2wos::NcDeviceSampleOptions ho=so; ho.walks_per_point=opt.hybrid_walks_per_point; ho.depth_m=opt.depth_m; ho.seed=static_cast<std::uint64_t>(opt.seed)^0x89abcdefull;
    timer.begin(stream); n2wos::launch_nc_hybrid_prefix(bvh,ho,prefix_inputs.data(),d_boundary,d_need,d_h_steps,d_h_forced,d_h_over,stream); network->inference(stream,prefix_inputs,cache_outputs); const float hybrid_ms=timer.end(stream);

    std::vector<float> h_pure(pure_samples), h_boundary(hybrid_samples), h_cache(hybrid_samples), h_hybrid(hybrid_samples); std::vector<int> h_ps(pure_samples),h_pf(pure_samples),h_po(pure_samples),h_hs(hybrid_samples),h_hf(hybrid_samples),h_ho(hybrid_samples); std::vector<unsigned char> h_need(hybrid_samples);
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_pure.data(),d_pure,sizeof(float)*pure_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_ps.data(),d_pure_steps,sizeof(int)*pure_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_pf.data(),d_pure_forced,sizeof(int)*pure_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_po.data(),d_pure_over,sizeof(int)*pure_samples,cudaMemcpyDeviceToHost,stream));
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_boundary.data(),d_boundary,sizeof(float)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_cache.data(),cache_outputs.data(),sizeof(float)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_need.data(),d_need,sizeof(unsigned char)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_hs.data(),d_h_steps,sizeof(int)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_hf.data(),d_h_forced,sizeof(int)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaMemcpyAsync(h_ho.data(),d_h_over,sizeof(int)*hybrid_samples,cudaMemcpyDeviceToHost,stream)); N2WOS_CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int i=0;i<hybrid_samples;++i) h_hybrid[i]=h_need[i]?h_cache[i]:h_boundary[i];
    const Stats pure=summarize(h_pure,h_ps,h_pf,h_po,nullptr,eval_points,opt.pure_walks_per_point,opt.boundary_mode); const Stats hybrid=summarize(h_hybrid,h_hs,h_hf,h_ho,&h_need,eval_points,opt.hybrid_walks_per_point,opt.boundary_mode);

    std::filesystem::path out_path(opt.output); if(!out_path.parent_path().empty()) std::filesystem::create_directories(out_path.parent_path()); std::ofstream out(opt.output); if(!out) throw std::runtime_error("failed to open output: "+opt.output); out << std::setprecision(9);
    out << "{\n  \"schema\": \"n2wos_tcnn_nc_wos_eval_v1\",\n  \"patch\": \"0005-add-tcnn-nc-wos-highfreq-probe\",\n  \"generated_at_utc\": " << n2wos::json_quote(now_utc()) << ",\n  \"command_line\": " << n2wos::json_quote(cmd) << ",\n  \"cuda\": " << cuda_json() << ",\n  \"implementation_mode\": {\n    \"solver\": \"biased_neural_cache_wos_screening\",\n    \"geometry_backend\": \"cubql_cuda\",\n    \"cache_backend\": \"tiny-cuda-nn_native_cpp\",\n    \"training_schedule\": \"fixed_refresh_count_and_fixed_train_steps\",\n    \"training_labels\": " << n2wos::json_quote(n2wos::nc_label_source_name(opt.label_source)) << ",\n    \"host_transfer_between_prefix_and_tcnn\": false,\n    \"csv_postprocess\": false,\n    \"python_bindings\": false,\n    \"unbiased_2lmc_correction\": false,\n    \"point_sampler\": \"inscribed_ball_screening_sampler\",\n    \"production_status\": \"screening_diagnostic_not_final_wall_clock\"\n  },\n";
    out << "  \"options\": {\n    \"mesh\": " << n2wos::json_quote(opt.mesh) << ",\n    \"mesh_path\": " << n2wos::json_quote(opt.mesh_path) << ",\n    \"boundary_condition\": " << n2wos::json_quote(n2wos::nc_boundary_mode_name(opt.boundary_mode)) << ",\n    \"label_source\": " << n2wos::json_quote(n2wos::nc_label_source_name(opt.label_source)) << ",\n    \"train_points_requested\": " << opt.train_points << ",\n    \"train_points_padded\": " << train_count << ",\n    \"eval_points\": " << opt.eval_points << ",\n    \"label_refreshes\": " << opt.label_refreshes << ",\n    \"walks_per_label_refresh\": " << opt.walks_per_label_refresh << ",\n    \"train_steps_per_refresh\": " << opt.train_steps_per_refresh << ",\n    \"pure_walks_per_point\": " << opt.pure_walks_per_point << ",\n    \"hybrid_walks_per_point\": " << opt.hybrid_walks_per_point << ",\n    \"depth_m\": " << opt.depth_m << ",\n    \"max_steps\": " << opt.max_steps << ",\n    \"epsilon\": " << opt.epsilon << ",\n    \"network\": " << n2wos::json_quote(opt.network) << ",\n    \"jit_requested\": " << (opt.jit?"true":"false") << ",\n    \"jit_supported\": " << (jit_supported?"true":"false") << ",\n    \"jit_enabled\": " << (network->jit_fusion()?"true":"false") << "\n  },\n";
    out << "  \"mesh_stats\": {\n    \"name\": " << n2wos::json_quote(mesh.name) << ",\n    \"vertices\": " << mesh.vertices.size() << ",\n    \"triangles\": " << mesh.triangles.size() << ",\n    \"degenerate_triangles\": " << deg << ",\n    \"bounds_min\": [" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << "],\n    \"bounds_max\": [" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << "],\n    \"normalization\": {\"center\": [" << norm.center.x << ", " << norm.center.y << ", " << norm.center.z << "], \"scale\": " << norm.scale << "}\n  },\n";
    out << "  \"bvh_stats\": {\n    \"triangles\": " << bvh.triangle_count() << ",\n    \"nodes\": " << bvh.node_count() << ",\n    \"prim_ids\": " << bvh.prim_id_count() << ",\n    \"leaf_size\": " << bvh.leaf_size() << ",\n    \"build_method\": " << n2wos::json_quote(bvh.build_method()) << ",\n    \"build_milliseconds\": " << bvh.build_milliseconds() << "\n  },\n";
    out << "  \"training\": {\n    \"label_update_ms\": " << label_ms << ",\n    \"tcnn_training_ms\": " << train_ms << ",\n    \"total_training_ms\": " << (label_ms+train_ms) << ",\n    \"host_readback_in_training_loop\": false\n  },\n";
    out << "  \"runs\": {\n    \"pure_wos\": " << stats_json(pure,opt.eval_points,opt.pure_walks_per_point,pure_ms) << ",\n    \"nc_wos\": " << stats_json(hybrid,opt.eval_points,opt.hybrid_walks_per_point,hybrid_ms,label_ms+train_ms) << "\n  },\n";
    out << "  \"comparison\": {\n    \"nc_wos_rmse_div_pure_wos_rmse\": " << (pure.rmse>0?hybrid.rmse/pure.rmse:0) << ",\n    \"pure_elapsed_div_nc_inference_elapsed\": " << (hybrid_ms>0?pure_ms/hybrid_ms:0) << ",\n    \"pure_elapsed_div_nc_total_with_training\": " << ((hybrid_ms+label_ms+train_ms)>0?pure_ms/(hybrid_ms+label_ms+train_ms):0) << ",\n    \"training_cost_counted_in_total_speedup\": true\n  }\n}\n"; out.close();

    cudaFree(d_train); cudaFree(d_eval); cudaFree(d_counts); cudaFree(d_pure); cudaFree(d_pure_steps); cudaFree(d_pure_forced); cudaFree(d_pure_over); cudaFree(d_boundary); cudaFree(d_need); cudaFree(d_h_steps); cudaFree(d_h_forced); cudaFree(d_h_over); N2WOS_CUDA_CHECK(cudaStreamDestroy(stream)); tcnn::free_all_gpu_memory_arenas();
    std::cout << "wrote " << opt.output << "\n"; std::cout << "pure RMSE " << pure.rmse << " in " << pure_ms << " ms\n"; std::cout << "NC+WoS RMSE " << hybrid.rmse << " in " << hybrid_ms << " ms\n"; return 0;
  } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
