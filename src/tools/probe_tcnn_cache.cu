#include "n2wos/json.hpp"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>

#include <cuda_runtime.h>

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
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
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

using precision_t = tcnn::network_precision_t;

struct Options {
  int samples = 262144;
  int batch_size = 262144;
  int train_steps = 200;
  int repeat = 20;
  int warmup = 3;
  int seed = 12345;
  int n_levels = 12;
  int n_features_per_level = 2;
  int log2_hashmap_size = 18;
  int base_resolution = 8;
  float per_level_scale = 1.5f;
  int n_neurons = 32;
  int n_hidden_layers = 2;
  float learning_rate = 1.0e-2f;
  int jit = 0;
  std::string network = "FullyFusedMLP";
  std::string output = "results/probe_tcnn_cache.json";
};

struct TimingStats {
  std::vector<float> ms;
  float median_ms = 0.0f;
  float mean_ms = 0.0f;
};

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[i]);
  }
  return argv[++i];
}

void print_usage(std::ostream& out, const char* argv0) {
  out << "Usage: " << argv0 << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --samples <int>                Inference sample count [262144]\n"
      << "  --batch-size <int>             Training batch size [262144]\n"
      << "  --train-steps <int>            TCNN training steps [200]\n"
      << "  --repeat <int>                 Repeated inference timings [20]\n"
      << "  --warmup <int>                 Untimed inference warmups [3]\n"
      << "  --seed <int>                   GPU sample-generation seed [12345]\n"
      << "  --network FullyFusedMLP|CutlassMLP [FullyFusedMLP]\n"
      << "  --n-levels <int>               HashGrid levels [12]\n"
      << "  --n-features-per-level <int>   HashGrid features per level [2]\n"
      << "  --log2-hashmap-size <int>      HashGrid log2 hashmap size [18]\n"
      << "  --base-resolution <int>        HashGrid base resolution [8]\n"
      << "  --per-level-scale <float>      HashGrid scale per level [1.5]\n"
      << "  --n-neurons <int>              MLP neurons per hidden layer [32]\n"
      << "  --n-hidden-layers <int>        MLP hidden layers [2]\n"
      << "  --learning-rate <float>        Adam learning rate [1e-2]\n"
      << "  --jit 0|1                      Request TCNN JIT fusion if available [0]\n"
      << "  --output <path>                JSON output [results/probe_tcnn_cache.json]\n"
      << "  --help\n";
}

Options parse_options(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    } else if (arg == "--samples") {
      opt.samples = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--batch-size") {
      opt.batch_size = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--train-steps") {
      opt.train_steps = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--repeat") {
      opt.repeat = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--warmup") {
      opt.warmup = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--seed") {
      opt.seed = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--network") {
      opt.network = require_value(i, argc, argv);
    } else if (arg == "--n-levels") {
      opt.n_levels = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--n-features-per-level") {
      opt.n_features_per_level = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--log2-hashmap-size") {
      opt.log2_hashmap_size = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--base-resolution") {
      opt.base_resolution = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--per-level-scale") {
      opt.per_level_scale = std::stof(require_value(i, argc, argv));
    } else if (arg == "--n-neurons") {
      opt.n_neurons = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--n-hidden-layers") {
      opt.n_hidden_layers = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--learning-rate") {
      opt.learning_rate = std::stof(require_value(i, argc, argv));
    } else if (arg == "--jit") {
      opt.jit = std::stoi(require_value(i, argc, argv));
    } else if (arg == "--output") {
      opt.output = require_value(i, argc, argv);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opt.samples <= 0 || opt.batch_size <= 0 || opt.train_steps < 0 || opt.repeat <= 0 || opt.warmup < 0) {
    throw std::runtime_error("samples, batch-size, repeat must be positive; train-steps and warmup must be nonnegative");
  }
  if (opt.network != "FullyFusedMLP" && opt.network != "CutlassMLP") {
    throw std::runtime_error("--network must be FullyFusedMLP or CutlassMLP");
  }
  return opt;
}

std::string now_utc_string() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buffer[64];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

std::string join_command_line(int argc, char** argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i) out << ' ';
    out << argv[i];
  }
  return out.str();
}

TimingStats summarize(std::vector<float> values) {
  TimingStats stats;
  stats.ms = std::move(values);
  if (stats.ms.empty()) return stats;
  std::vector<float> sorted = stats.ms;
  std::sort(sorted.begin(), sorted.end());
  if (sorted.size() % 2 == 0) {
    stats.median_ms = 0.5f * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
  } else {
    stats.median_ms = sorted[sorted.size() / 2];
  }
  stats.mean_ms = std::accumulate(stats.ms.begin(), stats.ms.end(), 0.0f) / static_cast<float>(stats.ms.size());
  return stats;
}

std::string vector_json(const std::vector<float>& v) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) out << ", ";
    out << std::setprecision(9) << v[i];
  }
  out << ']';
  return out.str();
}

struct EventTimer {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  EventTimer() {
    N2WOS_CUDA_CHECK(cudaEventCreate(&start));
    N2WOS_CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~EventTimer() {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
  }
  void record_start(cudaStream_t stream) { N2WOS_CUDA_CHECK(cudaEventRecord(start, stream)); }
  float record_stop_ms(cudaStream_t stream) {
    N2WOS_CUDA_CHECK(cudaEventRecord(stop, stream));
    N2WOS_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    N2WOS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ float unit_float(uint32_t x) {
  return static_cast<float>((x >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ float harmonic_target_from_unit(float ux, float uy) {
  const float x = 2.0f * ux - 1.0f;
  const float y = 2.0f * uy - 1.0f;
  return x * x - y * y;
}

__global__ void fill_points_and_targets_kernel(
    uint32_t n,
    uint32_t seed,
    uint32_t step,
    float* __restrict__ inputs,
    float* __restrict__ targets) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t h0 = mix_u32(seed ^ (0x9e3779b9u * (i + 1u)) ^ (0x85ebca6bu * (step + 1u)));
  const uint32_t h1 = mix_u32(h0 ^ 0x243f6a88u);
  const uint32_t h2 = mix_u32(h0 ^ 0xb7e15162u);
  const float ux = unit_float(h0);
  const float uy = unit_float(h1);
  const float uz = unit_float(h2);
  const uint32_t base = 3u * i;
  inputs[base + 0u] = ux;
  inputs[base + 1u] = uy;
  inputs[base + 2u] = uz;
  targets[i] = harmonic_target_from_unit(ux, uy);
}

__global__ void consume_cache_outputs_kernel(
    uint32_t n,
    uint32_t repeat_id,
    const float* __restrict__ inputs,
    const float* __restrict__ outputs,
    float* __restrict__ stats) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t base = 3u * i;
  const float exact = harmonic_target_from_unit(inputs[base + 0u], inputs[base + 1u]);
  const float pred = outputs[i];
  const float err = pred - exact;
  float* s = stats + 4u * repeat_id;
  atomicAdd(&s[0], pred);
  atomicAdd(&s[1], exact);
  atomicAdd(&s[2], err * err);
  atomicAdd(&s[3], fabsf(err));
}

void launch_fill(uint32_t n, uint32_t seed, uint32_t step, float* inputs, float* targets, cudaStream_t stream) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  fill_points_and_targets_kernel<<<grid, block, 0, stream>>>(n, seed, step, inputs, targets);
  N2WOS_CUDA_CHECK(cudaGetLastError());
}

void launch_consume(uint32_t n, uint32_t repeat_id, const float* inputs, const float* outputs, float* stats, cudaStream_t stream) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  consume_cache_outputs_kernel<<<grid, block, 0, stream>>>(n, repeat_id, inputs, outputs, stats);
  N2WOS_CUDA_CHECK(cudaGetLastError());
}

std::string gpu_info_json() {
  int device = 0;
  N2WOS_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  N2WOS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime_version = 0;
  int driver_version = 0;
  N2WOS_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  N2WOS_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  std::ostringstream out;
  out << "{\n"
      << "    \"runtime\": \"runtime=" << runtime_version << ",driver=" << driver_version << "\",\n"
      << "    \"device\": \"" << n2wos::json_escape(prop.name) << " sm_" << prop.major << prop.minor
      << " global_mem=" << static_cast<unsigned long long>(prop.totalGlobalMem) << "\"\n"
      << "  }";
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string command_line = join_command_line(argc, argv);
  try {
    const Options opt = parse_options(argc, argv);

    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = 1;
    const uint32_t samples = static_cast<uint32_t>(opt.samples);
    const uint32_t inference_batch = tcnn::next_multiple(samples, tcnn::BATCH_SIZE_GRANULARITY);
    const uint32_t training_batch = tcnn::next_multiple(static_cast<uint32_t>(opt.batch_size), tcnn::BATCH_SIZE_GRANULARITY);

    cudaStream_t stream{};
    N2WOS_CUDA_CHECK(cudaStreamCreate(&stream));

    tcnn::json config = {
        {"loss", {{"otype", "L2"}}},
        {"optimizer", {{"otype", "Adam"}, {"learning_rate", opt.learning_rate}, {"beta1", 0.9f}, {"beta2", 0.99f}, {"l2_reg", 0.0f}}},
        {"encoding", {{"otype", "HashGrid"}, {"n_levels", opt.n_levels}, {"n_features_per_level", opt.n_features_per_level}, {"log2_hashmap_size", opt.log2_hashmap_size}, {"base_resolution", opt.base_resolution}, {"per_level_scale", opt.per_level_scale}}},
        {"network", {{"otype", opt.network}, {"activation", "ReLU"}, {"output_activation", "None"}, {"n_neurons", opt.n_neurons}, {"n_hidden_layers", opt.n_hidden_layers}}},
    };

    tcnn::GPUMatrix<float> train_inputs(n_input_dims, training_batch);
    tcnn::GPUMatrix<float> train_targets(n_output_dims, training_batch);
    tcnn::GPUMatrix<float> infer_inputs(n_input_dims, inference_batch);
    tcnn::GPUMatrix<float> infer_outputs(n_output_dims, inference_batch);
    tcnn::GPUMemory<float> downstream_stats(static_cast<size_t>(4 * opt.repeat));
    N2WOS_CUDA_CHECK(cudaMemsetAsync(downstream_stats.data(), 0, sizeof(float) * 4ull * static_cast<unsigned long long>(opt.repeat), stream));

    launch_fill(training_batch, static_cast<uint32_t>(opt.seed), 0, train_inputs.data(), train_targets.data(), stream);
    launch_fill(inference_batch, static_cast<uint32_t>(opt.seed) ^ 0xa341316cu, 0, infer_inputs.data(), infer_outputs.data(), stream);

    const tcnn::json encoding_opts = config.value("encoding", tcnn::json::object());
    const tcnn::json loss_opts = config.value("loss", tcnn::json::object());
    const tcnn::json optimizer_opts = config.value("optimizer", tcnn::json::object());
    const tcnn::json network_opts = config.value("network", tcnn::json::object());

    std::shared_ptr<tcnn::Loss<precision_t>> loss{tcnn::create_loss<precision_t>(loss_opts)};
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizer_opts)};
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network =
        std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

    const bool requested_jit = opt.jit != 0;
    const bool jit_supported = tcnn::supports_jit_fusion();
    network->set_jit_fusion(requested_jit && jit_supported);
    const bool jit_enabled = network->jit_fusion();

    auto trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

    EventTimer timer;
    float train_ms = 0.0f;
    if (opt.train_steps > 0) {
      timer.record_start(stream);
      for (int step = 0; step < opt.train_steps; ++step) {
        launch_fill(training_batch, static_cast<uint32_t>(opt.seed), static_cast<uint32_t>(step), train_inputs.data(), train_targets.data(), stream);
        auto ctx = trainer->training_step(stream, train_inputs, train_targets);
        (void)ctx;
      }
      train_ms = timer.record_stop_ms(stream);
    }

    for (int i = 0; i < opt.warmup; ++i) {
      network->inference(stream, infer_inputs, infer_outputs);
    }
    N2WOS_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> inference_ms;
    inference_ms.reserve(static_cast<std::size_t>(opt.repeat));
    for (int r = 0; r < opt.repeat; ++r) {
      timer.record_start(stream);
      network->inference(stream, infer_inputs, infer_outputs);
      inference_ms.push_back(timer.record_stop_ms(stream));
    }

    std::vector<float> inference_consume_ms;
    inference_consume_ms.reserve(static_cast<std::size_t>(opt.repeat));
    for (int r = 0; r < opt.repeat; ++r) {
      timer.record_start(stream);
      network->inference(stream, infer_inputs, infer_outputs);
      launch_consume(samples, static_cast<uint32_t>(r), infer_inputs.data(), infer_outputs.data(), downstream_stats.data(), stream);
      inference_consume_ms.push_back(timer.record_stop_ms(stream));
    }

    std::vector<float> host_stats(static_cast<std::size_t>(4 * opt.repeat));
    N2WOS_CUDA_CHECK(cudaMemcpyAsync(host_stats.data(), downstream_stats.data(), sizeof(float) * host_stats.size(), cudaMemcpyDeviceToHost, stream));
    N2WOS_CUDA_CHECK(cudaStreamSynchronize(stream));

    double sum_pred = 0.0;
    double sum_exact = 0.0;
    double sum_sse = 0.0;
    double sum_abs = 0.0;
    for (int r = 0; r < opt.repeat; ++r) {
      sum_pred += host_stats[4 * r + 0];
      sum_exact += host_stats[4 * r + 1];
      sum_sse += host_stats[4 * r + 2];
      sum_abs += host_stats[4 * r + 3];
    }
    const double denom = static_cast<double>(samples) * static_cast<double>(opt.repeat);
    const double mse = sum_sse / denom;
    const double rmse = std::sqrt(mse);
    const double mae = sum_abs / denom;
    const double mean_pred = sum_pred / denom;
    const double mean_exact = sum_exact / denom;

    const TimingStats inference_stats = summarize(inference_ms);
    const TimingStats inference_consume_stats = summarize(inference_consume_ms);

    const double median_us_per_sample = 1000.0 * static_cast<double>(inference_stats.median_ms) / static_cast<double>(inference_batch);
    const double median_mqueries_per_second = static_cast<double>(inference_batch) / (1000.0 * static_cast<double>(inference_stats.median_ms));
    const double median_consume_us_per_sample = 1000.0 * static_cast<double>(inference_consume_stats.median_ms) / static_cast<double>(samples);

    std::filesystem::path out_path(opt.output);
    if (!out_path.parent_path().empty()) {
      std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(opt.output);
    if (!out) {
      throw std::runtime_error("failed to open output JSON: " + opt.output);
    }

    out << std::setprecision(9);
    out << "{\n"
        << "  \"schema\": \"n2wos_tcnn_cache_probe_v1\",\n"
        << "  \"patch\": \"0003-add-tcnn-device-resident-cache-probe\",\n"
        << "  \"generated_at_utc\": " << n2wos::json_quote(now_utc_string()) << ",\n"
        << "  \"command_line\": " << n2wos::json_quote(command_line) << ",\n"
        << "  \"cuda\": " << gpu_info_json() << ",\n"
        << "  \"implementation_mode\": {\n"
        << "    \"cache_backend\": \"tiny-cuda-nn_native_cpp\",\n"
        << "    \"python_bindings\": false,\n"
        << "    \"csv_postprocess\": false,\n"
        << "    \"training_batch_generation\": \"cuda_kernel\",\n"
        << "    \"training_inputs_location\": \"cuda_device_memory\",\n"
        << "    \"training_targets_location\": \"cuda_device_memory\",\n"
        << "    \"inference_inputs_location\": \"cuda_device_memory\",\n"
        << "    \"inference_outputs_location\": \"cuda_device_memory\",\n"
        << "    \"downstream_consumer\": \"cuda_kernel\",\n"
        << "    \"host_transfer_between_tcnn_and_consumer\": false,\n"
        << "    \"timing_scope\": \"cuda_events; inference timing excludes final validation readback\"\n"
        << "  },\n"
        << "  \"options\": {\n"
        << "    \"samples\": " << opt.samples << ",\n"
        << "    \"samples_padded_to_tcnn_granularity\": " << inference_batch << ",\n"
        << "    \"batch_size\": " << opt.batch_size << ",\n"
        << "    \"batch_size_padded_to_tcnn_granularity\": " << training_batch << ",\n"
        << "    \"train_steps\": " << opt.train_steps << ",\n"
        << "    \"repeat\": " << opt.repeat << ",\n"
        << "    \"warmup\": " << opt.warmup << ",\n"
        << "    \"seed\": " << opt.seed << ",\n"
        << "    \"target_function\": \"harmonic_x2_minus_y2_on_unit_cube_coordinates\",\n"
        << "    \"network\": " << n2wos::json_quote(opt.network) << ",\n"
        << "    \"jit_requested\": " << (requested_jit ? "true" : "false") << ",\n"
        << "    \"jit_supported\": " << (jit_supported ? "true" : "false") << ",\n"
        << "    \"jit_enabled\": " << (jit_enabled ? "true" : "false") << ",\n"
        << "    \"n_levels\": " << opt.n_levels << ",\n"
        << "    \"n_features_per_level\": " << opt.n_features_per_level << ",\n"
        << "    \"log2_hashmap_size\": " << opt.log2_hashmap_size << ",\n"
        << "    \"base_resolution\": " << opt.base_resolution << ",\n"
        << "    \"per_level_scale\": " << opt.per_level_scale << ",\n"
        << "    \"n_neurons\": " << opt.n_neurons << ",\n"
        << "    \"n_hidden_layers\": " << opt.n_hidden_layers << ",\n"
        << "    \"learning_rate\": " << opt.learning_rate << "\n"
        << "  },\n"
        << "  \"training\": {\n"
        << "    \"total_ms\": " << train_ms << ",\n"
        << "    \"ms_per_step\": " << (opt.train_steps > 0 ? train_ms / static_cast<float>(opt.train_steps) : 0.0f) << ",\n"
        << "    \"includes_gpu_batch_generation\": true,\n"
        << "    \"host_readback_in_training_loop\": false\n"
        << "  },\n"
        << "  \"inference\": {\n"
        << "    \"kernel_ms\": " << vector_json(inference_stats.ms) << ",\n"
        << "    \"kernel_ms_median\": " << inference_stats.median_ms << ",\n"
        << "    \"kernel_ms_mean\": " << inference_stats.mean_ms << ",\n"
        << "    \"median_us_per_sample\": " << median_us_per_sample << ",\n"
        << "    \"median_msamples_per_second\": " << median_mqueries_per_second << "\n"
        << "  },\n"
        << "  \"inference_plus_downstream_consumer\": {\n"
        << "    \"kernel_ms\": " << vector_json(inference_consume_stats.ms) << ",\n"
        << "    \"kernel_ms_median\": " << inference_consume_stats.median_ms << ",\n"
        << "    \"kernel_ms_mean\": " << inference_consume_stats.mean_ms << ",\n"
        << "    \"median_us_per_sample\": " << median_consume_us_per_sample << ",\n"
        << "    \"host_transfer_between_stages\": false\n"
        << "  },\n"
        << "  \"validation\": {\n"
        << "    \"checked_repeated_samples\": " << static_cast<unsigned long long>(samples) * static_cast<unsigned long long>(opt.repeat) << ",\n"
        << "    \"mean_prediction\": " << mean_pred << ",\n"
        << "    \"mean_exact\": " << mean_exact << ",\n"
        << "    \"mse\": " << mse << ",\n"
        << "    \"rmse\": " << rmse << ",\n"
        << "    \"mae\": " << mae << "\n"
        << "  }\n"
        << "}\n";
    out.close();

    N2WOS_CUDA_CHECK(cudaStreamDestroy(stream));
    tcnn::free_all_gpu_memory_arenas();
    std::cout << "wrote " << opt.output << "\n";
    std::cout << "median inference: " << median_us_per_sample << " us/sample\n";
    std::cout << "RMSE after training: " << rmse << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
