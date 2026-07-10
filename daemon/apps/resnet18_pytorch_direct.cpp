#include "config/cpu_affinity_config.hpp"
#include "thread_config.hpp"

#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* DEFAULT_TS_DIR = "artifacts/resnet18/pytorch";
constexpr std::size_t RESNET18_INPUT_FLOATS = 1 * 3 * 224 * 224;
constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

std::int64_t elapsed_microseconds(Clock::time_point start_time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start_time).count();
}

int positive_int_env(const char* primary, const char* fallback = nullptr) {
    const char* value = std::getenv(primary);
    if ((value == nullptr || value[0] == '\0') && fallback != nullptr) {
        value = std::getenv(fallback);
    }
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    return static_cast<int>(parsed);
}

bool bool_env_or_default(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    return std::string(value) != "0"
        && std::string(value) != "false"
        && std::string(value) != "False"
        && std::string(value) != "off"
        && std::string(value) != "OFF";
}

int parse_positive_int(int argc, char** argv, int index, int default_value) {
    if (argc <= index) {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || parsed <= 0) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int parse_nonnegative_int(int argc, char** argv, int index, int default_value) {
    if (argc <= index) {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || parsed < 0) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

std::string optional_arg(int argc, char** argv, int index, const char* default_value) {
    if (argc <= index || argv[index][0] == '\0') {
        return default_value;
    }
    return argv[index];
}

void configure_pytorch_runtime() {
    const int num_threads =
        positive_int_env("EXAM_PYTORCH_NUM_THREADS", "PYTORCH_NUM_THREADS");
    if (num_threads > 0) {
        at::set_num_threads(num_threads);
    }

    const int interop_threads = positive_int_env(
        "EXAM_PYTORCH_INTEROP_THREADS",
        "PYTORCH_INTEROP_THREADS");
    if (interop_threads > 0) {
        at::set_num_interop_threads(interop_threads);
    }

    const bool graph_executor_optimize =
        bool_env_or_default("EXAM_TORCH_JIT_OPTIMIZE", true);
    const bool executor_mode =
        bool_env_or_default("EXAM_TORCH_JIT_EXECUTOR_MODE", false);
    const bool profiling_mode =
        bool_env_or_default("EXAM_TORCH_JIT_PROFILING_MODE", false);
    const int num_profiled_runs =
        positive_int_env("EXAM_TORCH_JIT_NUM_PROFILED_RUNS");

    torch::jit::setGraphExecutorOptimize(graph_executor_optimize);
    torch::jit::getExecutorMode().store(executor_mode);
    torch::jit::getProfilingMode().store(profiling_mode);
    if (num_profiled_runs > 0) {
        torch::jit::getNumProfiledRuns().store(
            static_cast<std::size_t>(num_profiled_runs));
    }
}

std::uint64_t fnv1a64(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = FNV_OFFSET;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= FNV_PRIME;
    }
    return hash;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

std::vector<float> make_resnet18_input(int input_id) {
    std::vector<float> input(RESNET18_INPUT_FLOATS);
    std::uint32_t state =
        0x9e3779b9U ^ (static_cast<std::uint32_t>(input_id) * 0x85ebca6bU);
    for (std::size_t index = 0; index < RESNET18_INPUT_FLOATS; ++index) {
        state = state * 1664525U + 1013904223U;
        input[index] =
            (static_cast<float>((state >> 8) & 0xFFFFU) / 65535.0F) * 2.0F
            - 1.0F;
    }
    return input;
}

std::string tensor_hash(const at::Tensor& tensor) {
    at::Tensor cpu_tensor = tensor;
    if (!cpu_tensor.device().is_cpu()) {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }
    if (!cpu_tensor.is_contiguous()) {
        cpu_tensor = cpu_tensor.contiguous();
    }

    const std::size_t byte_size =
        static_cast<std::size_t>(cpu_tensor.numel()) * cpu_tensor.element_size();
    return hex64(fnv1a64(cpu_tensor.data_ptr(), byte_size));
}

std::string join_values(const std::vector<std::int64_t>& values) {
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ' ';
        }
        output << values[index];
    }
    return output.str();
}

std::string join_strings(const std::vector<std::string>& values) {
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ' ';
        }
        output << values[index];
    }
    return output.str();
}

torch::jit::script::Module load_module(const std::string& path) {
    torch::jit::script::Module module = torch::jit::load(path);
    module.to(torch::kCPU);
    module.eval();
    return module;
}

at::Tensor run_module(torch::jit::script::Module& module, const at::Tensor& input) {
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input);
    torch::jit::IValue result = module.forward(inputs);
    if (!result.isTensor()) {
        throw std::runtime_error("resnet18 pytorch direct requires tensor output");
    }
    return result.toTensor();
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;

        const int input_id = parse_positive_int(argc, argv, 1, 300);
        const std::string ts_dir = optional_arg(argc, argv, 2, DEFAULT_TS_DIR);
        const int warmup = parse_nonnegative_int(argc, argv, 3, 5);
        const int iterations = parse_positive_int(argc, argv, 4, 10);
        const int total = warmup + iterations;

        const exam::CpuAffinityConfig cpu_affinity =
            exam::CpuAffinityConfig::load_default();
        const std::vector<int> direct_cpu_ids =
            cpu_affinity.cpus_for_pytorch_direct();
        exam::ThreadConfig::bind_to_cpus(
            "resnet18-pytorch-direct",
            direct_cpu_ids);
        std::cout << "CPP_DIRECT_CPU_AFFINITY " << cpu_affinity.summary()
                  << " using=[" << exam::cpu_ids_to_string(direct_cpu_ids)
                  << "]\n";

        const int sched_fifo_priority =
            positive_int_env("EXAM_DIRECT_SCHED_FIFO_PRIORITY");
        if (sched_fifo_priority > 0) {
            exam::ThreadConfig::set_priority(
                "resnet18-pytorch-direct",
                sched_fifo_priority);
        }

        configure_pytorch_runtime();

        const auto load_started_at = Clock::now();
        std::vector<torch::jit::script::Module> modules;
        modules.reserve(3);
        modules.push_back(load_module(ts_dir + "/sg1.pt"));
        modules.push_back(load_module(ts_dir + "/sg2.pt"));
        modules.push_back(load_module(ts_dir + "/sg3.pt"));
        const auto load_us = elapsed_microseconds(load_started_at);

        std::vector<std::int64_t> latencies;
        std::vector<std::int64_t> sg1_latencies;
        std::vector<std::int64_t> sg2_latencies;
        std::vector<std::int64_t> sg3_latencies;
        std::vector<std::string> hashes;
        latencies.reserve(total);
        sg1_latencies.reserve(total);
        sg2_latencies.reserve(total);
        sg3_latencies.reserve(total);
        hashes.reserve(total);

        c10::InferenceMode inference_mode(true);
        for (int iteration = 0; iteration < total; ++iteration) {
            const int current_input_id = input_id + iteration;
            std::vector<float> input_values = make_resnet18_input(current_input_id);
            at::Tensor tensor = torch::from_blob(
                input_values.data(),
                {1, 3, 224, 224},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

            std::vector<std::int64_t> current_sg_latencies;
            current_sg_latencies.reserve(modules.size());
            const auto started_at = Clock::now();
            for (auto& module : modules) {
                const auto sg_started_at = Clock::now();
                tensor = run_module(module, tensor);
                current_sg_latencies.push_back(elapsed_microseconds(sg_started_at));
            }
            const auto latency_us = elapsed_microseconds(started_at);
            const std::string output_hash = tensor_hash(tensor);

            latencies.push_back(latency_us);
            sg1_latencies.push_back(current_sg_latencies[0]);
            sg2_latencies.push_back(current_sg_latencies[1]);
            sg3_latencies.push_back(current_sg_latencies[2]);
            hashes.push_back(output_hash);

            std::cout << "CPP_DIRECT_RESULT input_id=" << current_input_id
                      << " iteration=" << iteration
                      << " output_hash=" << output_hash
                      << " latency_us=" << latency_us
                      << " total_us=" << latency_us
                      << " sg1_us=" << current_sg_latencies[0]
                      << " sg2_us=" << current_sg_latencies[1]
                      << " sg3_us=" << current_sg_latencies[2] << '\n';
        }

        const auto measured_begin = static_cast<std::size_t>(warmup);
        std::vector<std::int64_t> measured_latencies(
            latencies.begin() + measured_begin,
            latencies.end());
        std::vector<std::int64_t> measured_sg1(
            sg1_latencies.begin() + measured_begin,
            sg1_latencies.end());
        std::vector<std::int64_t> measured_sg2(
            sg2_latencies.begin() + measured_begin,
            sg2_latencies.end());
        std::vector<std::int64_t> measured_sg3(
            sg3_latencies.begin() + measured_begin,
            sg3_latencies.end());
        std::vector<std::string> measured_hashes(
            hashes.begin() + measured_begin,
            hashes.end());

        std::cout << "CPP_DIRECT_LOAD_US=" << load_us << '\n';
        std::cout << "CPP_DIRECT_THREADS intra=" << at::get_num_threads()
                  << " interop=" << at::get_num_interop_threads() << '\n';
        std::cout << "CPP_DIRECT_JIT graph_executor_optimize="
                  << (torch::jit::getGraphExecutorOptimize() ? "on" : "off")
                  << " executor_mode="
                  << (torch::jit::getExecutorMode().load() ? "on" : "off")
                  << " profiling_mode="
                  << (torch::jit::getProfilingMode().load() ? "on" : "off")
                  << " num_profiled_runs="
                  << torch::jit::getNumProfiledRuns().load() << '\n';
        std::cout << "CPP_DIRECT_LATENCIES_US="
                  << join_values(measured_latencies) << '\n';
        std::cout << "CPP_DIRECT_SG1_US=" << join_values(measured_sg1) << '\n';
        std::cout << "CPP_DIRECT_SG2_US=" << join_values(measured_sg2) << '\n';
        std::cout << "CPP_DIRECT_SG3_US=" << join_values(measured_sg3) << '\n';
        std::cout << "CPP_DIRECT_OUTPUT_HASHES="
                  << join_strings(measured_hashes) << '\n';

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "resnet18_pytorch_direct: " << e.what() << '\n';
        return 1;
    }
}
