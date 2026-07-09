#include "worker/pytorch_worker.hpp"

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace exam {
namespace {

constexpr const char* PYTORCH_WORKER_TYPE = "pytorch";

using Clock = std::chrono::steady_clock;

std::int64_t elapsed_microseconds(Clock::time_point start_time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start_time).count();
}

int positive_int_env(const char* name) {
    const char* value = std::getenv(name);
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

void configure_pytorch_threads() {
    const int num_threads = positive_int_env("EXAM_PYTORCH_NUM_THREADS");
    if (num_threads > 0) {
        at::set_num_threads(num_threads);
    }

    const int interop_threads =
        positive_int_env("EXAM_PYTORCH_INTEROP_THREADS");
    if (interop_threads > 0) {
        at::set_num_interop_threads(interop_threads);
    }
}

void configure_torchscript_runtime() {
    torch::jit::setGraphExecutorOptimize(true);
    torch::jit::getExecutorMode().store(false);
    torch::jit::getProfilingMode().store(false);
    torch::jit::getNumProfiledRuns().store(1);
}

void initialize_pytorch_runtime() {
    static std::once_flag once;
    std::call_once(once, [] {
        configure_pytorch_threads();
        configure_torchscript_runtime();

        std::cerr << "pytorch worker: runtime intra_threads="
                  << at::get_num_threads()
                  << " interop_threads=" << at::get_num_interop_threads()
                  << " graph_executor_optimize="
                  << (torch::jit::getGraphExecutorOptimize() ? "on" : "off")
                  << " executor_mode="
                  << (torch::jit::getExecutorMode().load() ? "on" : "off")
                  << " profiling_mode="
                  << (torch::jit::getProfilingMode().load() ? "on" : "off")
                  << " num_profiled_runs="
                  << torch::jit::getNumProfiledRuns().load()
                  << '\n';

        const auto qengines = at::globalContext().supportedQEngines();
        if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK)
            == qengines.end()) {
            std::cerr
                << "pytorch worker: QNNPACK is not supported by this LibTorch; "
                << "float TorchScript models can still run\n";
            return;
        }

        at::globalContext().setQEngine(at::QEngine::QNNPACK);
    });
}

std::size_t checked_volume(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("pytorch tensor shape metadata is required");
    }

    std::size_t volume = 1;
    for (std::int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("pytorch tensor shape has non-positive dimension");
        }
        volume *= static_cast<std::size_t>(dim);
    }
    return volume;
}

c10::ScalarType scalar_type_from_dtype(const std::string& dtype) {
    if (dtype == "float32" || dtype == "float") {
        return c10::ScalarType::Float;
    }
    if (dtype == "float64" || dtype == "double") {
        return c10::ScalarType::Double;
    }
    if (dtype == "float16" || dtype == "half") {
        return c10::ScalarType::Half;
    }
    if (dtype == "bfloat16") {
        return c10::ScalarType::BFloat16;
    }
    if (dtype == "int64" || dtype == "long") {
        return c10::ScalarType::Long;
    }
    if (dtype == "int32" || dtype == "int") {
        return c10::ScalarType::Int;
    }
    if (dtype == "int16" || dtype == "short") {
        return c10::ScalarType::Short;
    }
    if (dtype == "int8") {
        return c10::ScalarType::Char;
    }
    if (dtype == "uint8") {
        return c10::ScalarType::Byte;
    }
    if (dtype == "bool") {
        return c10::ScalarType::Bool;
    }

    throw std::runtime_error("unsupported pytorch tensor dtype: " + dtype);
}

std::size_t scalar_type_size(c10::ScalarType scalar_type) {
    switch (scalar_type) {
        case c10::ScalarType::Double:
            return 8;
        case c10::ScalarType::Float:
            return 4;
        case c10::ScalarType::Half:
            return 2;
        case c10::ScalarType::BFloat16:
            return 2;
        case c10::ScalarType::Long:
            return 8;
        case c10::ScalarType::Int:
            return 4;
        case c10::ScalarType::Short:
            return 2;
        case c10::ScalarType::Char:
            return 1;
        case c10::ScalarType::Byte:
            return 1;
        case c10::ScalarType::Bool:
            return 1;
        default:
            break;
    }

    throw std::runtime_error(
        "unsupported pytorch tensor scalar type: "
        + std::to_string(static_cast<int>(scalar_type)));
}

std::size_t tensor_size_bytes(
    const std::vector<std::int64_t>& shape,
    const std::string& dtype) {
    return checked_volume(shape) * scalar_type_size(scalar_type_from_dtype(dtype));
}

std::vector<std::int64_t> tensor_shape(const at::Tensor& tensor) {
    const auto sizes = tensor.sizes();
    return std::vector<std::int64_t>(sizes.begin(), sizes.end());
}

void validate_tensor_metadata(
    const at::Tensor& tensor,
    const std::vector<std::int64_t>& expected_shape,
    const std::string& expected_dtype,
    const std::string& tensor_name) {
    if (!expected_shape.empty() && tensor_shape(tensor) != expected_shape) {
        throw std::runtime_error(
            "pytorch tensor shape does not match SG metadata: " + tensor_name);
    }

    if (!expected_dtype.empty()
        && tensor.scalar_type() != scalar_type_from_dtype(expected_dtype)) {
        throw std::runtime_error(
            "pytorch tensor dtype does not match SG metadata: " + tensor_name);
    }
}

at::Tensor payload_to_tensor(
    const Payload& input,
    const std::vector<std::int64_t>& shape,
    const std::string& dtype) {
    if (shape.empty() || dtype.empty()) {
        throw std::runtime_error(
            "pytorch worker requires input shape and dtype metadata");
    }

    const std::size_t expected_size = tensor_size_bytes(shape, dtype);
    if (input.bytes.size() < expected_size) {
        throw std::runtime_error("input payload is smaller than pytorch input tensor");
    }

    return torch::from_blob(
        const_cast<char*>(input.bytes.data()),
        shape,
        torch::TensorOptions()
            .dtype(scalar_type_from_dtype(dtype))
            .device(torch::kCPU));
}

Payload tensor_to_payload(const at::Tensor& tensor) {
    at::Tensor cpu_tensor = tensor;
    if (!cpu_tensor.device().is_cpu()) {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }
    if (!cpu_tensor.is_contiguous()) {
        cpu_tensor = cpu_tensor.contiguous();
    }

    const std::size_t output_size =
        static_cast<std::size_t>(cpu_tensor.numel()) * cpu_tensor.element_size();
    Payload output{};
    output.bytes.resize(output_size);
    if (output_size > 0) {
        std::memcpy(output.bytes.data(), cpu_tensor.data_ptr(), output_size);
    }
    return output;
}

} // namespace

class PyTorchWorker::CachedModule {
public:
    explicit CachedModule(const SubgraphConfig& config)
        : sg_path_(config.path()),
          expected_input_shape_(config.input_shape()),
          expected_input_dtype_(config.input_dtype()),
          expected_output_shape_(config.output_shape()),
          expected_output_dtype_(config.output_dtype()),
          module_(torch::jit::load(sg_path_)) {
        module_.to(torch::kCPU);
        module_.eval();
    }

    CachedModule(const CachedModule&) = delete;
    CachedModule& operator=(const CachedModule&) = delete;

    const std::string& sg_path() const {
        return sg_path_;
    }

    torch::jit::script::Module& module() {
        return module_;
    }

    const std::vector<std::int64_t>& input_shape() const {
        return expected_input_shape_;
    }

    const std::string& input_dtype() const {
        return expected_input_dtype_;
    }

    void validate_output(const at::Tensor& output) const {
        validate_tensor_metadata(
            output,
            expected_output_shape_,
            expected_output_dtype_,
            "output");
    }

private:
    std::string sg_path_;
    std::vector<std::int64_t> expected_input_shape_;
    std::string expected_input_dtype_;
    std::vector<std::int64_t> expected_output_shape_;
    std::string expected_output_dtype_;
    torch::jit::script::Module module_;
};

class PyTorchWorker::PyTorchExecutor {
public:
    explicit PyTorchExecutor(CachedModule& cached_module)
        : module_(cached_module) {}

    PyTorchExecutor(const PyTorchExecutor&) = delete;
    PyTorchExecutor& operator=(const PyTorchExecutor&) = delete;

    void run(const at::Tensor& input) {
        c10::InferenceMode inference_mode(true);
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input);

        const auto forward_started_at = Clock::now();
        torch::jit::IValue result = module_.module().forward(inputs);
        forward_latency_us_ = elapsed_microseconds(forward_started_at);

        const auto postprocess_started_at = Clock::now();
        if (!result.isTensor()) {
            throw std::runtime_error(
                "pytorch worker currently requires a single tensor output: "
                + module_.sg_path());
        }

        output_ = result.toTensor();
        if (!output_.device().is_cpu()) {
            output_ = output_.to(torch::kCPU);
        }
        if (!output_.is_contiguous()) {
            output_ = output_.contiguous();
        }
        module_.validate_output(output_);
        postprocess_latency_us_ = elapsed_microseconds(postprocess_started_at);
    }

    const at::Tensor& output() const {
        return output_;
    }

    std::int64_t forward_latency_us() const {
        return forward_latency_us_;
    }

    std::int64_t postprocess_latency_us() const {
        return postprocess_latency_us_;
    }

private:
    CachedModule& module_;
    at::Tensor output_;
    std::int64_t forward_latency_us_{0};
    std::int64_t postprocess_latency_us_{0};
};

PyTorchWorker::PyTorchWorker(
    std::uint32_t worker_id,
    ThreadConfig thread_config)
    : Worker(
          worker_id,
          PYTORCH_WORKER_TYPE,
          "pytorch-worker",
          std::move(thread_config)) {}

PyTorchWorker::~PyTorchWorker() = default;

void PyTorchWorker::init() {
    initialize_pytorch_runtime();
}

void PyTorchWorker::set_input(const Payload& input) {
    input_ = input;
}

PyTorchWorker::CachedModule& PyTorchWorker::module_for(
    const SubgraphConfig& config) {
    const std::string& sg_path = config.path();
    auto cache_entry = module_cache_.find(sg_path);
    if (cache_entry != module_cache_.end()) {
        return *cache_entry->second;
    }

    auto new_module = std::make_unique<CachedModule>(config);
    CachedModule* module = new_module.get();
    module_cache_[sg_path] = std::move(new_module);
    return *module;
}

std::string PyTorchWorker::current_request_key() const {
    std::ostringstream key;
    key << current_request().channel_name_string()
        << '#'
        << current_request().request_id;
    return key.str();
}

void PyTorchWorker::execute() {
    const auto execute_started_at = Clock::now();
    const SubgraphConfig& config = current_sg().config_for(PYTORCH_WORKER_TYPE);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    CachedModule& module = module_for(config);
    const std::string request_key = current_request_key();
    auto current_executor = std::make_unique<PyTorchExecutor>(module);
    std::int64_t input_prepare_us = 0;
    std::int64_t output_payload_us = 0;

    if (current_sg().is_first()) {
        const auto input_prepare_started_at = Clock::now();
        at::Tensor input_tensor =
            payload_to_tensor(input_, module.input_shape(), module.input_dtype());
        input_prepare_us = elapsed_microseconds(input_prepare_started_at);
        current_executor->run(input_tensor);
    } else {
        auto previous_executor = last_executor_by_request_.find(request_key);
        if (previous_executor == last_executor_by_request_.end()) {
            throw std::runtime_error("pytorch worker has no previous SG output");
        }
        current_executor->run(previous_executor->second->output());
    }

    const std::int64_t forward_us = current_executor->forward_latency_us();
    const std::int64_t postprocess_us = current_executor->postprocess_latency_us();

    if (current_sg().is_last()) {
        const auto output_payload_started_at = Clock::now();
        output_ = tensor_to_payload(current_executor->output());
        output_payload_us = elapsed_microseconds(output_payload_started_at);
        last_executor_by_request_.erase(request_key);
    } else {
        last_executor_by_request_[request_key] = std::move(current_executor);
    }

    std::ostringstream log;
    log << name() << ": timing request="
        << current_request().request_id
        << " channel=" << current_request().channel_name_string()
        << " sg=" << current_sg().id()
        << " execute_us=" << elapsed_microseconds(execute_started_at)
        << " input_prepare_us=" << input_prepare_us
        << " forward_us=" << forward_us
        << " postprocess_us=" << postprocess_us
        << " output_payload_us=" << output_payload_us
        << '\n';
    std::cout << log.str();
}

Payload PyTorchWorker::get_output() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return output_;
}

void PyTorchWorker::terminate() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    last_executor_by_request_.clear();
    module_cache_.clear();
}

void PyTorchWorker::release_sg_cache(const Subgraph& sg) {
    if (!sg.supports(PYTORCH_WORKER_TYPE)) {
        return;
    }

    const SubgraphConfig& config = sg.config_for(PYTORCH_WORKER_TYPE);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    module_cache_.erase(config.path());
}

} // namespace exam
