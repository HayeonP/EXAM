#include "worker/tensor_rt_gpu_worker.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace exam {
namespace {

constexpr const char* TENSOR_RT_GPU_WORKER_TYPE = "tensor_rt_gpu";

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "tensor_rt: " << msg << '\n';
        }
    }
};

TensorRtLogger& tensor_rt_logger() {
    static TensorRtLogger logger;
    return logger;
}

void initialize_tensor_rt_plugins() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (!initLibNvInferPlugins(&tensor_rt_logger(), "")) {
            throw std::runtime_error("tensor_rt plugin initialization failed");
        }
    });
}

std::vector<char> load_sg(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open tensor_rt SG: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("tensor_rt SG file is empty: " + path);
    }

    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("failed to read tensor_rt SG: " + path);
    }

    return bytes;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::size_t dim_volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        throw std::runtime_error("tensor_rt tensor has invalid dimensions");
    }

    std::size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("tensor_rt tensor has non-positive dimensions");
        }
        volume *= static_cast<std::size_t>(dims.d[i]);
    }

    return volume;
}

std::size_t data_type_size(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
        case nvinfer1::DataType::kFP8:
            return 1;
    }

    throw std::runtime_error("unsupported tensor_rt data type");
}

std::string data_type_name(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT8:
            return "int8";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        case nvinfer1::DataType::kUINT8:
            return "uint8";
        case nvinfer1::DataType::kFP8:
            return "fp8";
    }

    return "unknown";
}

std::vector<std::int64_t> dims_to_shape(const nvinfer1::Dims& dims) {
    std::vector<std::int64_t> shape;
    if (dims.nbDims < 0) {
        return shape;
    }

    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

std::size_t tensor_size_bytes(
    const nvinfer1::ICudaEngine& engine,
    const std::string& tensor_name) {
    const nvinfer1::Dims dims = engine.getTensorShape(tensor_name.c_str());
    return dim_volume(dims)
        * data_type_size(engine.getTensorDataType(tensor_name.c_str()));
}

void require_device_tensor(
    const nvinfer1::ICudaEngine& engine,
    const std::string& tensor_name) {
    if (engine.getTensorLocation(tensor_name.c_str()) != nvinfer1::TensorLocation::kDEVICE) {
        throw std::runtime_error(
            "host tensor_rt tensors are not supported yet: " + tensor_name);
    }
}

void validate_tensor_info(
    const nvinfer1::ICudaEngine& engine,
    const std::string& tensor_name,
    const std::vector<std::int64_t>& expected_shape,
    const std::string& expected_dtype) {
    if (expected_shape.empty() || expected_dtype.empty()) {
        return;
    }

    const std::vector<std::int64_t> actual_shape =
        dims_to_shape(engine.getTensorShape(tensor_name.c_str()));
    if (actual_shape != expected_shape) {
        throw std::runtime_error(
            "tensor_rt tensor shape does not match SG metadata: " + tensor_name);
    }

    const std::string actual_dtype =
        data_type_name(engine.getTensorDataType(tensor_name.c_str()));
    if (actual_dtype != expected_dtype) {
        throw std::runtime_error(
            "tensor_rt tensor dtype does not match SG metadata: " + tensor_name);
    }
}

} // namespace

class TensorRtGpuWorker::CachedEngine {
public:
    CachedEngine(
        int device_id,
        const SubgraphConfig& config)
        : sg_path_(config.path()),
          device_id_(device_id),
          expected_input_shape_(config.input_shape()),
          expected_input_dtype_(config.input_dtype()),
          expected_output_shape_(config.output_shape()),
          expected_output_dtype_(config.output_dtype()) {
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        initialize_tensor_rt_plugins();

        const std::vector<char> serialized_engine = load_sg(sg_path_);
        runtime_.reset(nvinfer1::createInferRuntime(tensor_rt_logger()));
        if (!runtime_) {
            throw std::runtime_error("failed to create tensor_rt runtime");
        }

        engine_.reset(runtime_->deserializeCudaEngine(
            serialized_engine.data(),
            serialized_engine.size()));
        if (!engine_) {
            throw std::runtime_error("failed to deserialize tensor_rt engine: " + sg_path_);
        }

        configure_io_tensors();
        validate_tensor_info(
            *engine_,
            input_name_,
            expected_input_shape_,
            expected_input_dtype_);
        validate_tensor_info(
            *engine_,
            output_name_,
            expected_output_shape_,
            expected_output_dtype_);
    }

    ~CachedEngine() = default;

    CachedEngine(const CachedEngine&) = delete;
    CachedEngine& operator=(const CachedEngine&) = delete;

    int device_id() const {
        return device_id_;
    }

    const std::string& sg_path() const {
        return sg_path_;
    }

    nvinfer1::ICudaEngine& engine() const {
        return *engine_;
    }

    const std::string& input_name() const {
        return input_name_;
    }

    const std::string& output_name() const {
        return output_name_;
    }

    std::size_t input_size() const {
        return input_size_;
    }

    std::size_t output_size() const {
        return output_size_;
    }

private:
    void configure_io_tensors() {
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            if (tensor_name == nullptr) {
                throw std::runtime_error("tensor_rt engine has unnamed IO tensor");
            }

            const std::string name = tensor_name;
            const nvinfer1::TensorIOMode mode =
                engine_->getTensorIOMode(tensor_name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_names_.push_back(name);
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_names_.push_back(name);
            }
        }

        if (input_names_.size() != 1 || output_names_.size() != 1) {
            throw std::runtime_error(
                "tensor_rt worker currently requires exactly one input and one output tensor");
        }

        input_name_ = input_names_.front();
        output_name_ = output_names_.front();
        require_device_tensor(*engine_, input_name_);
        require_device_tensor(*engine_, output_name_);
        input_size_ = tensor_size_bytes(*engine_, input_name_);
        output_size_ = tensor_size_bytes(*engine_, output_name_);
    }

    std::string sg_path_;
    int device_id_{0};
    std::vector<std::int64_t> expected_input_shape_;
    std::string expected_input_dtype_;
    std::vector<std::int64_t> expected_output_shape_;
    std::string expected_output_dtype_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::string input_name_;
    std::string output_name_;
    std::size_t input_size_{0};
    std::size_t output_size_{0};
};

class TensorRtGpuWorker::TensorRtExecutor {
public:
    explicit TensorRtExecutor(CachedEngine& cached_engine)
        : engine_(cached_engine) {
        check_cuda(cudaSetDevice(engine_.device_id()), "cudaSetDevice");
        context_.reset(engine_.engine().createExecutionContext());
        if (!context_) {
            throw std::runtime_error(
                "failed to create tensor_rt execution context: "
                + engine_.sg_path());
        }

        allocate_output_buffer();
        bind_output();
    }

    ~TensorRtExecutor() {
        cudaSetDevice(engine_.device_id());
        if (input_device_ != nullptr) {
            cudaFree(input_device_);
            input_device_ = nullptr;
        }
        if (output_device_ != nullptr) {
            cudaFree(output_device_);
            output_device_ = nullptr;
        }
    }

    TensorRtExecutor(const TensorRtExecutor&) = delete;
    TensorRtExecutor& operator=(const TensorRtExecutor&) = delete;

    void copy_h2d(const Payload& input, cudaStream_t stream) {
        check_cuda(cudaSetDevice(engine_.device_id()), "cudaSetDevice");
        if (input.bytes.size() < engine_.input_size()) {
            throw std::runtime_error(
                "input payload is smaller than tensor_rt input tensor");
        }

        allocate_input_buffer();
        bind_input(input_device_);
        check_cuda(
            cudaMemcpyAsync(
                input_device_,
                input.bytes.data(),
                engine_.input_size(),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync input");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize h2d");
    }

    void use_previous_output(const TensorRtExecutor& previous_executor) {
        check_cuda(cudaSetDevice(engine_.device_id()), "cudaSetDevice");
        if (previous_executor.output_size() != engine_.input_size()) {
            throw std::runtime_error(
                "previous SG output size does not match current SG input size");
        }

        bind_input(previous_executor.output_device());
    }

    void run(cudaStream_t stream) {
        check_cuda(cudaSetDevice(engine_.device_id()), "cudaSetDevice");
        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error(
                "tensor_rt enqueueV3 failed: " + engine_.sg_path());
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize execute");
    }

    Payload copy_d2h(cudaStream_t stream) {
        check_cuda(cudaSetDevice(engine_.device_id()), "cudaSetDevice");
        Payload output{};
        output.bytes.resize(engine_.output_size());
        check_cuda(
            cudaMemcpyAsync(
                output.bytes.data(),
                output_device_,
                engine_.output_size(),
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync output");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize d2h");
        return output;
    }

    const void* output_device() const {
        return output_device_;
    }

    std::size_t output_size() const {
        return engine_.output_size();
    }

private:
    void allocate_input_buffer() {
        if (input_device_ != nullptr) {
            return;
        }

        check_cuda(
            cudaMalloc(&input_device_, engine_.input_size()),
            "cudaMalloc input");
    }

    void allocate_output_buffer() {
        check_cuda(
            cudaMalloc(&output_device_, engine_.output_size()),
            "cudaMalloc output");
    }

    void bind_input(const void* device_ptr) {
        if (!context_->setInputTensorAddress(
                engine_.input_name().c_str(),
                const_cast<void*>(device_ptr))) {
            throw std::runtime_error(
                "failed to bind tensor_rt input tensor: "
                + engine_.input_name());
        }
    }

    void bind_output() {
        if (!context_->setTensorAddress(
                engine_.output_name().c_str(),
                output_device_)) {
            throw std::runtime_error(
                "failed to bind tensor_rt output tensor: "
                + engine_.output_name());
        }
    }

    CachedEngine& engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    void* input_device_{nullptr};
    void* output_device_{nullptr};
};

TensorRtGpuWorker::TensorRtGpuWorker(
    std::uint32_t worker_id,
    int device_id,
    ThreadConfig thread_config)
    : Worker(
          worker_id,
          TENSOR_RT_GPU_WORKER_TYPE,
          "tensor-rt-gpu-worker",
          std::move(thread_config)),
      device_id_(device_id) {}

TensorRtGpuWorker::~TensorRtGpuWorker() = default;

void TensorRtGpuWorker::init() {
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

void TensorRtGpuWorker::set_input(const Payload& input) {
    input_ = input;
}

TensorRtGpuWorker::CachedEngine& TensorRtGpuWorker::engine_for(
    const SubgraphConfig& config) {
    const std::string& sg_path = config.path();
    auto cache_entry = engine_cache_.find(sg_path);
    if (cache_entry != engine_cache_.end()) {
        return *cache_entry->second;
    }

    auto new_engine = std::make_unique<CachedEngine>(device_id_, config);
    CachedEngine* engine = new_engine.get();
    engine_cache_[sg_path] = std::move(new_engine);
    return *engine;
}

std::string TensorRtGpuWorker::current_request_key() const {
    std::ostringstream key;
    key << current_request().channel_name_string()
        << '#'
        << current_request().request_id;
    return key.str();
}

void TensorRtGpuWorker::execute() {
    const SubgraphConfig& config = current_sg().config_for(TENSOR_RT_GPU_WORKER_TYPE);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    CachedEngine& engine = engine_for(config);
    const std::string request_key = current_request_key();
    auto current_executor = std::make_unique<TensorRtExecutor>(engine);

    if (current_sg().is_first()) {
        current_executor->copy_h2d(input_, stream_);
    } else {
        auto previous_executor = last_executor_by_request_.find(request_key);
        if (previous_executor == last_executor_by_request_.end()) {
            throw std::runtime_error("tensor_rt worker has no previous SG output");
        }
        current_executor->use_previous_output(*previous_executor->second);
    }

    current_executor->run(stream_);
    if (current_sg().is_last()) {
        output_ = current_executor->copy_d2h(stream_);
        last_executor_by_request_.erase(request_key);
    } else {
        last_executor_by_request_[request_key] = std::move(current_executor);
    }
}

Payload TensorRtGpuWorker::get_output() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return output_;
}

void TensorRtGpuWorker::terminate() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    last_executor_by_request_.clear();
    engine_cache_.clear();
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void TensorRtGpuWorker::release_sg_cache(const Subgraph& sg) {
    if (!sg.supports(TENSOR_RT_GPU_WORKER_TYPE)) {
        return;
    }

    const SubgraphConfig& config = sg.config_for(TENSOR_RT_GPU_WORKER_TYPE);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    engine_cache_.erase(config.path());
}

} // namespace exam
