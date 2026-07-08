#include <NvInfer.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr const char* MODEL_DIR = "artifacts/sample_model";
constexpr const char* MOCK_DIR = "artifacts/sample_model/mock";
constexpr const char* TENSOR_RT_DIR = "artifacts/sample_model/tensor_rt_gpu";
constexpr const char* SG_SEQUENCE_PATH = "artifacts/sample_model/sg_sequence.json";

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "tensor_rt: " << msg << '\n';
        }
    }
};

void make_directory(const char* path) {
    if (mkdir(path, 0775) == -1 && errno != EEXIST) {
        throw std::runtime_error(
            std::string("mkdir failed for ") + path + ": " + std::strerror(errno));
    }
}

void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open file for write: " + path);
    }
    file << text;
}

nvinfer1::Dims sample_tensor_dims() {
    nvinfer1::Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = 1;
    dims.d[1] = 4;
    return dims;
}

std::unique_ptr<nvinfer1::IHostMemory> build_fake_resnet_engine(
    nvinfer1::ILogger& logger,
    const std::string& stage_name) {
    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(logger)};
    if (!builder) {
        throw std::runtime_error("failed to create tensor_rt builder");
    }

    const auto explicit_batch =
        1U << static_cast<std::uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(explicit_batch)};
    if (!network) {
        throw std::runtime_error("failed to create tensor_rt network");
    }

    nvinfer1::ITensor* input = network->addInput(
        "input",
        nvinfer1::DataType::kFLOAT,
        sample_tensor_dims());
    if (input == nullptr) {
        throw std::runtime_error("failed to add tensor_rt input");
    }

    nvinfer1::IIdentityLayer* identity = network->addIdentity(*input);
    if (identity == nullptr) {
        throw std::runtime_error("failed to add fake ResNet identity layer");
    }
    identity->setName(stage_name.c_str());

    nvinfer1::ITensor* output = identity->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    std::unique_ptr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()};
    if (!config) {
        throw std::runtime_error("failed to create tensor_rt builder config");
    }

    std::unique_ptr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *config)};
    if (!serialized) {
        throw std::runtime_error("failed to build fake ResNet tensor_rt engine");
    }

    return serialized;
}

void write_engine_file(
    const std::string& path,
    nvinfer1::ILogger& logger,
    const std::string& stage_name) {
    std::unique_ptr<nvinfer1::IHostMemory> engine =
        build_fake_resnet_engine(logger, stage_name);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open tensor_rt engine for write: " + path);
    }

    file.write(
        static_cast<const char*>(engine->data()),
        static_cast<std::streamsize>(engine->size()));
    if (!file) {
        throw std::runtime_error("failed to write tensor_rt engine: " + path);
    }
}

} // namespace

int main() {
    try {
        make_directory("artifacts");
        make_directory(MODEL_DIR);
        make_directory(MOCK_DIR);
        make_directory(TENSOR_RT_DIR);

        TensorRtLogger logger;
        const std::vector<std::string> labels{
            "fake_resnet_stem",
            "fake_resnet_block1",
            "fake_resnet_block2",
            "fake_resnet_classifier"};

        for (std::size_t i = 0; i < labels.size(); ++i) {
            const std::string index = std::to_string(i + 1);
            write_text_file(
                std::string(MOCK_DIR) + "/fake_resnet_sg" + index + ".txt",
                "mock " + labels[i] + "\n");
            write_engine_file(
                std::string(TENSOR_RT_DIR) + "/fake_resnet_sg" + index + ".plan",
                logger,
                labels[i]);
        }

        std::ofstream sg_sequence(SG_SEQUENCE_PATH);
        if (!sg_sequence) {
            throw std::runtime_error("failed to open SG sequence file");
        }

        sg_sequence << "[\n";
        for (std::size_t i = 0; i < labels.size(); ++i) {
            const std::string index = std::to_string(i + 1);
            sg_sequence
                << "  {\n"
                << "    \"label\": \"" << labels[i] << "\",\n"
                << "    \"mock\": {\n"
                << "      \"path\": \"mock/fake_resnet_sg" << index << ".txt\",\n"
                << "      \"input\": { \"shape\": [256], \"dtype\": \"bytes\" },\n"
                << "      \"output\": { \"shape\": [256], \"dtype\": \"bytes\" }\n"
                << "    },\n"
                << "    \"tensor_rt_gpu\": {\n"
                << "      \"path\": \"tensor_rt_gpu/fake_resnet_sg"
                << index << ".plan\",\n"
                << "      \"input\": { \"shape\": [1, 4], \"dtype\": \"float32\" },\n"
                << "      \"output\": { \"shape\": [1, 4], \"dtype\": \"float32\" }\n"
                << "    }\n"
                << "  }" << (i + 1 == labels.size() ? "\n" : ",\n");
        }
        sg_sequence << "]\n";

        std::cout << "sample model generated at " << MODEL_DIR << '\n';
        std::cout << "sg_sequence_path=" << SG_SEQUENCE_PATH << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build_sample_model: " << e.what() << '\n';
        return 1;
    }
}
