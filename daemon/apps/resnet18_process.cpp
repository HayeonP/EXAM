#include "datatype/payload.hpp"
#include "exam_client.hpp"
#include "thread_config.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

constexpr int PROCESS_PRIORITY = 85;
constexpr const char* DEFAULT_SG_SEQUENCE_FILE_PATH =
    "artifacts/resnet18/sg_sequence.json";
constexpr std::size_t RESNET18_INPUT_FLOATS = 1 * 3 * 224 * 224;
constexpr std::size_t RESNET18_INPUT_BYTES =
    RESNET18_INPUT_FLOATS * sizeof(float);
constexpr std::size_t RESNET18_OUTPUT_BYTES = 1000 * sizeof(float);

int parse_input_id(int argc, char** argv) {
    if (argc < 2) {
        return 1;
    }

    const int input_id = std::atoi(argv[1]);
    return input_id > 0 ? input_id : 1;
}

std::string parse_sg_sequence_file_path(int argc, char** argv) {
    if (argc < 3) {
        return DEFAULT_SG_SEQUENCE_FILE_PATH;
    }

    return argv[2];
}

std::string optional_arg(int argc, char** argv, int index) {
    if (argc <= index) {
        return "";
    }

    return argv[index];
}

int parse_repeat_count(int argc, char** argv) {
    if (argc < 6) {
        return 1;
    }

    const int count = std::atoi(argv[5]);
    return count > 0 ? count : 1;
}

std::uint64_t fnv1a64(const std::vector<char>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (char byte : bytes) {
        hash ^= static_cast<unsigned char>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

exam::Payload make_resnet18_input(int input_id) {
    exam::Payload input{};
    input.bytes.resize(RESNET18_INPUT_BYTES);

    std::uint32_t state =
        0x9e3779b9U ^ (static_cast<std::uint32_t>(input_id) * 0x85ebca6bU);
    for (std::size_t index = 0; index < RESNET18_INPUT_FLOATS; ++index) {
        state = state * 1664525U + 1013904223U;
        const float value =
            (static_cast<float>((state >> 8) & 0xFFFFU) / 65535.0F) * 2.0F
            - 1.0F;
        std::memcpy(
            input.bytes.data() + index * sizeof(float),
            &value,
            sizeof(float));
    }

    return input;
}

void create_ready_file(const std::string& ready_dir, int input_id) {
    if (ready_dir.empty()) {
        return;
    }

    const std::string path =
        ready_dir + "/input_" + std::to_string(input_id) + ".ready";
    std::ofstream ready_file(path);
    if (!ready_file) {
        throw std::runtime_error("failed to create ready file: " + path);
    }

    ready_file << getpid() << '\n';
}

void wait_for_gate(const std::string& gate_path) {
    if (gate_path.empty()) {
        return;
    }

    while (access(gate_path.c_str(), F_OK) != 0) {
        usleep(10000);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;

        exam::ThreadConfig::set_priority("resnet18-process", PROCESS_PRIORITY);

        const int input_id = parse_input_id(argc, argv);
        const std::string sg_sequence_file_path =
            parse_sg_sequence_file_path(argc, argv);
        const std::string ready_dir = optional_arg(argc, argv, 3);
        const std::string gate_path = optional_arg(argc, argv, 4);
        const int repeat_count = parse_repeat_count(argc, argv);

        exam::ExamClient client(
            sg_sequence_file_path,
            RESNET18_INPUT_BYTES,
            RESNET18_OUTPUT_BYTES);

        std::cout << "resnet18_process pid=" << getpid()
                  << " input_id=" << input_id
                  << " channel=" << client.channel_name()
                  << " input_size=" << RESNET18_INPUT_BYTES
                  << " output_size=" << RESNET18_OUTPUT_BYTES
                  << " repeat_count=" << repeat_count
                  << " sg_sequence_file=" << sg_sequence_file_path << '\n';

        create_ready_file(ready_dir, input_id);
        wait_for_gate(gate_path);

        for (int iteration = 0; iteration < repeat_count; ++iteration) {
            const int iteration_input_id = input_id + iteration;
            exam::Payload input = make_resnet18_input(iteration_input_id);
            const std::string input_hash = hex64(fnv1a64(input.bytes));
            const auto start_time = std::chrono::steady_clock::now();
            if (!client.request(input)) {
                std::cerr << "resnet18_process input_id=" << iteration_input_id
                          << " iteration=" << iteration
                          << " request dropped\n";
                return 2;
            }

            exam::Payload output = client.wait();
            const auto end_time = std::chrono::steady_clock::now();
            const auto latency_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time).count();
            std::cout << "RESULT input_id=" << iteration_input_id
                      << " iteration=" << iteration
                      << " input_hash=" << input_hash
                      << " output_hash=" << hex64(fnv1a64(output.bytes))
                      << " output_size=" << output.bytes.size()
                      << " latency_us=" << latency_us << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "resnet18_process: " << e.what() << '\n';
        return 1;
    }
}
