#include "exam_client.hpp"
#include "thread_config.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace {

constexpr int PROCESS_PRIORITY = 85;
constexpr const char* DEFAULT_SG_SEQUENCE_FILE_PATH = "mock_sg_sequence.fake";

int parse_request_count(int argc, char** argv) {
    if (argc < 2) {
        return 3;
    }

    const int count = std::atoi(argv[1]);
    return count > 0 ? count : 3;
}

std::size_t parse_payload_size(int argc, char** argv, int index, std::size_t default_value) {
    if (argc <= index) {
        return default_value;
    }

    const long size = std::atol(argv[index]);
    return size > 0 ? static_cast<std::size_t>(size) : default_value;
}

std::string parse_sg_sequence_file_path(int argc, char** argv, int index) {
    if (argc <= index) {
        return DEFAULT_SG_SEQUENCE_FILE_PATH;
    }

    return argv[index];
}

exam::Payload make_process_input(int sequence) {
    std::ostringstream input;
    input << "pid=" << getpid() << ", request=" << sequence;
    return exam::Payload::from_text(input.str());
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;

        exam::ThreadConfig::set_priority("process", PROCESS_PRIORITY);

        
        const std::size_t input_size = parse_payload_size(
            argc,
            argv,
            2,
            exam::DEFAULT_INPUT_SIZE);
        const std::size_t output_size = parse_payload_size(
            argc,
            argv,
            3,
            exam::DEFAULT_OUTPUT_SIZE);
        const std::string sg_sequence_file_path = parse_sg_sequence_file_path(argc, argv, 4);
        
        // (1) Init
        exam::ExamClient client(sg_sequence_file_path, input_size, output_size);
        std::cout << "process " << getpid()
                  << ": channel=" << client.channel_name()
                  << ", input_size=" << input_size
                  << ", output_size=" << output_size
                  << ", sg_sequence_file=" << sg_sequence_file_path << '\n';

        const int request_count = parse_request_count(argc, argv);
        for (int i = 0; i < request_count; ++i) {
            // (2) Requset
            const bool accepted = client.request(make_process_input(i));
            if (!accepted) {
                std::cout << "process " << getpid()
                          << ": dropped sequence=" << i << '\n';
                continue;
            }

            std::cout << "process " << getpid()
                      << ": requested sequence=" << i << '\n';

            // (3) Wait
            exam::Payload output = client.wait();
            std::cout << "process " << getpid()
                      << ": completed sequence=" << i
                      << " output=\"" << output.to_string() << "\"\n";
        }

        std::cout << "process " << getpid()
                  << ": stats accepted=" << client.accepted_requests()
                  << ", completed=" << client.completed_requests()
                  << ", dropped=" << client.dropped_requests() << '\n';

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "process: " << e.what() << '\n';
        return 1;
    }
}
