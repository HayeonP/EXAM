#include "datatype/payload.hpp"
#include "exam_client.hpp"
#include "thread_config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int PROCESS_PRIORITY = 85;
constexpr const char* DEFAULT_SG_SEQUENCE_FILE_PATH =
    "artifacts/resnet18/sg_sequence_tensor_rt_gpu.json";
constexpr std::size_t RESNET18_INPUT_FLOATS = 1 * 3 * 224 * 224;
constexpr std::size_t RESNET18_INPUT_BYTES =
    RESNET18_INPUT_FLOATS * sizeof(float);
constexpr std::size_t RESNET18_OUTPUT_BYTES = 1000 * sizeof(float);
constexpr int RESNET18_INPUT_WIDTH = 224;
constexpr int RESNET18_INPUT_HEIGHT = 224;
constexpr int RESNET18_RESIZE_SHORT_SIDE = 256;

struct Classification {
    int index = -1;
    float logit = 0.0F;
    double probability = 0.0;
    std::string label;
};

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

std::string base_name(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return path;
    }
    return path.substr(slash + 1);
}

std::string lower_copy(const std::string& value) {
    std::string lowered = value;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return lowered;
}

bool has_supported_image_extension(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }

    const std::string extension = lower_copy(path.substr(dot));
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg";
}

std::vector<std::string> list_image_files(const std::string& image_dir) {
    DIR* dir = opendir(image_dir.c_str());
    if (dir == nullptr) {
        throw std::runtime_error("failed to open image dir: " + image_dir);
    }

    std::vector<std::string> paths;
    while (dirent* entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name == "." || name == ".."
            || !has_supported_image_extension(name)) {
            continue;
        }
        paths.push_back(image_dir + "/" + name);
    }
    closedir(dir);

    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("no .png/.jpg/.jpeg images found in: "
                                 + image_dir);
    }
    return paths;
}

std::vector<std::string> load_labels(const std::string& labels_path) {
    std::vector<std::string> labels;
    if (labels_path.empty()) {
        return labels;
    }

    std::ifstream label_file(labels_path);
    if (!label_file) {
        throw std::runtime_error("failed to open labels file: " + labels_path);
    }

    std::string line;
    while (std::getline(label_file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        labels.push_back(line);
    }
    return labels;
}

exam::Payload make_resnet18_image_input(const std::string& image_path) {
    const cv::Mat image_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image_bgr.empty()) {
        throw std::runtime_error("failed to read image: " + image_path);
    }

    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

    int resized_width = RESNET18_RESIZE_SHORT_SIDE;
    int resized_height = RESNET18_RESIZE_SHORT_SIDE;
    if (image_rgb.cols <= image_rgb.rows) {
        resized_height = static_cast<int>(std::lround(
            static_cast<double>(image_rgb.rows) * RESNET18_RESIZE_SHORT_SIDE
            / image_rgb.cols));
    } else {
        resized_width = static_cast<int>(std::lround(
            static_cast<double>(image_rgb.cols) * RESNET18_RESIZE_SHORT_SIDE
            / image_rgb.rows));
    }

    cv::Mat resized;
    cv::resize(
        image_rgb,
        resized,
        cv::Size(resized_width, resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    const int crop_x = (resized.cols - RESNET18_INPUT_WIDTH) / 2;
    const int crop_y = (resized.rows - RESNET18_INPUT_HEIGHT) / 2;
    if (crop_x < 0 || crop_y < 0) {
        throw std::runtime_error("resized image is smaller than crop: "
                                 + image_path);
    }
    const cv::Mat cropped = resized(cv::Rect(
        crop_x,
        crop_y,
        RESNET18_INPUT_WIDTH,
        RESNET18_INPUT_HEIGHT));

    constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> stddev = {0.229F, 0.224F, 0.225F};
    std::array<float, RESNET18_INPUT_FLOATS> tensor{};

    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < RESNET18_INPUT_HEIGHT; ++y) {
            const auto* row = cropped.ptr<cv::Vec3b>(y);
            for (int x = 0; x < RESNET18_INPUT_WIDTH; ++x) {
                const float value =
                    static_cast<float>(row[x][channel]) / 255.0F;
                const float normalized = (value - mean[channel])
                                         / stddev[channel];
                const std::size_t tensor_offset =
                    static_cast<std::size_t>(channel)
                        * RESNET18_INPUT_HEIGHT * RESNET18_INPUT_WIDTH
                    + static_cast<std::size_t>(y) * RESNET18_INPUT_WIDTH + x;
                tensor[tensor_offset] = normalized;
            }
        }
    }

    exam::Payload input{};
    input.bytes.resize(RESNET18_INPUT_BYTES);
    std::memcpy(input.bytes.data(), tensor.data(), RESNET18_INPUT_BYTES);
    return input;
}

Classification classify_resnet18_output(
    const exam::Payload& output,
    const std::vector<std::string>& labels) {
    if (output.bytes.size() < RESNET18_OUTPUT_BYTES) {
        throw std::runtime_error("ResNet18 output is too small for logits");
    }

    std::array<float, 1000> logits{};
    std::memcpy(logits.data(), output.bytes.data(), RESNET18_OUTPUT_BYTES);

    const auto best_it = std::max_element(logits.begin(), logits.end());
    const int best_index =
        static_cast<int>(std::distance(logits.begin(), best_it));
    const float best_logit = *best_it;

    double denominator = 0.0;
    for (float logit : logits) {
        denominator += std::exp(static_cast<double>(logit - best_logit));
    }

    Classification classification;
    classification.index = best_index;
    classification.logit = best_logit;
    classification.probability = denominator > 0.0 ? 1.0 / denominator : 0.0;
    if (best_index >= 0 && best_index < static_cast<int>(labels.size())
        && !labels[best_index].empty()) {
        classification.label = labels[best_index];
    } else {
        classification.label = "class_" + std::to_string(best_index);
    }
    return classification;
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
        const std::string image_dir = optional_arg(argc, argv, 6);
        const std::string labels_path = optional_arg(argc, argv, 7);
        const bool image_mode = !image_dir.empty();
        const std::vector<std::string> image_paths =
            image_mode ? list_image_files(image_dir)
                       : std::vector<std::string>{};
        const std::vector<std::string> labels =
            image_mode ? load_labels(labels_path) : std::vector<std::string>{};

        exam::ExamClient client(
            sg_sequence_file_path,
            RESNET18_INPUT_BYTES,
            RESNET18_OUTPUT_BYTES);

        std::cout << "resnet18_process pid=" << getpid()
                  << " channel=" << client.channel_name()
                  << " input_size=" << RESNET18_INPUT_BYTES
                  << " output_size=" << RESNET18_OUTPUT_BYTES
                  << " repeat_count=" << repeat_count
                  << " image_mode=" << (image_mode ? "on" : "off")
                  << " image_count=" << image_paths.size();
        if (image_mode) {
            std::cout << " image_dir=" << image_dir;
        } else {
            std::cout << " input_id=" << input_id;
        }
        std::cout << " sg_sequence_file=" << sg_sequence_file_path << '\n';

        create_ready_file(ready_dir, input_id);
        wait_for_gate(gate_path);

        for (int iteration = 0; iteration < repeat_count; ++iteration) {
            const int iteration_input_id = input_id + iteration;
            const std::string image_path =
                image_mode
                    ? image_paths[static_cast<std::size_t>(iteration)
                                  % image_paths.size()]
                    : "";
            exam::Payload input =
                image_mode ? make_resnet18_image_input(image_path)
                           : make_resnet18_input(iteration_input_id);
            const std::string input_hash = hex64(fnv1a64(input.bytes));
            const auto start_time = std::chrono::steady_clock::now();
            if (!client.request(input)) {
                std::cerr << "resnet18_process iteration=" << iteration;
                if (image_mode) {
                    std::cerr << " image=" << std::quoted(base_name(image_path));
                } else {
                    std::cerr << " input_id=" << iteration_input_id;
                }
                std::cerr << " request dropped\n";
                return 2;
            }

            exam::Payload output = client.wait();
            const auto end_time = std::chrono::steady_clock::now();
            const auto latency_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time).count();
            if (image_mode) {
                const Classification classification =
                    classify_resnet18_output(output, labels);
                std::cout << "RESULT sample=" << iteration
                          << " image=" << std::quoted(base_name(image_path))
                          << " input_hash=" << input_hash
                          << " output_hash=" << hex64(fnv1a64(output.bytes))
                          << " output_size=" << output.bytes.size()
                          << " latency_us=" << latency_us
                          << " top1_index=" << classification.index
                          << " top1_label="
                          << std::quoted(classification.label)
                          << " top1_probability=" << std::fixed
                          << std::setprecision(6)
                          << classification.probability
                          << " top1_logit=" << classification.logit;
            } else {
                std::cout << "RESULT input_id=" << iteration_input_id
                          << " iteration=" << iteration
                          << " input_hash=" << input_hash
                          << " output_hash=" << hex64(fnv1a64(output.bytes))
                          << " output_size=" << output.bytes.size()
                          << " latency_us=" << latency_us;
            }
            std::cout << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "resnet18_process: " << e.what() << '\n';
        return 1;
    }
}
