#include "config/cpu_affinity_config.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace exam {
namespace {

std::string trim(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](unsigned char ch) { return ch == ' ' || ch == '\t'; });
    const auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](unsigned char ch) { return ch == ' ' || ch == '\t'; }).base();

    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::string strip_comment(const std::string& line) {
    const std::size_t comment = line.find('#');
    if (comment == std::string::npos) {
        return line;
    }
    return line.substr(0, comment);
}

std::size_t indentation(const std::string& line) {
    std::size_t count = 0;
    while (count < line.size() && line[count] == ' ') {
        ++count;
    }
    return count;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

int parse_cpu_id(const std::string& token, const std::string& path, int line_no) {
    const std::string cleaned = trim(token);
    if (cleaned.empty()) {
        throw std::runtime_error(
            "empty CPU id in " + path + ":" + std::to_string(line_no));
    }

    char* end = nullptr;
    const long parsed = std::strtol(cleaned.c_str(), &end, 10);
    if (end == cleaned.c_str() || *end != '\0' || parsed < 0) {
        throw std::runtime_error(
            "invalid CPU id '" + cleaned + "' in " + path + ":"
            + std::to_string(line_no));
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_inline_cpu_list(
    const std::string& value,
    const std::string& path,
    int line_no) {
    const std::size_t open = value.find('[');
    const std::size_t close = value.find(']', open == std::string::npos ? 0 : open);
    if (open == std::string::npos || close == std::string::npos || close < open) {
        throw std::runtime_error(
            "expected inline CPU list in " + path + ":" + std::to_string(line_no));
    }

    std::vector<int> cpu_ids;
    std::stringstream input(value.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(input, token, ',')) {
        const std::string cleaned = trim(token);
        if (!cleaned.empty()) {
            cpu_ids.push_back(parse_cpu_id(cleaned, path, line_no));
        }
    }
    return cpu_ids;
}

void assign_cpu_list(
    CpuAffinityConfig& config,
    const std::string& key,
    std::vector<int> cpu_ids) {
    if (key == "daemon") {
        config.daemon_cpu_ids = std::move(cpu_ids);
    } else if (key == "tensor_rt_gpu_worker" || key == "tensor_rt_gpu") {
        config.tensor_rt_gpu_worker_cpu_ids = std::move(cpu_ids);
    } else if (key == "pytorch_worker" || key == "pytorch") {
        config.pytorch_worker_cpu_ids = std::move(cpu_ids);
    } else if (key == "worker") {
        if (config.tensor_rt_gpu_worker_cpu_ids.empty()) {
            config.tensor_rt_gpu_worker_cpu_ids = cpu_ids;
        }
        if (config.pytorch_worker_cpu_ids.empty()) {
            config.pytorch_worker_cpu_ids = std::move(cpu_ids);
        }
    }
}

} // anonymous namespace

std::string cpu_ids_to_string(const std::vector<int>& cpu_ids) {
    if (cpu_ids.empty()) {
        return "-";
    }

    std::ostringstream output;
    for (std::size_t index = 0; index < cpu_ids.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << cpu_ids[index];
    }
    return output.str();
}

CpuAffinityConfig CpuAffinityConfig::load_default() {
    if (const char* env_path = std::getenv("EXAM_CONFIG_PATH")) {
        if (env_path[0] != '\0') {
            return load_from_file(env_path);
        }
    }

    for (const char* candidate : {"config/examl.yaml", "config/exam.yaml"}) {
        if (file_exists(candidate)) {
            return load_from_file(candidate);
        }
    }

    return {};
}

CpuAffinityConfig CpuAffinityConfig::load_from_file(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("failed to open config file: " + config_path);
    }

    CpuAffinityConfig config;
    config.path = config_path;

    bool in_cpu_affinity = false;
    std::size_t cpu_affinity_indent = 0;
    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        const std::string without_comment = strip_comment(line);
        const std::string cleaned = trim(without_comment);
        if (cleaned.empty()) {
            continue;
        }

        const std::size_t indent = indentation(without_comment);
        if (cleaned == "cpu_affinity:") {
            in_cpu_affinity = true;
            cpu_affinity_indent = indent;
            continue;
        }

        if (in_cpu_affinity && indent <= cpu_affinity_indent) {
            in_cpu_affinity = false;
        }
        if (!in_cpu_affinity) {
            continue;
        }

        const std::size_t colon = cleaned.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        const std::string key = trim(cleaned.substr(0, colon));
        const std::string value = trim(cleaned.substr(colon + 1));
        if (value.find('[') == std::string::npos) {
            continue;
        }
        assign_cpu_list(
            config,
            key,
            parse_inline_cpu_list(value, config_path, line_no));
    }

    return config;
}

std::vector<int> CpuAffinityConfig::cpus_for_pytorch_direct() const {
    return pytorch_worker_cpu_ids;
}

std::string CpuAffinityConfig::summary() const {
    std::ostringstream output;
    output << "config=" << (path.empty() ? "-" : path)
           << " daemon=[" << cpu_ids_to_string(daemon_cpu_ids) << ']'
           << " tensor_rt_gpu_worker=["
           << cpu_ids_to_string(tensor_rt_gpu_worker_cpu_ids) << ']'
           << " pytorch_worker=[" << cpu_ids_to_string(pytorch_worker_cpu_ids)
           << ']';
    return output.str();
}

} // namespace exam
