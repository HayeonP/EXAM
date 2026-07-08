#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace exam {

class SubgraphConfig {
public:
    SubgraphConfig() = default;
    SubgraphConfig(
        std::string worker_type,
        std::string path,
        std::vector<std::int64_t> input_shape,
        std::string input_dtype,
        std::vector<std::int64_t> output_shape,
        std::string output_dtype);

    const std::string& worker_type() const;
    const std::string& path() const;
    const std::vector<std::int64_t>& input_shape() const;
    const std::string& input_dtype() const;
    const std::vector<std::int64_t>& output_shape() const;
    const std::string& output_dtype() const;
    bool has_input_tensor_metadata() const;
    bool has_output_tensor_metadata() const;

private:
    std::string worker_type_;
    std::string path_;
    std::vector<std::int64_t> input_shape_;
    std::string input_dtype_;
    std::vector<std::int64_t> output_shape_;
    std::string output_dtype_;
};

} // namespace exam
