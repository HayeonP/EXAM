#include "datatype/subgraph_config.hpp"

#include <utility>

namespace exam {

SubgraphConfig::SubgraphConfig(
    std::string worker_type,
    std::string path,
    std::vector<std::int64_t> input_shape,
    std::string input_dtype,
    std::vector<std::int64_t> output_shape,
    std::string output_dtype)
    : worker_type_(std::move(worker_type)),
      path_(std::move(path)),
      input_shape_(std::move(input_shape)),
      input_dtype_(std::move(input_dtype)),
      output_shape_(std::move(output_shape)),
      output_dtype_(std::move(output_dtype)) {}

const std::string& SubgraphConfig::worker_type() const {
    return worker_type_;
}

const std::string& SubgraphConfig::path() const {
    return path_;
}

const std::vector<std::int64_t>& SubgraphConfig::input_shape() const {
    return input_shape_;
}

const std::string& SubgraphConfig::input_dtype() const {
    return input_dtype_;
}

const std::vector<std::int64_t>& SubgraphConfig::output_shape() const {
    return output_shape_;
}

const std::string& SubgraphConfig::output_dtype() const {
    return output_dtype_;
}

bool SubgraphConfig::has_input_tensor_metadata() const {
    return !input_shape_.empty() && !input_dtype_.empty();
}

bool SubgraphConfig::has_output_tensor_metadata() const {
    return !output_shape_.empty() && !output_dtype_.empty();
}

} // namespace exam
