#include "datatype/subgraph.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace exam {
namespace {

constexpr const char* MOCK_WORKER_TYPE = "mock";

} // namespace

Subgraph::Subgraph(
    std::uint32_t id,
    std::string label,
    bool is_first,
    bool is_last)
    : id_(id),
      label_(std::move(label)),
      is_first_(is_first),
      is_last_(is_last) {}

std::uint32_t Subgraph::id() const {
    return id_;
}

const std::string& Subgraph::label() const {
    return label_;
}

bool Subgraph::is_first() const {
    return is_first_;
}

bool Subgraph::is_last() const {
    return is_last_;
}

void Subgraph::add_config(SubgraphConfig config) {
    const std::string worker_type = config.worker_type();
    if (worker_type.empty()) {
        throw std::runtime_error("subgraph config has empty worker type");
    }

    configs_[worker_type] = std::move(config);
}

bool Subgraph::supports(const std::string& worker_type) const {
    return configs_.find(worker_type) != configs_.end();
}

const SubgraphConfig& Subgraph::config_for(const std::string& worker_type) const {
    auto it = configs_.find(worker_type);
    if (it == configs_.end()) {
        throw std::runtime_error("SG does not support worker type: " + worker_type);
    }

    return it->second;
}

std::vector<Subgraph> Subgraph::make_mock_sg_sequence(std::size_t count) {
    std::vector<Subgraph> sg_sequence;
    sg_sequence.reserve(count);

    for (std::uint32_t id = 0; id < count; ++id) {
        const bool is_first_subgraph = (id == 0);
        const bool is_final_subgraph = (id + 1 == count);

        Subgraph sg(
            id,
            "mock_sg_" + std::to_string(id),
            is_first_subgraph,
            is_final_subgraph);
        sg.add_config(SubgraphConfig(
            MOCK_WORKER_TYPE,
            "mock_sg_" + std::to_string(id),
            {},
            "",
            {},
            ""));
        sg_sequence.push_back(std::move(sg));
    }

    return sg_sequence;
}

} // namespace exam
