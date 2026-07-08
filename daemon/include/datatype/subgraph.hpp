#pragma once

#include "datatype/subgraph_config.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace exam {

class Subgraph {
public:
    Subgraph() = default;
    Subgraph(
        std::uint32_t id,
        std::string label,
        bool is_first,
        bool is_last);

    std::uint32_t id() const;
    const std::string& label() const;
    bool is_first() const;
    bool is_last() const;
    void add_config(SubgraphConfig config);
    bool supports(const std::string& worker_type) const;
    const SubgraphConfig& config_for(const std::string& worker_type) const;

    static std::vector<Subgraph> make_mock_sg_sequence(std::size_t count);

private:
    std::uint32_t id_{0};
    std::string label_;
    bool is_first_{true};
    bool is_last_{false};
    std::unordered_map<std::string, SubgraphConfig> configs_;
};

} // namespace exam
