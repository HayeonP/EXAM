#include "datatype/subgraph.hpp"

namespace exam {

Subgraph::Subgraph(
    std::uint32_t id,
    bool is_first,
    bool is_last)
    : id_(id),
      is_first_(is_first),
      is_last_(is_last) {}

std::uint32_t Subgraph::id() const {
    return id_;
}

bool Subgraph::is_first() const {
    return is_first_;
}

bool Subgraph::is_last() const {
    return is_last_;
}

std::vector<Subgraph> Subgraph::make_mock_sg_sequence(std::size_t count) {
    std::vector<Subgraph> sg_sequence;
    sg_sequence.reserve(count);

    for (std::uint32_t id = 0; id < count; ++id) {
        const bool is_first_subgraph = (id == 0);
        const bool is_final_subgraph = (id + 1 == count);
        sg_sequence.emplace_back(id, is_first_subgraph, is_final_subgraph);
    }

    return sg_sequence;
}

} // namespace exam
