#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace exam {

class Subgraph {
public:
    Subgraph() = default;
    Subgraph(std::uint32_t id, bool is_first, bool is_last);

    std::uint32_t id() const;
    bool is_first() const;
    bool is_last() const;

    static std::vector<Subgraph> make_mock_sg_sequence(std::size_t count);

private:
    std::uint32_t id_{0};
    bool is_first_{true};
    bool is_last_{false};
};

} // namespace exam
