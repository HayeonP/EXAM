#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace exam {

constexpr std::size_t CHANNEL_NAME_SIZE = 64;

class Request {
public:
    std::uint64_t request_id{0};
    std::uint64_t release_time_ns{0};
    char channel_name[CHANNEL_NAME_SIZE]{};

    void set_channel_name(const std::string& source);
    std::string channel_name_string() const;
};

} // namespace exam
