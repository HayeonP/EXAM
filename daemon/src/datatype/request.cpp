#include "datatype/request.hpp"

#include <cstring>
#include <stdexcept>

namespace exam {

void Request::set_channel_name(const std::string& source) {
    if (source.size() >= CHANNEL_NAME_SIZE) {
        throw std::runtime_error("channel shared memory name is too long");
    }

    std::memset(channel_name, 0, CHANNEL_NAME_SIZE);
    std::memcpy(channel_name, source.data(), source.size());
}

std::string Request::channel_name_string() const {
    return std::string(channel_name);
}

} // namespace exam
