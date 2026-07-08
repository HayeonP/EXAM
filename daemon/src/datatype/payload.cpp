#include "datatype/payload.hpp"

namespace exam {

Payload Payload::from_text(const std::string& text) {
    Payload payload{};
    payload.bytes.assign(text.begin(), text.end());
    return payload;
}

std::string Payload::to_string() const {
    auto end = bytes.begin();
    while (end != bytes.end() && *end != '\0') {
        ++end;
    }
    return std::string(bytes.begin(), end);
}

} // namespace exam
