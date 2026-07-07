#pragma once

#include <string>
#include <vector>

namespace exam {

class Payload {
public:
    std::vector<char> bytes;

    static Payload from_text(const std::string& text);
    std::string to_string() const;
};

} // namespace exam
