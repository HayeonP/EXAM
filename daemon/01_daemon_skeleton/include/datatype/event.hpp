#pragma once

#include "datatype/request.hpp"

#include <cstdint>
#include <string>

namespace exam {

constexpr std::size_t SG_SEQUENCE_FILE_PATH_SIZE = 256;

enum EventType : std::uint32_t {
    EVENT_REGISTER_CLIENT = 1,
    EVENT_REQUEST_SUBMIT = 2,
    EVENT_SG_COMPLETE = 3,
    EVENT_REQUEST_COMPLETE = 4,
};

class Event {
public:
    EventType type{EVENT_REGISTER_CLIENT};
    Request request{};
    char sg_sequence_file_path[SG_SEQUENCE_FILE_PATH_SIZE]{};

    static Event create_register_client_event(
        const std::string& channel_name,
        const std::string& sg_sequence_file_path);
    static Event create_request_submit_event(const Request& request);
    static Event create_sg_complete_event(const Request& request);
    static Event create_request_complete_event(const Request& request);

    void set_sg_sequence_file_path(const std::string& source);
    std::string sg_sequence_file_path_string() const;
};

} // namespace exam
