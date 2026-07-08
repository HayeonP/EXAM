#include "datatype/event.hpp"

#include <cstring>
#include <stdexcept>

namespace exam {

Event Event::create_register_client_event(
    const std::string& channel_name,
    const std::string& sg_sequence_file_path) {
    Event event{};
    event.type = EVENT_REGISTER_CLIENT;
    event.request.set_channel_name(channel_name);
    event.set_sg_sequence_file_path(sg_sequence_file_path);
    return event;
}

Event Event::create_unregister_client_event(const std::string& channel_name) {
    Event event{};
    event.type = EVENT_UNREGISTER_CLIENT;
    event.request.set_channel_name(channel_name);
    return event;
}

Event Event::create_request_submit_event(const Request& request) {
    Event event{};
    event.type = EVENT_REQUEST_SUBMIT;
    event.request = request;
    return event;
}

Event Event::create_sg_complete_event(const Request& request) {
    Event event{};
    event.type = EVENT_SG_COMPLETE;
    event.request = request;
    return event;
}

Event Event::create_request_complete_event(const Request& request) {
    Event event{};
    event.type = EVENT_REQUEST_COMPLETE;
    event.request = request;
    return event;
}

void Event::set_sg_sequence_file_path(const std::string& source) {
    if (source.size() >= SG_SEQUENCE_FILE_PATH_SIZE) {
        throw std::runtime_error("SG sequence file path is too long");
    }

    std::memset(sg_sequence_file_path, 0, SG_SEQUENCE_FILE_PATH_SIZE);
    std::memcpy(sg_sequence_file_path, source.data(), source.size());
}

std::string Event::sg_sequence_file_path_string() const {
    return std::string(sg_sequence_file_path);
}

} // namespace exam
