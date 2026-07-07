#pragma once

#include "datatype/event.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <pthread.h>

namespace exam {

constexpr std::size_t MAX_EVENT_QUEUE_SIZE = 1024;

class EventQueue {
public:
    bool initialized;
    bool priority_inheritance_enabled;
    pthread_mutex_t mutex;
    pthread_cond_t daemon_cv;

    std::uint32_t event_head;
    std::uint32_t event_tail;
    std::uint32_t event_count;
    std::array<Event, MAX_EVENT_QUEUE_SIZE> events;

    void post_event(const Event& event);
    Event wait_event();
};

} // namespace exam
