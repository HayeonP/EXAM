#include "datatype/event_queue.hpp"

#include "ipc/process_shared_synchronizer.hpp"

#include <stdexcept>

namespace exam {
namespace {

bool push_event_locked(EventQueue* queue, const Event& event) {
    if (queue->event_count == MAX_EVENT_QUEUE_SIZE) {
        return false;
    }

    queue->events[queue->event_tail] = event;
    queue->event_tail = (queue->event_tail + 1) % MAX_EVENT_QUEUE_SIZE;
    ++queue->event_count;
    return true;
}

int event_priority(EventType type) {
    switch (type) {
        case EVENT_REQUEST_COMPLETE:
            return 4;
        case EVENT_REQUEST_SUBMIT:
            return 3;
        case EVENT_SG_COMPLETE:
            return 2;
        case EVENT_REGISTER_CLIENT:
        case EVENT_UNREGISTER_CLIENT:
            return 1;
    }

    return 0;
}

std::uint32_t event_index(const EventQueue* queue, std::uint32_t offset) {
    return (queue->event_head + offset) % MAX_EVENT_QUEUE_SIZE;
}

bool pop_event_locked(EventQueue* queue, Event* event) {
    if (queue->event_count == 0) {
        return false;
    }

    std::uint32_t selected_offset = 0;
    int selected_priority =
        event_priority(queue->events[event_index(queue, selected_offset)].type);

    for (std::uint32_t offset = 1; offset < queue->event_count; ++offset) {
        const int priority = event_priority(queue->events[event_index(queue, offset)].type);
        if (priority > selected_priority) {
            selected_offset = offset;
            selected_priority = priority;
        }
    }

    *event = queue->events[event_index(queue, selected_offset)];

    for (std::uint32_t offset = selected_offset; offset + 1 < queue->event_count; ++offset) {
        queue->events[event_index(queue, offset)] =
            queue->events[event_index(queue, offset + 1)];
    }

    queue->event_tail =
        (queue->event_tail + MAX_EVENT_QUEUE_SIZE - 1) % MAX_EVENT_QUEUE_SIZE;
    --queue->event_count;
    return true;
}

} // namespace

void EventQueue::post_event(const Event& event) {
    ProcessSharedSynchronizer::check(
        pthread_mutex_lock(&mutex),
        "pthread_mutex_lock event queue");

    if (!push_event_locked(this, event)) {
        ProcessSharedSynchronizer::check(
            pthread_mutex_unlock(&mutex),
            "pthread_mutex_unlock event queue");
        throw std::runtime_error("event queue is full");
    }

    ProcessSharedSynchronizer::check(
        pthread_cond_signal(&daemon_cv),
        "pthread_cond_signal daemon");
    ProcessSharedSynchronizer::check(
        pthread_mutex_unlock(&mutex),
        "pthread_mutex_unlock event queue");
}

Event EventQueue::wait_event() {
    ProcessSharedSynchronizer::check(
        pthread_mutex_lock(&mutex),
        "pthread_mutex_lock event queue");

    while (event_count == 0) {
        ProcessSharedSynchronizer::check(
            pthread_cond_wait(&daemon_cv, &mutex),
            "pthread_cond_wait daemon");
    }

    Event event{};
    pop_event_locked(this, &event);
    ProcessSharedSynchronizer::check(
        pthread_mutex_unlock(&mutex),
        "pthread_mutex_unlock event queue");
    return event;
}

} // namespace exam
