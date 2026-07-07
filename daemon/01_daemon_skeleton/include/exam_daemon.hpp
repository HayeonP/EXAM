#pragma once

#include "datatype/client_control.hpp"
#include "datatype/event_queue.hpp"
#include "datatype/request.hpp"
#include "datatype/subgraph.hpp"
#include "ipc/shared_memory_region.hpp"
#include "worker.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace exam {

class ExamDaemon {
public:
    enum class SchedulingPolicy {
        MOCK_FIFO,
    };

    explicit ExamDaemon(
        Worker& worker,
        SchedulingPolicy scheduling_policy = SchedulingPolicy::MOCK_FIFO);
    ~ExamDaemon();

    int run_loop();

private:
    class ClientContext {
    public:
        std::string channel_name;
        std::string sg_sequence_file_path;
        std::vector<Subgraph> sg_sequence;
    };

    class RequestContext {
    public:
        Request request;
        std::string channel_name;
        const std::vector<Subgraph>* sg_sequence{nullptr};
        std::size_t next_sg_index{0};

        bool has_next_sg() const;
        const Subgraph& take_next_sg();
    };

    void handle_submit_request_event(const Event& event);
    void handle_sg_complete_event(const Event& event);
    void handle_request_complete_event(const Event& event);
    void handle_register_client_event(const Event& event);
    const char* scheduling_policy_name() const;
    RequestContext* select_request();
    RequestContext* select_request_mock_fifo();
    void dispatch_next_sg();
    void launch(const Request& request, const Subgraph& sg);

    SharedMemoryRegion event_queue_shm_;
    EventQueue* event_queue_{nullptr};
    Worker& worker_;
    SchedulingPolicy scheduling_policy_{SchedulingPolicy::MOCK_FIFO};
    std::unordered_map<std::string, ClientContext> active_client_map_;
    std::unordered_map<std::string, RequestContext> active_request_map_;
    bool worker_busy_{false};
};

} // namespace exam
