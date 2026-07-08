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
        MOCK_INTERLEAVING,
    };

    explicit ExamDaemon(
        std::vector<Worker*> workers,
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
        bool sg_running{false};

        bool has_next_sg() const;
        const Subgraph& peek_next_sg() const;
        const Subgraph& take_next_sg();
    };

    class DispatchCandidate {
    public:
        RequestContext* request_context{nullptr};
        Worker* worker{nullptr};

        bool valid() const;
    };

    void handle_submit_request_event(const Event& event);
    void handle_sg_complete_event(const Event& event);
    void handle_request_complete_event(const Event& event);
    void handle_register_client_event(const Event& event);
    void handle_unregister_client_event(const Event& event);
    const char* scheduling_policy_name() const;
    DispatchCandidate select_dispatch_candidate();
    DispatchCandidate select_dispatch_candidate_mock_fifo();
    DispatchCandidate select_dispatch_candidate_mock_interleaving();
    Worker* select_worker(const Subgraph& sg);
    void dispatch_next_sg();
    void launch(Worker& worker, const Request& request, const Subgraph& sg);
    bool client_has_active_request(const std::string& channel_name) const;
    void retain_sg_sequence(const ClientContext& client_context);
    std::size_t release_sg_sequence_reference(const ClientContext& client_context);
    void release_client_resources(const ClientContext& client_context);
    void cleanup_inactive_clients();

    SharedMemoryRegion event_queue_shm_;
    EventQueue* event_queue_{nullptr};
    std::vector<Worker*> workers_;
    SchedulingPolicy scheduling_policy_{SchedulingPolicy::MOCK_FIFO};
    std::unordered_map<std::string, ClientContext> active_client_map_;
    std::unordered_map<std::string, RequestContext> active_request_map_;
    std::unordered_map<std::string, std::size_t> sg_sequence_ref_counts_;
};

} // namespace exam
