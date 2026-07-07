#include "exam_daemon.hpp"

#include "datatype/channel.hpp"
#include "datatype/client_control.hpp"
#include "ipc/process_shared_synchronizer.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace exam {
namespace {

constexpr std::size_t MOCK_SG_COUNT = 3;
constexpr const char* EVENT_QUEUE_SHM_NAME = "/exam_event_queue";
constexpr const char* CHANNEL_SHM_PREFIX = "/exam_channel_";
constexpr const char* CLIENT_CONTROL_SHM_PREFIX = "/exam_client_control_";

void initialize_event_queue(EventQueue* queue) {
    queue->initialized = false;
    queue->priority_inheritance_enabled = false;
    queue->event_head = 0;
    queue->event_tail = 0;
    queue->event_count = 0;
    queue->events = {};
    ProcessSharedSynchronizer::init_mutex(
        &queue->mutex,
        true,
        &queue->priority_inheritance_enabled);
    ProcessSharedSynchronizer::init_cond(&queue->daemon_cv);
    queue->initialized = true;
}

void cleanup_regions_with_pid_prefix(const char* fs_prefix, std::size_t fs_prefix_len) {
    DIR* dir = opendir("/dev/shm");
    if (dir == nullptr) {
        return;
    }

    while (dirent* entry = readdir(dir)) {
        if (std::strncmp(entry->d_name, fs_prefix, fs_prefix_len) != 0) {
            continue;
        }

        char* end = nullptr;
        const long pid_value = std::strtol(entry->d_name + fs_prefix_len, &end, 10);
        if (end == entry->d_name + fs_prefix_len || *end != '\0' || pid_value <= 0) {
            continue;
        }

        const pid_t pid = static_cast<pid_t>(pid_value);
        if (kill(pid, 0) == -1 && errno == ESRCH) {
            SharedMemoryRegion::unlink_noexcept((std::string("/") + entry->d_name).c_str());
        }
    }

    closedir(dir);
}

void cleanup_stale_shared_memory() {
    SharedMemoryRegion::unlink_noexcept(EVENT_QUEUE_SHM_NAME);
    cleanup_regions_with_pid_prefix("exam_channel_", 13);
    cleanup_regions_with_pid_prefix("exam_client_control_", 20);
}

std::string make_client_control_name_from_channel_name(const std::string& channel_name) {
    const std::string channel_prefix = CHANNEL_SHM_PREFIX;
    if (channel_name.rfind(channel_prefix, 0) != 0) {
        throw std::runtime_error("channel name has unexpected prefix");
    }

    return std::string(CLIENT_CONTROL_SHM_PREFIX) + channel_name.substr(channel_prefix.size());
}

std::string make_request_key(const Request& request) {
    return request.channel_name_string() + "#" + std::to_string(request.request_id);
}

} // namespace

ExamDaemon::ExamDaemon(Worker& worker, SchedulingPolicy scheduling_policy)
    : worker_(worker),
      scheduling_policy_(scheduling_policy) { // FUTHER_WORKS: 여기서 worker 종류를 여러개 받게 해야 함
    cleanup_stale_shared_memory();
    event_queue_shm_ = SharedMemoryRegion::create(EVENT_QUEUE_SHM_NAME, sizeof(EventQueue));
    event_queue_ = static_cast<EventQueue*>(event_queue_shm_.data());
    initialize_event_queue(event_queue_);
}

ExamDaemon::~ExamDaemon() {
    SharedMemoryRegion::unlink_noexcept(EVENT_QUEUE_SHM_NAME);
}

bool ExamDaemon::RequestContext::has_next_sg() const {
    return sg_sequence != nullptr && next_sg_index < sg_sequence->size();
}

const Subgraph& ExamDaemon::RequestContext::take_next_sg() {
    if (!has_next_sg()) {
        throw std::runtime_error("request has no pending SG");
    }

    return (*sg_sequence)[next_sg_index++];
}

int ExamDaemon::run_loop() {
    // FURTHER_WORKS: init 단계에서 worker들 객체 받아놓고 여기서 다 시작시키도록. worker 시작이 끝난 뒤 loop가 시작되야 함.
    worker_.start(event_queue_);

    std::cout << "daemon: ready, event_queue_shm=" << EVENT_QUEUE_SHM_NAME
              << ", worker=" << worker_.name()
              << ", scheduling_policy=" << scheduling_policy_name()
              << ", priority inheritance="
              << (event_queue_->priority_inheritance_enabled ? "on" : "off") << '\n';

    while (true) {
        Event event = event_queue_->wait_event();

        switch (event.type) {
            case EVENT_REGISTER_CLIENT:
                handle_register_client_event(event);
                break;

            case EVENT_REQUEST_SUBMIT:
                handle_submit_request_event(event);
                break;

            case EVENT_SG_COMPLETE:
                handle_sg_complete_event(event);
                break;

            case EVENT_REQUEST_COMPLETE:
                handle_request_complete_event(event);
                break;

            default:
                std::cerr << "daemon: unknown event.type " << event.type << '\n';
                break;
        }

        cleanup_regions_with_pid_prefix("exam_channel_", 13);
        cleanup_regions_with_pid_prefix("exam_client_control_", 20);
    }
}

void ExamDaemon::handle_submit_request_event(const Event& event) {
    const Request& request = event.request;
    const std::string channel_name = request.channel_name_string();
    const std::string key = make_request_key(request);
    if (active_request_map_.find(key) != active_request_map_.end()) {
        std::cerr << "daemon: duplicate active request " << request.request_id
                  << " channel=" << channel_name << '\n';
        return;
    }

    auto client_it = active_client_map_.find(channel_name);
    
    if (client_it == active_client_map_.end()) {
        std::cerr << "daemon: reject request from unregistered client request="
                  << request.request_id
                  << " channel=" << channel_name << '\n';
        try {
            ClientControl client_control(
                make_client_control_name_from_channel_name(channel_name));
            client_control.complete_request();
        } catch (const std::exception& e) {
            std::cerr << "daemon: failed to unblock rejected request: "
                      << e.what() << '\n';
        }
        return;
    }

    std::cout << "daemon: request " << request.request_id
              << " channel=" << channel_name
              << " -> add to active_request_map\n";

    RequestContext request_context{};
    request_context.request = request;
    request_context.channel_name = channel_name;
    request_context.next_sg_index = 0;
    request_context.sg_sequence = &client_it->second.sg_sequence;

    active_request_map_.emplace(key, std::move(request_context));
    dispatch_next_sg();
}

void ExamDaemon::handle_sg_complete_event(const Event& event) {
    const Request& request = event.request;
    const std::string channel_name = request.channel_name_string();
    const std::string key = make_request_key(request);
    worker_busy_ = false;

    auto it = active_request_map_.find(key);
    if (it == active_request_map_.end()) {
        std::cerr << "daemon: SG complete for unknown request " << request.request_id
                  << " channel=" << channel_name << '\n';
        return;
    }

    std::cout << "daemon: SG complete request=" << request.request_id
              << " channel=" << channel_name << '\n';

    dispatch_next_sg();
}

void ExamDaemon::handle_request_complete_event(const Event& event) {
    const Request& request = event.request;
    const std::string channel_name = request.channel_name_string();
    const std::string key = make_request_key(request);
    worker_busy_ = false;

    auto it = active_request_map_.find(key);
    if (it == active_request_map_.end()) {
        std::cerr << "daemon: request complete for unknown request " << request.request_id
                  << " channel=" << channel_name << '\n';
        return;
    }

    try {
        std::cout << "daemon: request complete request=" << request.request_id
                  << " channel=" << channel_name << '\n';
        active_request_map_.erase(it);
        ClientControl client_control(
            make_client_control_name_from_channel_name(channel_name));
        client_control.complete_request();        
    } catch (const std::exception& e) {
        std::cerr << "daemon: request complete failed: " << e.what() << '\n';
    }

    dispatch_next_sg();
}

void ExamDaemon::handle_register_client_event(const Event& event) {
    const std::string channel_name = event.request.channel_name_string();
    if (active_client_map_.find(channel_name) != active_client_map_.end()) {
        return;
    }

    const std::string sg_sequence_file_path = event.sg_sequence_file_path_string();

    // FURTHER_WORKS: sg path에서 읽어서 parsing해서 sequence 만들도록 바꿔야 함
    ClientContext client_context{};
    client_context.channel_name = channel_name;
    client_context.sg_sequence_file_path = sg_sequence_file_path;
    client_context.sg_sequence = Subgraph::make_mock_sg_sequence(MOCK_SG_COUNT);

    active_client_map_[channel_name] = std::move(client_context);

    std::cout << "daemon: registered client channel=" << channel_name
              << " sg_sequence_file=" << sg_sequence_file_path
              << " sg_count=" << MOCK_SG_COUNT
              << '\n';
}

const char* ExamDaemon::scheduling_policy_name() const {
    switch (scheduling_policy_) {
        case SchedulingPolicy::MOCK_FIFO:
            return "mock-fifo";
    }

    return "unknown";
}

ExamDaemon::RequestContext* ExamDaemon::select_request() {
    switch (scheduling_policy_) {
        case SchedulingPolicy::MOCK_FIFO:
            return select_request_mock_fifo();
    }

    return nullptr;
}

ExamDaemon::RequestContext* ExamDaemon::select_request_mock_fifo() {
    RequestContext* selected = nullptr;

    for (auto& entry : active_request_map_) {
        RequestContext& request_context = entry.second;
        if (!request_context.has_next_sg()) {
            continue;
        }

        if (selected == nullptr
            || request_context.request.release_time_ns < selected->request.release_time_ns) {
            selected = &request_context;
        }
    }

    return selected;
}

void ExamDaemon::dispatch_next_sg() {
    // FURTHER_WORKS: busy관련된 작업은 나중에 잘 바꿔야 함
    if (worker_busy_) {
        return;
    }

    RequestContext* request_context = select_request();
    if (request_context == nullptr || !request_context->has_next_sg()) {
        return;
    }

    const Subgraph& sg = request_context->take_next_sg();

    std::cout << "daemon: launch SG request=" << request_context->request.request_id
              << " sg=" << sg.id()
              << " worker=" << worker_.name() << '\n';

    launch(request_context->request, sg);
    worker_busy_ = true;
}

void ExamDaemon::launch(const Request& request, const Subgraph& sg) {
    worker_.assign(request, sg);
}

} // namespace exam
