#include "exam_daemon.hpp"

#include "datatype/channel.hpp"
#include "datatype/client_control.hpp"
#include "datatype/subgraph_config.hpp"
#include "ipc/process_shared_synchronizer.hpp"

#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <pthread.h>
#include <regex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
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

std::int64_t elapsed_microseconds(
    std::chrono::steady_clock::time_point start_time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time).count();
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

void cleanup_client_regions() {
    cleanup_regions_with_pid_prefix("exam_channel_", 13);
    cleanup_regions_with_pid_prefix("exam_client_control_", 20);
}

void cleanup_stale_shared_memory() {
    SharedMemoryRegion::unlink_noexcept(EVENT_QUEUE_SHM_NAME);
    cleanup_client_regions();
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

bool shared_memory_exists(const std::string& name) {
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd == -1) {
        return false;
    }

    close(fd);
    return true;
}

std::string parent_directory(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return "";
    }
    return path.substr(0, slash);
}

std::string resolve_sg_path(
    const std::string& sg_sequence_file_path,
    const std::string& sg_path) {
    if (sg_path.empty() || sg_path.front() == '/') {
        return sg_path;
    }

    const std::string parent = parent_directory(sg_sequence_file_path);
    if (parent.empty()) {
        return sg_path;
    }
    return parent + "/" + sg_path;
}

std::string read_text_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return "";
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

std::string json_string_field(const std::string& object, const std::string& key) {
    const std::regex field_regex(
        "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(object, match, field_regex)) {
        return "";
    }

    return match[1].str();
}

std::string json_object_field(const std::string& object, const std::string& key) {
    const std::regex field_regex(
        "\"" + key + "\"\\s*:\\s*\\{");
    std::smatch match;
    if (!std::regex_search(object, match, field_regex)) {
        return "";
    }

    const std::size_t object_begin = static_cast<std::size_t>(
        match.position() + match.length() - 1);
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t index = object_begin; index < object.size(); ++index) {
        const char ch = object[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = in_string;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return object.substr(object_begin, index - object_begin + 1);
            }
        }
    }

    return "";
}

std::vector<std::int64_t> json_int_array_field(
    const std::string& object,
    const std::string& key) {
    const std::regex field_regex(
        "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(object, match, field_regex)) {
        return {};
    }

    std::vector<std::int64_t> values;
    const std::string array_text = match[1].str();
    const std::regex integer_regex("-?[0-9]+");
    auto begin = std::sregex_iterator(array_text.begin(), array_text.end(), integer_regex);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        values.push_back(std::stoll(it->str()));
    }

    return values;
}

std::vector<std::string> top_level_json_objects(const std::string& text) {
    std::vector<std::string> objects;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    std::size_t object_begin = 0;

    for (std::size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = in_string;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            if (depth == 0) {
                object_begin = index;
            }
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                objects.push_back(text.substr(object_begin, index - object_begin + 1));
            }
        }
    }

    return objects;
}

std::size_t matching_brace_end(const std::string& text, std::size_t object_begin) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t index = object_begin; index < text.size(); ++index) {
        const char ch = text[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = in_string;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return index;
            }
        }
    }

    return std::string::npos;
}

std::vector<std::pair<std::string, std::string>> top_level_json_object_fields(
    const std::string& object) {
    std::vector<std::pair<std::string, std::string>> fields;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;

    for (std::size_t index = 0; index < object.size(); ++index) {
        const char ch = object[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = in_string;
            continue;
        }
        if (ch == '"') {
            if (!in_string && depth == 1) {
                const std::size_t key_begin = index + 1;
                const std::size_t key_end = object.find('"', key_begin);
                if (key_end == std::string::npos) {
                    break;
                }

                std::size_t cursor = key_end + 1;
                while (cursor < object.size()
                       && std::isspace(static_cast<unsigned char>(object[cursor]))) {
                    ++cursor;
                }
                if (cursor >= object.size() || object[cursor] != ':') {
                    index = key_end;
                    continue;
                }

                ++cursor;
                while (cursor < object.size()
                       && std::isspace(static_cast<unsigned char>(object[cursor]))) {
                    ++cursor;
                }
                if (cursor < object.size() && object[cursor] == '{') {
                    const std::size_t value_end = matching_brace_end(object, cursor);
                    if (value_end == std::string::npos) {
                        break;
                    }
                    fields.emplace_back(
                        object.substr(key_begin, key_end - key_begin),
                        object.substr(cursor, value_end - cursor + 1));
                    index = value_end;
                    continue;
                }

                index = key_end;
                continue;
            }

            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
        }
    }

    return fields;
}

SubgraphConfig parse_subgraph_config(
    const std::string& sg_sequence_file_path,
    const std::string& worker_type,
    const std::string& worker_config) {
    const std::string input = json_object_field(worker_config, "input");
    const std::string output = json_object_field(worker_config, "output");

    return SubgraphConfig(
        worker_type,
        resolve_sg_path(
            sg_sequence_file_path,
            json_string_field(worker_config, "path")),
        input.empty() ? std::vector<std::int64_t>{} : json_int_array_field(input, "shape"),
        input.empty() ? "" : json_string_field(input, "dtype"),
        output.empty() ? std::vector<std::int64_t>{} : json_int_array_field(output, "shape"),
        output.empty() ? "" : json_string_field(output, "dtype"));
}

std::vector<Subgraph> load_sg_sequence_from_file(
    const std::string& sg_sequence_file_path) {
    const std::string text = read_text_file(sg_sequence_file_path);
    if (text.empty()) {
        return {};
    }

    const std::vector<std::string> objects = top_level_json_objects(text);
    std::vector<Subgraph> sg_sequence;
    sg_sequence.reserve(objects.size());
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const std::string& object = objects[index];
        const std::uint32_t id = static_cast<std::uint32_t>(sg_sequence.size());
        const std::string label = json_string_field(object, "label");

        Subgraph sg(
            id,
            label.empty() ? "sg" + std::to_string(id) : label,
            id == 0,
            index + 1 == objects.size());

        for (const auto& field : top_level_json_object_fields(object)) {
            const std::string& worker_type = field.first;
            const std::string& worker_config = field.second;
            sg.add_config(parse_subgraph_config(
                sg_sequence_file_path,
                worker_type,
                worker_config));
        }

        sg_sequence.push_back(std::move(sg));
    }

    return sg_sequence;
}

} // namespace

ExamDaemon::ExamDaemon(std::vector<Worker*> workers, SchedulingPolicy scheduling_policy)
    : workers_(std::move(workers)),
      scheduling_policy_(scheduling_policy) {
    if (workers_.empty()) {
        throw std::runtime_error("daemon requires at least one worker");
    }

    for (Worker* worker : workers_) {
        if (worker == nullptr) {
            throw std::runtime_error("daemon worker list contains null");
        }
    }

    cleanup_stale_shared_memory();
    event_queue_shm_ = SharedMemoryRegion::create(EVENT_QUEUE_SHM_NAME, sizeof(EventQueue));
    event_queue_ = static_cast<EventQueue*>(event_queue_shm_.data());
    initialize_event_queue(event_queue_);
}

ExamDaemon::~ExamDaemon() {
    for (Worker* worker : workers_) {
        if (worker != nullptr) {
            worker->stop();
        }
    }
    SharedMemoryRegion::unlink_noexcept(EVENT_QUEUE_SHM_NAME);
}

bool ExamDaemon::RequestContext::has_next_sg() const {
    return sg_sequence != nullptr && next_sg_index < sg_sequence->size();
}

const Subgraph& ExamDaemon::RequestContext::peek_next_sg() const {
    if (!has_next_sg()) {
        throw std::runtime_error("request has no pending SG");
    }

    return (*sg_sequence)[next_sg_index];
}

const Subgraph& ExamDaemon::RequestContext::take_next_sg() {
    if (!has_next_sg()) {
        throw std::runtime_error("request has no pending SG");
    }

    return (*sg_sequence)[next_sg_index++];
}

bool ExamDaemon::DispatchCandidate::valid() const {
    return request_context != nullptr && worker != nullptr;
}

int ExamDaemon::run_loop() {
    for (Worker* worker : workers_) {
        worker->start(event_queue_);
    }

    std::cout << "daemon: ready, event_queue_shm=" << EVENT_QUEUE_SHM_NAME
              << ", worker_count=" << workers_.size()
              << ", scheduling_policy=" << scheduling_policy_name()
              << ", priority inheritance="
              << (event_queue_->priority_inheritance_enabled ? "on" : "off") << '\n';

    while (true) {
        Event event = event_queue_->wait_event();

        switch (event.type) {
            case EVENT_REGISTER_CLIENT:
                handle_register_client_event(event);
                break;

            case EVENT_UNREGISTER_CLIENT:
                handle_unregister_client_event(event);
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
    request_context.sg_running = false;

    active_request_map_.emplace(key, std::move(request_context));
    dispatch_next_sg();
}

void ExamDaemon::handle_sg_complete_event(const Event& event) {
    const Request& request = event.request;
    const std::string channel_name = request.channel_name_string();
    const std::string key = make_request_key(request);

    auto it = active_request_map_.find(key);
    if (it == active_request_map_.end()) {
        std::cerr << "daemon: SG complete for unknown request " << request.request_id
                  << " channel=" << channel_name << '\n';
        return;
    }

    RequestContext& request_context = it->second;
    const auto sg_latency_us =
        elapsed_microseconds(request_context.sg_started_at);

    {
        std::ostringstream log;
        log << "daemon: SG complete request=" << request.request_id
            << " channel=" << channel_name
            << " sg=" << request_context.running_sg_id
            << " sg_latency_us=" << sg_latency_us << '\n';
        std::cout << log.str();
    }

    request_context.sg_running = false;
    dispatch_next_sg();
}

void ExamDaemon::handle_request_complete_event(const Event& event) {
    const Request& request = event.request;
    const std::string channel_name = request.channel_name_string();
    const std::string key = make_request_key(request);

    auto it = active_request_map_.find(key);
    if (it == active_request_map_.end()) {
        std::cerr << "daemon: request complete for unknown request " << request.request_id
                  << " channel=" << channel_name << '\n';
        return;
    }

    try {
        RequestContext& request_context = it->second;
        const auto sg_latency_us =
            elapsed_microseconds(request_context.sg_started_at);

        {
            std::ostringstream log;
            log << "daemon: request complete request=" << request.request_id
                << " channel=" << channel_name
                << " sg=" << request_context.running_sg_id
                << " sg_latency_us=" << sg_latency_us << '\n';
            std::cout << log.str();
        }
        active_request_map_.erase(it);
        complete_client_request(channel_name);
    } catch (const std::exception& e) {
        std::cerr << "daemon: request complete failed: " << e.what() << '\n';
    }

    dispatch_next_sg();
}

void ExamDaemon::handle_register_client_event(const Event& event) {
    const std::string channel_name = event.request.channel_name_string();
    if (active_client_map_.find(channel_name) != active_client_map_.end()) {
        try {
            ClientControl client_control(
                make_client_control_name_from_channel_name(channel_name));
            client_control.complete_request();
        } catch (const std::exception& e) {
            std::cerr << "daemon: failed to acknowledge duplicate client registration: "
                      << e.what() << '\n';
        }
        return;
    }

    const std::string sg_sequence_file_path = event.sg_sequence_file_path_string();

    ClientContext client_context{};
    client_context.channel_name = channel_name;
    client_context.sg_sequence_file_path = sg_sequence_file_path;
    client_context.client_control = std::make_unique<ClientControl>(
        make_client_control_name_from_channel_name(channel_name));
    // TODO: Replace this line-based parser after the official SG sequence format is fixed.
    client_context.sg_sequence = load_sg_sequence_from_file(sg_sequence_file_path);
    if (client_context.sg_sequence.empty()) {
        client_context.sg_sequence = Subgraph::make_mock_sg_sequence(MOCK_SG_COUNT);
    }

    auto insert_result =
        active_client_map_.emplace(channel_name, std::move(client_context));
    retain_sg_sequence(insert_result.first->second);

    std::cout << "daemon: registered client channel=" << channel_name
              << " sg_sequence_file=" << sg_sequence_file_path
              << " sg_count=" << insert_result.first->second.sg_sequence.size()
              << " sg_sequence_ref_count="
              << sg_sequence_ref_counts_[sg_sequence_file_path]
              << '\n';

    try {
        complete_client_request(channel_name);
    } catch (const std::exception& e) {
        std::cerr << "daemon: failed to acknowledge client registration: "
                  << e.what() << '\n';
    }
}

void ExamDaemon::handle_unregister_client_event(const Event& event) {
    const std::string channel_name = event.request.channel_name_string();
    auto client_it = active_client_map_.find(channel_name);
    if (client_it == active_client_map_.end()) {
        cleanup_client_regions();
        cleanup_inactive_clients();
        return;
    }

    if (client_has_active_request(channel_name)) {
        std::cerr << "daemon: defer unregister for active client channel="
                  << channel_name << '\n';
        cleanup_client_regions();
        cleanup_inactive_clients();
        return;
    }

    const std::size_t remaining_refs =
        release_sg_sequence_reference(client_it->second);
    if (remaining_refs > 0) {
        std::cout << "daemon: unregistered client channel=" << channel_name
                  << " -> keep shared SG cache, sg_sequence_ref_count="
                  << remaining_refs << '\n';
    } else {
        std::cout << "daemon: unregistered client channel=" << channel_name
                  << " -> release SG cache\n";
        release_client_resources(client_it->second);
    }
    active_client_map_.erase(client_it);
    cleanup_client_regions();
    cleanup_inactive_clients();
}

void ExamDaemon::complete_client_request(const std::string& channel_name) {
    auto client_it = active_client_map_.find(channel_name);
    if (client_it != active_client_map_.end() && client_it->second.client_control) {
        client_it->second.client_control->complete_request();
        return;
    }

    ClientControl client_control(
        make_client_control_name_from_channel_name(channel_name));
    client_control.complete_request();
}

const char* ExamDaemon::scheduling_policy_name() const {
    switch (scheduling_policy_) {
        case SchedulingPolicy::MOCK_FIFO:
            return "mock-fifo";
        case SchedulingPolicy::MOCK_INTERLEAVING:
            return "mock-interleaving";
    }

    return "unknown";
}

ExamDaemon::DispatchCandidate ExamDaemon::select_dispatch_candidate() {
    switch (scheduling_policy_) {
        case SchedulingPolicy::MOCK_FIFO:
            return select_dispatch_candidate_mock_fifo();
        case SchedulingPolicy::MOCK_INTERLEAVING:
            return select_dispatch_candidate_mock_interleaving();
    }

    return {};
}

ExamDaemon::DispatchCandidate ExamDaemon::select_dispatch_candidate_mock_fifo() {
    DispatchCandidate selected{};

    for (auto& entry : active_request_map_) {
        RequestContext& request_context = entry.second;
        if (request_context.sg_running || !request_context.has_next_sg()) {
            continue;
        }

        const Subgraph& sg = request_context.peek_next_sg();
        Worker* worker = select_worker(sg);
        if (worker == nullptr) {
            continue;
        }

        if (!selected.valid()
            || request_context.request.release_time_ns
                < selected.request_context->request.release_time_ns) {
            selected.request_context = &request_context;
            selected.worker = worker;
        }
    }

    return selected;
}

ExamDaemon::DispatchCandidate
ExamDaemon::select_dispatch_candidate_mock_interleaving() {
    DispatchCandidate selected{};

    for (auto& entry : active_request_map_) {
        RequestContext& request_context = entry.second;
        if (request_context.sg_running || !request_context.has_next_sg()) {
            continue;
        }

        const Subgraph& sg = request_context.peek_next_sg();
        Worker* worker = select_worker(sg);
        if (worker == nullptr) {
            continue;
        }

        if (!selected.valid()
            || request_context.next_sg_index
                < selected.request_context->next_sg_index
            || (request_context.next_sg_index
                    == selected.request_context->next_sg_index
                && request_context.request.release_time_ns
                    < selected.request_context->request.release_time_ns)
            || (request_context.next_sg_index
                    == selected.request_context->next_sg_index
                && request_context.request.release_time_ns
                    == selected.request_context->request.release_time_ns
                && entry.first
                    < make_request_key(selected.request_context->request))) {
            selected.request_context = &request_context;
            selected.worker = worker;
        }
    }

    return selected;
}

Worker* ExamDaemon::select_worker(const Subgraph& sg) {
    for (Worker* worker : workers_) {
        if (worker->is_busy() || !worker->supports(sg)) {
            continue;
        }
        return worker;
    }

    return nullptr;
}

void ExamDaemon::dispatch_next_sg() {
    while (true) {
        DispatchCandidate candidate = select_dispatch_candidate();
        if (!candidate.valid()) {
            return;
        }

        const Subgraph& sg = candidate.request_context->take_next_sg();
        candidate.request_context->sg_running = true;
        candidate.request_context->running_sg_id = sg.id();
        candidate.request_context->sg_started_at = std::chrono::steady_clock::now();

        {
            std::ostringstream log;
            log << "daemon: launch SG channel="
                << candidate.request_context->channel_name
                << " request="
                << candidate.request_context->request.request_id
                << " sg=" << sg.id()
                << " worker=" << candidate.worker->name() << '\n';
            std::cout << log.str();
        }

        launch(*candidate.worker, candidate.request_context->request, sg);
    }
}

void ExamDaemon::launch(Worker& worker, const Request& request, const Subgraph& sg) {
    worker.assign(request, sg);
}

bool ExamDaemon::client_has_active_request(const std::string& channel_name) const {
    for (const auto& entry : active_request_map_) {
        if (entry.second.channel_name == channel_name) {
            return true;
        }
    }

    return false;
}

void ExamDaemon::retain_sg_sequence(const ClientContext& client_context) {
    ++sg_sequence_ref_counts_[client_context.sg_sequence_file_path];
}

std::size_t ExamDaemon::release_sg_sequence_reference(
    const ClientContext& client_context) {
    auto ref_count_it =
        sg_sequence_ref_counts_.find(client_context.sg_sequence_file_path);
    if (ref_count_it == sg_sequence_ref_counts_.end()
        || ref_count_it->second == 0) {
        std::cerr << "daemon: SG sequence refcount underflow channel="
                  << client_context.channel_name
                  << " sg_sequence_file="
                  << client_context.sg_sequence_file_path << '\n';
        return 0;
    }

    --ref_count_it->second;
    const std::size_t remaining_refs = ref_count_it->second;
    if (remaining_refs == 0) {
        sg_sequence_ref_counts_.erase(ref_count_it);
    }

    return remaining_refs;
}

void ExamDaemon::release_client_resources(const ClientContext& client_context) {
    for (Worker* worker : workers_) {
        for (const Subgraph& sg : client_context.sg_sequence) {
            if (worker->supports(sg)) {
                worker->release_sg_cache(sg);
            }
        }
    }
}

void ExamDaemon::cleanup_inactive_clients() {
    for (auto it = active_client_map_.begin(); it != active_client_map_.end();) {
        ClientContext& client_context = it->second;
        if (shared_memory_exists(client_context.channel_name)
            || client_has_active_request(client_context.channel_name)) {
            ++it;
            continue;
        }

        const std::size_t remaining_refs =
            release_sg_sequence_reference(client_context);
        if (remaining_refs > 0) {
            std::cout << "daemon: cleanup inactive client channel="
                      << client_context.channel_name
                      << " -> keep shared SG cache, sg_sequence_ref_count="
                      << remaining_refs << '\n';
        } else {
            std::cout << "daemon: cleanup inactive client channel="
                      << client_context.channel_name
                      << " -> release SG cache\n";
            release_client_resources(client_context);
        }
        it = active_client_map_.erase(it);
    }
}

} // namespace exam
