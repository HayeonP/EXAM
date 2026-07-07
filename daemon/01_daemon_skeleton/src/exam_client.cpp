#include "exam_client.hpp"

#include "ipc/process_shared_synchronizer.hpp"

#include <csignal>
#include <ctime>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace exam {
namespace {

constexpr std::size_t CLEANUP_NAME_SIZE = 128;
constexpr const char* EVENT_QUEUE_SHM_NAME = "/exam_event_queue";
constexpr const char* CHANNEL_SHM_PREFIX = "/exam_channel_";
constexpr const char* CLIENT_CONTROL_SHM_PREFIX = "/exam_client_control_";

char channel_name_for_cleanup[CLEANUP_NAME_SIZE]{};
char client_control_name_for_cleanup[CLEANUP_NAME_SIZE]{};

std::string make_channel_name() {
    return std::string(CHANNEL_SHM_PREFIX) + std::to_string(getpid());
}

std::string make_client_control_name() {
    return std::string(CLIENT_CONTROL_SHM_PREFIX) + std::to_string(getpid());
}

std::size_t channel_control_block_size(std::size_t input_size, std::size_t output_size) {
    return sizeof(ChannelControlBlock) + input_size + output_size;
}

std::uint64_t current_release_time_ns() {
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        throw std::runtime_error("clock_gettime CLOCK_MONOTONIC failed");
    }

    constexpr std::uint64_t NS_PER_SECOND = 1000000000ULL;
    return static_cast<std::uint64_t>(timestamp.tv_sec) * NS_PER_SECOND
        + static_cast<std::uint64_t>(timestamp.tv_nsec);
}

void cleanup_channel_and_exit(int signo) {
    SharedMemoryRegion::unlink_noexcept(channel_name_for_cleanup);
    SharedMemoryRegion::unlink_noexcept(client_control_name_for_cleanup);
    _exit(128 + signo);
}

void set_cleanup_name(char* destination, std::size_t destination_size, const std::string& name) {
    if (name.size() >= destination_size) {
        throw std::runtime_error("shared memory cleanup name is too long");
    }

    std::memset(destination, 0, destination_size);
    std::memcpy(destination, name.c_str(), name.size());
}

void install_channel_cleanup_signal(
    const std::string& channel_name,
    const std::string& client_control_name) {
    std::memset(channel_name_for_cleanup, 0, sizeof(channel_name_for_cleanup));
    std::memset(client_control_name_for_cleanup, 0, sizeof(client_control_name_for_cleanup));
    set_cleanup_name(channel_name_for_cleanup, sizeof(channel_name_for_cleanup), channel_name);
    set_cleanup_name(
        client_control_name_for_cleanup,
        sizeof(client_control_name_for_cleanup),
        client_control_name);

    std::signal(SIGINT, cleanup_channel_and_exit);
    std::signal(SIGTERM, cleanup_channel_and_exit);
    std::signal(SIGHUP, cleanup_channel_and_exit);
    std::signal(SIGQUIT, cleanup_channel_and_exit);
}

} // namespace

ExamClient::ExamClient(
    std::string sg_sequence_file_path,
    std::size_t input_size,
    std::size_t output_size)
    : sg_sequence_file_path_(std::move(sg_sequence_file_path)) {
    try {
        event_queue_shm_ = SharedMemoryRegion::open(EVENT_QUEUE_SHM_NAME);
        event_queue_ = static_cast<EventQueue*>(event_queue_shm_.data());
        if (!event_queue_->initialized) {
            throw std::runtime_error("event queue is not initialized");
        }

        channel_name_ = make_channel_name();
        channel_shm_ = SharedMemoryRegion::create(
            channel_name_,
            channel_control_block_size(input_size, output_size));
        control_block_ = static_cast<ChannelControlBlock*>(channel_shm_.data());
        control_block_->initialize(input_size, output_size);

        client_control_name_ = make_client_control_name();
        client_control_shm_ = SharedMemoryRegion::create(
            client_control_name_,
            sizeof(ClientControlBlock));
        auto* client_control_block = static_cast<ClientControlBlock*>(client_control_shm_.data());
        client_control_block->initialize();
        client_control_.bind(client_control_block);

        install_channel_cleanup_signal(channel_name_, client_control_name_);
        register_channel();
    } catch (...) {
        if (!channel_name_.empty()) {
            SharedMemoryRegion::unlink_noexcept(channel_name_.c_str());
        }
        if (!client_control_name_.empty()) {
            SharedMemoryRegion::unlink_noexcept(client_control_name_.c_str());
        }
        throw;
    }
}

ExamClient::~ExamClient() {
    if (!channel_name_.empty()) {
        SharedMemoryRegion::unlink_noexcept(channel_name_.c_str());
    }
    if (!client_control_name_.empty()) {
        SharedMemoryRegion::unlink_noexcept(client_control_name_.c_str());
    }
}

bool ExamClient::request(const Payload& input) {
    if (request_in_flight_) {
        ++dropped_requests_;
        return false;
    }

    ChannelControlBlock* control_block = control_block_;
    ProcessSharedSynchronizer::check(
        pthread_mutex_lock(&control_block->mutex),
        "pthread_mutex_lock channel");

    if (input.bytes.size() > control_block->input_size) {
        ProcessSharedSynchronizer::check(
            pthread_mutex_unlock(&control_block->mutex),
            "pthread_mutex_unlock channel");
        throw std::runtime_error("input payload is larger than channel input size");
    }

    std::memset(control_block->input_buffer_ptr(), 0, control_block->input_size);
    std::memcpy(control_block->input_buffer_ptr(), input.bytes.data(), input.bytes.size());

    ProcessSharedSynchronizer::check(
        pthread_mutex_unlock(&control_block->mutex),
        "pthread_mutex_unlock channel");

    client_control_.reset_completion();

    Request request{};
    request.request_id = next_request_id_++;
    request.release_time_ns = current_release_time_ns();
    request.set_channel_name(channel_name_);

    Event request_submit_event = Event::create_request_submit_event(request);

    request_in_flight_ = true;

    try {
        event_queue_->post_event(request_submit_event);
    } catch (...) {
        release_active_request();
        throw;
    }

    ++accepted_requests_;
    return true;
}

Payload ExamClient::wait() {
    if (!request_in_flight_) {
        throw std::runtime_error("no active request to wait for");
    }

    client_control_.wait_completion();

    ChannelControlBlock* control_block = control_block_;
    ProcessSharedSynchronizer::check(
        pthread_mutex_lock(&control_block->mutex),
        "pthread_mutex_lock channel");
    try {
        Payload output{};
        output.bytes.assign(
            control_block->output_buffer_ptr(),
            control_block->output_buffer_ptr() + control_block->output_size);
        request_in_flight_ = false;
        ++completed_requests_;

        ProcessSharedSynchronizer::check(
            pthread_mutex_unlock(&control_block->mutex),
            "pthread_mutex_unlock channel");
        return output;
    } catch (...) {
        pthread_mutex_unlock(&control_block->mutex);
        throw;
    }
}

const std::string& ExamClient::channel_name() const {
    return channel_name_;
}

std::uint64_t ExamClient::accepted_requests() const {
    return accepted_requests_;
}

std::uint64_t ExamClient::completed_requests() const {
    return completed_requests_;
}

std::uint64_t ExamClient::dropped_requests() const {
    return dropped_requests_;
}

void ExamClient::register_channel() {
    Event register_client_event = Event::create_register_client_event(channel_name_, sg_sequence_file_path_);
    event_queue_->post_event(register_client_event);
}

void ExamClient::release_active_request() {
    ChannelControlBlock* control_block = control_block_;
    ProcessSharedSynchronizer::check(
        pthread_mutex_lock(&control_block->mutex),
        "pthread_mutex_lock channel");

    std::memset(control_block->input_buffer_ptr(), 0, control_block->input_size);
    std::memset(control_block->output_buffer_ptr(), 0, control_block->output_size);
    request_in_flight_ = false;

    ProcessSharedSynchronizer::check(
        pthread_mutex_unlock(&control_block->mutex),
        "pthread_mutex_unlock channel");
    client_control_.reset_completion();
}

} // namespace exam
