#include "worker.hpp"

#include "datatype/channel.hpp"
#include "datatype/event_queue.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace exam {

Worker::Worker(
    std::uint32_t worker_id,
    std::string worker_type,
    std::string name,
    ThreadConfig thread_config)
    : id_(worker_id),
      worker_type_(std::move(worker_type)),
      name_(std::move(name)),
      thread_config_(std::move(thread_config)) {}

Worker::~Worker() {
    stop();
}

std::uint32_t Worker::id() const {
    return id_;
}

const std::string& Worker::type() const {
    return worker_type_;
}

const char* Worker::name() const {
    return name_.c_str();
}

const ThreadConfig& Worker::thread_config() const {
    return thread_config_;
}

bool Worker::is_busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_assignment_ || running_ || migration_retain_count_ > 0;
}

bool Worker::supports(const Subgraph& sg) const {
    return sg.supports(worker_type_);
}

const Request& Worker::current_request() const {
    return assigned_request_;
}

const Subgraph& Worker::current_sg() const {
    if (assigned_sg_ == nullptr) {
        throw std::runtime_error("worker has no current SG");
    }

    return *assigned_sg_;
}

bool Worker::has_prepared_input() const {
    return input_prepared_;
}

void Worker::start(EventQueue* event_queue) {
    if (event_queue == nullptr) {
        throw std::runtime_error("worker requires event queue");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_.joinable()) {
        return;
    }

    event_queue_ = event_queue;
    stopping_ = false;
    thread_ = std::thread(&Worker::run_loop, this);
}

void Worker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }
}

void Worker::assign(
    const Request& request,
    const Subgraph& sg,
    Worker* migration_source_worker) {
    if (migration_source_worker == this) {
        migration_source_worker = nullptr;
    }
    if (migration_source_worker != nullptr) {
        migration_source_worker->retain_output_for_migration();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_assignment_ || running_) {
            if (migration_source_worker != nullptr) {
                migration_source_worker->release_output_for_migration();
            }
            throw std::runtime_error("worker is not ready for another SG");
        }

        assigned_request_ = request;
        assigned_sg_ = &sg;
        migration_source_worker_ = migration_source_worker;
        input_prepared_ = false;
        has_assignment_ = true;
    }
    cv_.notify_one();
}

void Worker::release_sg_cache(const Subgraph&) {}

void Worker::prepare_input() {
    input_prepared_ = false;

    if (current_sg().is_first()) {
        release_migration_source();
        handle_first_sg();
        return;
    }

    if (migration_source_worker_ != nullptr) {
        try {
            tensor_migration(*migration_source_worker_);
        } catch (...) {
            release_migration_source();
            throw;
        }
        release_migration_source();
    }
}

void Worker::handle_first_sg() {
    Channel channel(assigned_request_.channel_name_string());
    set_input(channel.read_input());
    input_prepared_ = true;
}

void Worker::handle_last_sg() {
    Channel channel(assigned_request_.channel_name_string());
    channel.write_output(get_output());
}

void Worker::tensor_migration(Worker& migration_source_worker) {
    set_input(migration_source_worker.get_output());
    input_prepared_ = true;
}

void Worker::retain_output_for_migration() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++migration_retain_count_;
}

void Worker::release_output_for_migration() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (migration_retain_count_ > 0) {
        --migration_retain_count_;
    }
}

void Worker::release_migration_source() noexcept {
    if (migration_source_worker_ == nullptr) {
        return;
    }

    migration_source_worker_->release_output_for_migration();
    migration_source_worker_ = nullptr;
}

void Worker::run_loop() {
    thread_config_.apply(name());

    try {
        init();

        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return stopping_ || has_assignment_;
                });

                if (stopping_ && !has_assignment_) {
                    break;
                }

                running_ = true;
            }

            try {
                if (assigned_sg_ == nullptr || event_queue_ == nullptr) {
                    throw std::runtime_error("worker assignment is incomplete");
                }

                const bool is_last_sg = assigned_sg_->is_last();

                prepare_input();
                execute();
                if (is_last_sg) {
                    handle_last_sg();
                }

                Event complete_event =
                    is_last_sg
                        ? Event::create_request_complete_event(assigned_request_)
                        : Event::create_sg_complete_event(assigned_request_);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    release_migration_source();
                    assigned_sg_ = nullptr;
                    has_assignment_ = false;
                    running_ = false;
                    input_prepared_ = false;
                }
                event_queue_->post_event(complete_event);
            } catch (const std::exception& e) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    release_migration_source();
                    assigned_sg_ = nullptr;
                    has_assignment_ = false;
                    running_ = false;
                    input_prepared_ = false;
                }
                std::cerr << name() << "-" << id() << ": " << e.what() << '\n';
            }
        }
    } catch (const std::exception& e) {
        std::cerr << name() << "-" << id() << ": worker loop stopped: " << e.what() << '\n';
    }

    try {
        terminate();
    } catch (const std::exception& e) {
        std::cerr << name() << "-" << id() << ": terminate failed: " << e.what() << '\n';
    }
}

} // namespace exam
