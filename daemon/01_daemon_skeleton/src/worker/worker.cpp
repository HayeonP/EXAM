#include "worker.hpp"

#include "datatype/channel.hpp"
#include "datatype/event_queue.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace exam {

Worker::Worker(std::uint32_t worker_id, std::string name, ThreadConfig thread_config)
    : id_(worker_id),
      name_(std::move(name)),
      thread_config_(std::move(thread_config)) {}

Worker::~Worker() {
    stop();
}

std::uint32_t Worker::id() const {
    return id_;
}

const char* Worker::name() const {
    return name_.c_str();
}

const ThreadConfig& Worker::thread_config() const {
    return thread_config_;
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

void Worker::assign(const Request& request, const Subgraph& sg) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_assignment_ || running_) {
            throw std::runtime_error("worker is not ready for another SG");
        }

        assigned_request_ = request;
        assigned_sg_ = &sg;
        has_assignment_ = true;
    }
    cv_.notify_one();
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

                Channel channel(assigned_request_.channel_name_string());

                if (assigned_sg_->is_first()) {
                    set_input(channel.read_input());
                }

                execute();

                if (assigned_sg_->is_last()) {
                    channel.write_output(get_output());
                }

                Event complete_event =
                    assigned_sg_->is_last()
                        ? Event::create_request_complete_event(assigned_request_)
                        : Event::create_sg_complete_event(assigned_request_);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    assigned_sg_ = nullptr;
                    has_assignment_ = false;
                    running_ = false;
                }
                event_queue_->post_event(complete_event);
            } catch (const std::exception& e) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    assigned_sg_ = nullptr;
                    has_assignment_ = false;
                    running_ = false;
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
