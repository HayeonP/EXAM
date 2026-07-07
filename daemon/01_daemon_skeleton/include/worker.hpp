#pragma once

#include "datatype/payload.hpp"
#include "datatype/request.hpp"
#include "datatype/subgraph.hpp"
#include "thread_config.hpp"

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace exam {

class EventQueue;

class Worker {
public:
    Worker(std::uint32_t worker_id, std::string name, ThreadConfig thread_config);
    virtual ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    std::uint32_t id() const;
    const char* name() const;
    const ThreadConfig& thread_config() const;

    void start(EventQueue* event_queue);
    void stop();
    void assign(const Request& request, const Subgraph& sg);

protected:
    virtual void init() = 0;
    virtual void set_input(const Payload& input) = 0;
    virtual void execute() = 0;
    virtual Payload get_output() = 0;
    virtual void terminate() = 0;

private:
    void run_loop();

    std::uint32_t id_;
    std::string name_;
    ThreadConfig thread_config_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    EventQueue* event_queue_{nullptr};
    Request assigned_request_{};
    const Subgraph* assigned_sg_{nullptr};
    bool has_assignment_{false};
    bool running_{false};
    bool stopping_{false};
};

} // namespace exam
