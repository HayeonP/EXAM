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
    Worker(
        std::uint32_t worker_id,
        std::string worker_type,
        std::string name,
        ThreadConfig thread_config);
    virtual ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    std::uint32_t id() const;
    const std::string& type() const;
    const char* name() const;
    const ThreadConfig& thread_config() const;
    bool is_busy() const;
    bool supports(const Subgraph& sg) const;

    void start(EventQueue* event_queue);
    void stop();
    void assign(
        const Request& request,
        const Subgraph& sg,
        Worker* migration_source_worker = nullptr);
    virtual void release_sg_cache(const Subgraph& sg);

protected:
    const Request& current_request() const;
    const Subgraph& current_sg() const;
    bool has_prepared_input() const;

    virtual void init() = 0;
    virtual void set_input(const Payload& input) = 0;
    virtual void execute() = 0;
    virtual Payload get_output() = 0;
    virtual void terminate() = 0;

private:
    void prepare_input();
    void handle_first_sg();
    void handle_last_sg();
    void tensor_migration(Worker& migration_source_worker);
    void retain_output_for_migration();
    void release_output_for_migration() noexcept;
    void release_migration_source() noexcept;
    void run_loop();

    std::uint32_t id_;
    std::string worker_type_;
    std::string name_;
    ThreadConfig thread_config_;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    EventQueue* event_queue_{nullptr};
    Request assigned_request_{};
    const Subgraph* assigned_sg_{nullptr};
    Worker* migration_source_worker_{nullptr};
    std::uint32_t migration_retain_count_{0};
    bool input_prepared_{false};
    bool has_assignment_{false};
    bool running_{false};
    bool stopping_{false};
};

} // namespace exam
