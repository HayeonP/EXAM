#pragma once

#include "ipc/shared_memory_region.hpp"

#include <pthread.h>
#include <string>
#include <sys/types.h>

namespace exam {

class ClientControlBlock {
public:
    bool initialized;
    bool priority_inheritance_enabled;
    bool request_completed;
    pid_t owner_pid;
    pthread_mutex_t mutex;
    pthread_cond_t completed_cv;

    void initialize();
};

class ClientControl {
public:
    ClientControl() = default;
    explicit ClientControl(std::string name);
    explicit ClientControl(ClientControlBlock* control_block);
    ~ClientControl();

    ClientControl(const ClientControl&) = delete;
    ClientControl& operator=(const ClientControl&) = delete;

    void bind(ClientControlBlock* control_block);
    void reset_completion() const;
    void wait_completion() const;
    void complete_request() const;

private:
    ClientControlBlock* require_control_block() const;

    std::string name_;
    SharedMemoryRegion shm_;
    ClientControlBlock* control_block_{nullptr};
};

} // namespace exam
