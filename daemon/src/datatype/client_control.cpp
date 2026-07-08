#include "datatype/client_control.hpp"

#include "ipc/process_shared_synchronizer.hpp"

#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace exam {
namespace {

class PthreadMutexGuard {
public:
    explicit PthreadMutexGuard(pthread_mutex_t* mutex, const char* lock_context)
        : mutex_(mutex) {
        ProcessSharedSynchronizer::check(pthread_mutex_lock(mutex_), lock_context);
        locked_ = true;
    }

    PthreadMutexGuard(const PthreadMutexGuard&) = delete;
    PthreadMutexGuard& operator=(const PthreadMutexGuard&) = delete;

    ~PthreadMutexGuard() {
        if (locked_) {
            pthread_mutex_unlock(mutex_);
        }
    }

private:
    pthread_mutex_t* mutex_;
    bool locked_{false};
};

} // namespace

void ClientControlBlock::initialize() {
    std::memset(this, 0, sizeof(*this));
    owner_pid = getpid();
    ProcessSharedSynchronizer::init_mutex(&mutex, true, &priority_inheritance_enabled);
    ProcessSharedSynchronizer::init_cond(&completed_cv);
    initialized = true;
}

ClientControl::ClientControl(std::string name)
    : name_(std::move(name)),
      shm_(SharedMemoryRegion::open(name_)),
      control_block_(static_cast<ClientControlBlock*>(shm_.data())) {
    if (!control_block_->initialized) {
        throw std::runtime_error("client control block is not initialized");
    }
}

ClientControl::ClientControl(ClientControlBlock* control_block) {
    bind(control_block);
}

ClientControl::~ClientControl() = default;

void ClientControl::bind(ClientControlBlock* control_block) {
    if (control_block == nullptr) {
        throw std::runtime_error("client control block is null");
    }
    if (!control_block->initialized) {
        throw std::runtime_error("client control block is not initialized");
    }
    control_block_ = control_block;
}

void ClientControl::reset_completion() const {
    ClientControlBlock* control_block = require_control_block();
    PthreadMutexGuard lock(&control_block->mutex, "pthread_mutex_lock client control");
    control_block->request_completed = false;
}

void ClientControl::wait_completion() const {
    ClientControlBlock* control_block = require_control_block();
    ProcessSharedSynchronizer::check(
        pthread_mutex_lock(&control_block->mutex),
        "pthread_mutex_lock client control");

    while (!control_block->request_completed) {
        ProcessSharedSynchronizer::check(
            pthread_cond_wait(&control_block->completed_cv, &control_block->mutex),
            "pthread_cond_wait request complete");
    }

    ProcessSharedSynchronizer::check(
        pthread_mutex_unlock(&control_block->mutex),
        "pthread_mutex_unlock client control");
}

void ClientControl::complete_request() const {
    ClientControlBlock* control_block = require_control_block();
    PthreadMutexGuard lock(&control_block->mutex, "pthread_mutex_lock client control");
    control_block->request_completed = true;
    ProcessSharedSynchronizer::check(
        pthread_cond_signal(&control_block->completed_cv),
        "pthread_cond_signal request complete");
}

ClientControlBlock* ClientControl::require_control_block() const {
    if (control_block_ == nullptr) {
        throw std::runtime_error("client control is not bound");
    }
    return control_block_;
}

} // namespace exam
