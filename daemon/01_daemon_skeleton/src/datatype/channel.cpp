#include "datatype/channel.hpp"

#include "ipc/process_shared_synchronizer.hpp"

#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace exam {
namespace {

class PthreadMutexGuard {
public:
    PthreadMutexGuard(pthread_mutex_t* mutex, const char* lock_context)
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

void ChannelControlBlock::initialize(std::size_t input_size_value, std::size_t output_size_value) {
    std::memset(this, 0, sizeof(*this));
    owner_pid = getpid();
    input_size = static_cast<std::uint32_t>(input_size_value);
    output_size = static_cast<std::uint32_t>(output_size_value);
    input_offset = static_cast<std::uint32_t>(sizeof(ChannelControlBlock));
    output_offset = static_cast<std::uint32_t>(sizeof(ChannelControlBlock) + input_size_value);
    ProcessSharedSynchronizer::init_mutex(&mutex, true, &priority_inheritance_enabled);
    initialized = true;
}

char* ChannelControlBlock::input_buffer_ptr() {
    return reinterpret_cast<char*>(this) + input_offset;
}

char* ChannelControlBlock::output_buffer_ptr() {
    return reinterpret_cast<char*>(this) + output_offset;
}

const char* ChannelControlBlock::input_buffer_ptr() const {
    return reinterpret_cast<const char*>(this) + input_offset;
}

const char* ChannelControlBlock::output_buffer_ptr() const {
    return reinterpret_cast<const char*>(this) + output_offset;
}

Channel::Channel(std::string name)
    : name_(std::move(name)),
      shm_(SharedMemoryRegion::open(name_)),
      control_block_(static_cast<ChannelControlBlock*>(shm_.data())) {
    if (!control_block_->initialized) {
        throw std::runtime_error("channel control block is not initialized");
    }
}

Channel::~Channel() = default;

Payload Channel::read_input() const {
    PthreadMutexGuard lock(
        &control_block_->mutex,
        "pthread_mutex_lock channel");

    Payload input{};
    input.bytes.assign(
        control_block_->input_buffer_ptr(),
        control_block_->input_buffer_ptr() + control_block_->input_size);
    return input;
}

void Channel::write_output(const Payload& output) const {
    PthreadMutexGuard lock(
        &control_block_->mutex,
        "pthread_mutex_lock channel");

    if (output.bytes.size() > control_block_->output_size) {
        throw std::runtime_error("output payload is larger than channel output size");
    }

    std::memset(control_block_->output_buffer_ptr(), 0, control_block_->output_size);
    std::memcpy(control_block_->output_buffer_ptr(), output.bytes.data(), output.bytes.size());
}

const std::string& Channel::name() const {
    return name_;
}

} // namespace exam
