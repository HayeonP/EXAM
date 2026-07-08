#pragma once

#include "datatype/payload.hpp"
#include "ipc/shared_memory_region.hpp"

#include <cstddef>
#include <cstdint>
#include <pthread.h>
#include <string>
#include <sys/types.h>

namespace exam {

constexpr std::size_t DEFAULT_INPUT_SIZE = 256;
constexpr std::size_t DEFAULT_OUTPUT_SIZE = 256;

class ChannelControlBlock {
public:
    bool initialized;
    bool priority_inheritance_enabled;
    pid_t owner_pid;
    std::uint32_t input_size;
    std::uint32_t output_size;
    std::uint32_t input_offset;
    std::uint32_t output_offset;
    pthread_mutex_t mutex;

    void initialize(std::size_t input_size, std::size_t output_size);
    char* input_buffer_ptr();
    char* output_buffer_ptr();
    const char* input_buffer_ptr() const;
    const char* output_buffer_ptr() const;
};

class Channel {
public:
    explicit Channel(std::string name);
    ~Channel();

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    Payload read_input() const;
    void write_output(const Payload& output) const;
    const std::string& name() const;

private:
    std::string name_;
    SharedMemoryRegion shm_;
    ChannelControlBlock* control_block_{nullptr};
};

} // namespace exam
