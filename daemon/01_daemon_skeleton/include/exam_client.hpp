#pragma once

#include "datatype/channel.hpp"
#include "datatype/client_control.hpp"
#include "datatype/event_queue.hpp"
#include "ipc/shared_memory_region.hpp"

#include <string>

namespace exam {

class ExamClient {
public:
    ExamClient(
        std::string sg_sequence_file_path = "mock_sg_sequence.fake",
        std::size_t input_size = DEFAULT_INPUT_SIZE,    // bytes
        std::size_t output_size = DEFAULT_OUTPUT_SIZE   // bytes
        );
    ~ExamClient();

    bool request(const Payload& input);
    Payload wait();

    const std::string& channel_name() const;
    std::uint64_t accepted_requests() const;
    std::uint64_t completed_requests() const;
    std::uint64_t dropped_requests() const;

private:
    void register_channel();
    void release_active_request();

    SharedMemoryRegion event_queue_shm_;
    EventQueue* event_queue_{nullptr};
    std::string channel_name_;
    SharedMemoryRegion channel_shm_;
    ChannelControlBlock* control_block_{nullptr};
    std::string client_control_name_;
    SharedMemoryRegion client_control_shm_;
    ClientControl client_control_;
    std::string sg_sequence_file_path_;
    bool request_in_flight_{false};
    std::uint64_t next_request_id_{1};
    std::uint64_t accepted_requests_{0};
    std::uint64_t completed_requests_{0};
    std::uint64_t dropped_requests_{0};
};

} // namespace exam
