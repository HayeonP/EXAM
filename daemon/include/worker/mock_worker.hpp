#pragma once

#include "worker.hpp"

namespace exam {

class MockWorker final : public Worker {
public:
    explicit MockWorker(
        std::uint32_t worker_id,
        ThreadConfig thread_config = ThreadConfig{0, {}});

protected:
    void init() override;
    void set_input(const Payload& input) override;
    void execute() override;
    Payload get_output() override;
    void terminate() override;

private:
    Payload input_;
    Payload output_;
};

} // namespace exam
