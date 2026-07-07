#include "worker/mock_worker.hpp"

#include <chrono>
#include <sstream>
#include <thread>

namespace exam {
namespace {

Payload run_mock_work_sequence(std::uint32_t worker_id, const Payload& input) {
    const std::string input_text = input.to_string();

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const std::string stage1 = "parsed(" + input_text + ")";

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const std::string stage2 = "scheduled(" + stage1 + ")";

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    std::ostringstream output;
    output << "worker " << worker_id << " output: " << stage2;
    return Payload::from_text(output.str());
}

} // anonymous namespace

MockWorker::MockWorker(std::uint32_t worker_id, ThreadConfig thread_config)
    : Worker(worker_id, "mock-worker", thread_config) {}

void MockWorker::init() {}

void MockWorker::set_input(const Payload& input) {
    input_ = input;
}

void MockWorker::execute() {
    output_ = run_mock_work_sequence(id(), input_);
}

Payload MockWorker::get_output() {
    return output_;
}

void MockWorker::terminate() {}

} // namespace exam
