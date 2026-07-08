#include "worker/mock_worker.hpp"

#include <sstream>

namespace exam {
namespace {

constexpr const char* MOCK_WORKER_TYPE = "mock";

Payload print_mock_sg(
    std::uint32_t worker_id,
    const Subgraph& sg,
    const Payload&) {
    const SubgraphConfig& config = sg.config_for(MOCK_WORKER_TYPE);
    std::ostringstream output;
    output << "mock-worker " << worker_id
           << " ran " << sg.label()
           << " path=" << config.path();
    return Payload::from_text(output.str());
}

} // anonymous namespace

MockWorker::MockWorker(std::uint32_t worker_id, ThreadConfig thread_config)
    : Worker(worker_id, MOCK_WORKER_TYPE, "mock-worker", thread_config) {}

void MockWorker::init() {}

void MockWorker::set_input(const Payload& input) {
    input_ = input;
}

void MockWorker::execute() {
    output_ = print_mock_sg(id(), current_sg(), input_);
    input_ = output_;
}

Payload MockWorker::get_output() {
    return output_;
}

void MockWorker::terminate() {}

} // namespace exam
