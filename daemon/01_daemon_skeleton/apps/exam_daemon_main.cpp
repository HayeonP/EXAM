#include "exam_daemon.hpp"
#include "thread_config.hpp"
#include "worker/mock_worker.hpp"

#include <exception>
#include <iostream>

namespace {

constexpr int DAEMON_PRIORITY = 80;
constexpr int WORKER_PRIORITY = 60;
constexpr std::uint32_t MOCK_WORKER_ID = 0;

} // namespace

int main() {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;

        const exam::ThreadConfig daemon_thread{DAEMON_PRIORITY, {}};
        const exam::ThreadConfig worker_thread{WORKER_PRIORITY, {}};

        daemon_thread.apply("daemon");
        exam::MockWorker worker(MOCK_WORKER_ID, worker_thread);
        exam::ExamDaemon daemon(worker, exam::ExamDaemon::SchedulingPolicy::MOCK_FIFO);
        return daemon.run_loop();
    } catch (const std::exception& e) {
        std::cerr << "daemon: " << e.what() << '\n';
        return 1;
    }
}
