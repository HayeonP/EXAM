#include "config/cpu_affinity_config.hpp"
#include "exam_daemon.hpp"
#include "thread_config.hpp"
#include "worker/mock_worker.hpp"
#include "worker/tensor_rt_gpu_worker.hpp"

#ifdef EXAM_ENABLE_PYTORCH
#include "worker/pytorch_worker.hpp"
#endif

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int DAEMON_PRIORITY = 80;
constexpr int WORKER_PRIORITY = 60;
constexpr std::uint32_t MOCK_WORKER_ID = 0;
constexpr std::uint32_t TENSOR_RT_GPU_WORKER_ID = 1;
#ifdef EXAM_ENABLE_PYTORCH
constexpr std::uint32_t PYTORCH_WORKER_ID = 2;
#endif
constexpr int TENSOR_RT_GPU_DEVICE_ID = 0;

exam::ExamDaemon::SchedulingPolicy parse_scheduling_policy(int argc, char** argv) {
    if (argc < 2) {
        return exam::ExamDaemon::SchedulingPolicy::MOCK_FIFO;
    }

    const std::string policy = argv[1];
    if (policy == "mock-fifo" || policy == "MOCK_FIFO") {
        return exam::ExamDaemon::SchedulingPolicy::MOCK_FIFO;
    }
    if (policy == "mock-interleaving" || policy == "MOCK_INTERLEAVING") {
        return exam::ExamDaemon::SchedulingPolicy::MOCK_INTERLEAVING;
    }

    throw std::runtime_error("unknown scheduling policy: " + policy);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;

        const exam::CpuAffinityConfig cpu_affinity =
            exam::CpuAffinityConfig::load_default();
        std::cout << "daemon: cpu affinity " << cpu_affinity.summary() << '\n';

        const exam::ThreadConfig daemon_thread{
            DAEMON_PRIORITY,
            cpu_affinity.daemon_cpu_ids};
        const exam::ThreadConfig tensor_rt_gpu_worker_thread{
            WORKER_PRIORITY,
            cpu_affinity.tensor_rt_gpu_worker_cpu_ids};
        const exam::ThreadConfig pytorch_worker_thread{
            WORKER_PRIORITY,
            cpu_affinity.pytorch_worker_cpu_ids};

        daemon_thread.apply("daemon");
        exam::TensorRtGpuWorker tensor_rt_gpu_worker(
            TENSOR_RT_GPU_WORKER_ID,
            TENSOR_RT_GPU_DEVICE_ID,
            tensor_rt_gpu_worker_thread);
#ifdef EXAM_ENABLE_PYTORCH
        exam::PyTorchWorker pytorch_worker(PYTORCH_WORKER_ID, pytorch_worker_thread);
#endif
        exam::MockWorker mock_worker(MOCK_WORKER_ID, tensor_rt_gpu_worker_thread);
        std::vector<exam::Worker*> workers{&tensor_rt_gpu_worker};
#ifdef EXAM_ENABLE_PYTORCH
        workers.push_back(&pytorch_worker);
#endif
        workers.push_back(&mock_worker);
        exam::ExamDaemon daemon(workers, parse_scheduling_policy(argc, argv));
        return daemon.run_loop();
    } catch (const std::exception& e) {
        std::cerr << "daemon: " << e.what() << '\n';
        return 1;
    }
}
