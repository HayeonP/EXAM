#include "thread_config.hpp"

#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <vector>

namespace exam {
namespace {

std::string thread_error_message(int rc) {
    return std::strerror(rc);
}

std::string cpu_list_to_string(const std::vector<int>& cpu_ids) {
    std::ostringstream out;
    for (std::size_t i = 0; i < cpu_ids.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << cpu_ids[i];
    }
    return out.str();
}

} // anonymous namespace

void ThreadConfig::set_priority(const char* name, int priority) {
    if (priority <= 0) {
        return;
    }

    sched_param param{};
    param.sched_priority = priority;
    const int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    if (rc != 0) {
        std::cerr << name << ": SCHED_FIFO priority " << priority
                  << " was not applied: " << thread_error_message(rc)
                  << " (run with CAP_SYS_NICE/root for strict wakeup ordering)\n";
    }
}

void ThreadConfig::bind_to_cpu(const char* name, int cpu_id) {
    if (cpu_id < 0) {
        return;
    }

    bind_to_cpus(name, std::vector<int>{cpu_id});
}

void ThreadConfig::bind_to_cpus(const char* name, const std::vector<int>& cpu_ids) {
    if (cpu_ids.empty()) {
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    bool has_cpu = false;
    for (int cpu_id : cpu_ids) {
        if (cpu_id >= 0) {
            CPU_SET(cpu_id, &cpuset);
            has_cpu = true;
        }
    }
    if (!has_cpu) {
        return;
    }

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (rc != 0) {
        std::cerr << name << ": CPU binding to cpus [" << cpu_list_to_string(cpu_ids)
                  << "] was not applied: " << thread_error_message(rc) << '\n';
    }
}

void ThreadConfig::apply(const char* name) const {
    bind_to_cpus(name, cpu_ids);
    set_priority(name, sched_fifo_priority);
}

} // namespace exam
