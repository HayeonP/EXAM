#pragma once

#include <vector>

namespace exam {

class ThreadConfig {
public:
    int sched_fifo_priority;
    std::vector<int> cpu_ids;

    void apply(const char* name) const;

    static void set_priority(const char* name, int priority);
    static void bind_to_cpu(const char* name, int cpu_id);
    static void bind_to_cpus(const char* name, const std::vector<int>& cpu_ids);
};

} // namespace exam
