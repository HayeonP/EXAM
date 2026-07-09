#pragma once

#include <string>
#include <vector>

namespace exam {

class CpuAffinityConfig {
public:
    std::string path;
    std::vector<int> daemon_cpu_ids;
    std::vector<int> tensor_rt_gpu_worker_cpu_ids;
    std::vector<int> pytorch_worker_cpu_ids;

    static CpuAffinityConfig load_default();
    static CpuAffinityConfig load_from_file(const std::string& path);

    std::vector<int> cpus_for_pytorch_direct() const;
    std::string summary() const;
};

std::string cpu_ids_to_string(const std::vector<int>& cpu_ids);

} // namespace exam
