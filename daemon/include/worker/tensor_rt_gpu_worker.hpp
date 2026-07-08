#pragma once

#include "worker.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace exam {

class TensorRtGpuWorker final : public Worker {
public:
    explicit TensorRtGpuWorker(
        std::uint32_t worker_id,
        int device_id,
        ThreadConfig thread_config = ThreadConfig{0, {}});
    ~TensorRtGpuWorker() override;

protected:
    void init() override;
    void set_input(const Payload& input) override;
    void execute() override;
    Payload get_output() override;
    void terminate() override;
    void release_sg_cache(const Subgraph& sg) override;

private:
    class CachedEngine;
    class TensorRtExecutor;

    CachedEngine& engine_for(const SubgraphConfig& config);
    std::string current_request_key() const;

    int device_id_{0};
    cudaStream_t stream_{nullptr};
    std::mutex cache_mutex_;
    std::unordered_map<std::string, std::unique_ptr<CachedEngine>> engine_cache_;
    std::unordered_map<std::string, std::unique_ptr<TensorRtExecutor>>
        last_executor_by_request_;
    Payload input_;
    Payload output_;
};

} // namespace exam
