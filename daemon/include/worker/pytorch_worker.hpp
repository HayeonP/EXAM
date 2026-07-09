#pragma once

#include "worker.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace exam {

class PyTorchWorker final : public Worker {
public:
    explicit PyTorchWorker(
        std::uint32_t worker_id,
        ThreadConfig thread_config = ThreadConfig{0, {}});
    ~PyTorchWorker() override;

protected:
    void init() override;
    void set_input(const Payload& input) override;
    void execute() override;
    Payload get_output() override;
    void terminate() override;
    void release_sg_cache(const Subgraph& sg) override;

private:
    class CachedModule;
    class PyTorchExecutor;

    CachedModule& module_for(const SubgraphConfig& config);
    std::string current_request_key() const;

    std::mutex cache_mutex_;
    std::unordered_map<std::string, std::unique_ptr<CachedModule>> module_cache_;
    std::unordered_map<std::string, std::unique_ptr<PyTorchExecutor>>
        last_executor_by_request_;
    Payload input_;
    Payload output_;
};

} // namespace exam
