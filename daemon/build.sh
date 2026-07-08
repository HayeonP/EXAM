#!/bin/bash
set -euo pipefail

CXX=${CXX:-g++}
CXXFLAGS=(-std=c++17 -Wall -Wextra -pedantic -O2 -pthread -Iinclude -I/usr/local/cuda/targets/aarch64-linux/include)
TENSOR_RT_LDFLAGS=(-L/usr/local/cuda/targets/aarch64-linux/lib -lnvinfer -lnvinfer_plugin -lcudart)

COMMON_SRCS=(
    src/ipc/shared_memory_region.cpp
    src/ipc/process_shared_synchronizer.cpp
    src/datatype/request.cpp
    src/datatype/event.cpp
    src/datatype/event_queue.cpp
    src/datatype/payload.cpp
    src/datatype/channel.cpp
    src/datatype/client_control.cpp
    src/datatype/subgraph_config.cpp
    src/thread_config.cpp
    src/datatype/subgraph.cpp
    src/exam_client.cpp
)

"$CXX" "${CXXFLAGS[@]}" apps/exam_daemon_main.cpp src/exam_daemon.cpp src/worker/worker.cpp src/worker/mock_worker.cpp src/worker/tensor_rt_gpu_worker.cpp "${COMMON_SRCS[@]}" -o exam_daemon -lrt "${TENSOR_RT_LDFLAGS[@]}"
"$CXX" "${CXXFLAGS[@]}" apps/process.cpp "${COMMON_SRCS[@]}" -o process -lrt
"$CXX" "${CXXFLAGS[@]}" apps/resnet18_process.cpp "${COMMON_SRCS[@]}" -o resnet18_process -lrt
"$CXX" "${CXXFLAGS[@]}" apps/shared_memory_region_sample.cpp src/ipc/shared_memory_region.cpp -o shared_memory_region_sample -lrt
"$CXX" "${CXXFLAGS[@]}" apps/build_sample_model.cpp -o build_sample_model "${TENSOR_RT_LDFLAGS[@]}"
