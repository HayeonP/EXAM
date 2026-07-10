#!/bin/bash
set -euo pipefail

# Usage: VAR=value ./build.sh
# Inputs:
#   CXX: C++ compiler command.
#   ENABLE_PYTORCH: 1 builds PyTorch worker/direct binaries.
#   TORCH_PREFIX: LibTorch or Python torch root prefix.
#   TORCH_INCLUDE_DIR: explicit torch include directory.
#   TORCH_LIB_DIR: explicit torch library directory.
#   PYTORCH_CXXFLAGS_EXTRA: extra PyTorch compile flags.
#   PYTORCH_LDFLAGS_EXTRA: extra PyTorch link flags.

CXX=${CXX:-g++}
CXXFLAGS=(-std=c++17 -Wall -Wextra -pedantic -O2 -pthread -Iinclude -I/usr/local/cuda/targets/aarch64-linux/include)
TENSOR_RT_LDFLAGS=(-L/usr/local/cuda/targets/aarch64-linux/lib -lnvinfer -lnvinfer_plugin -lcudart)
ENABLE_PYTORCH=${ENABLE_PYTORCH:-0}
PYTORCH_SRCS=()
PYTORCH_CXXFLAGS=()
PYTORCH_LDFLAGS=()
OPENCV_CXXFLAGS=()
OPENCV_LDFLAGS=()

append_env_words() {
    local value=$1
    local target_name=$2

    if [[ -z "$value" ]]; then
        return
    fi

    local -n target=$target_name
    local words=()
    read -r -a words <<< "$value"
    target+=("${words[@]}")
}

configure_pytorch() {
    PYTORCH_SRCS=(src/worker/pytorch_worker.cpp)
    PYTORCH_CXXFLAGS=(-DEXAM_ENABLE_PYTORCH)

    if [[ -n "${PYTORCH_CXXFLAGS_EXTRA:-}" || -n "${PYTORCH_LDFLAGS_EXTRA:-}" ]]; then
        append_env_words "${PYTORCH_CXXFLAGS_EXTRA:-}" PYTORCH_CXXFLAGS
        append_env_words "${PYTORCH_LDFLAGS_EXTRA:-}" PYTORCH_LDFLAGS
        return
    fi

    if [[ -n "${TORCH_PREFIX:-}" ]]; then
        local torch_include_dir=${TORCH_INCLUDE_DIR:-"$TORCH_PREFIX/include"}
        local torch_lib_dir=${TORCH_LIB_DIR:-"$TORCH_PREFIX/lib"}
        PYTORCH_CXXFLAGS+=(
            -isystem
            "$torch_include_dir"
            -isystem
            "$torch_include_dir/torch/csrc/api/include"
        )
        PYTORCH_LDFLAGS+=(
            "-L$torch_lib_dir"
            "-Wl,-rpath,$torch_lib_dir"
            -ltorch
            -ltorch_cpu
            -lc10
        )
        return
    fi

    local python_torch_flags
    if python_torch_flags=$(python3 - <<'PY' 2>/dev/null
import os
import torch

torch_root = os.path.dirname(torch.__file__)
include_dir = os.path.join(torch_root, "include")
api_include_dir = os.path.join(include_dir, "torch", "csrc", "api", "include")
lib_dir = os.path.join(torch_root, "lib")

print("CXX -D_GLIBCXX_USE_CXX11_ABI={}".format(
    int(torch._C._GLIBCXX_USE_CXX11_ABI)))
print("CXX -isystem")
print("CXX {}".format(include_dir))
print("CXX -isystem")
print("CXX {}".format(api_include_dir))
print("LD -L{}".format(lib_dir))
print("LD -Wl,-rpath,{}".format(lib_dir))
print("LD -ltorch")
print("LD -ltorch_cpu")
print("LD -lc10")
PY
    ); then
        while IFS= read -r line; do
            case "$line" in
                CXX\ *) PYTORCH_CXXFLAGS+=("${line#CXX }") ;;
                LD\ *) PYTORCH_LDFLAGS+=("${line#LD }") ;;
            esac
        done <<< "$python_torch_flags"
        return
    fi

    echo "ENABLE_PYTORCH=1 requires TORCH_PREFIX, python3 torch, or PYTORCH_*_EXTRA flags" >&2
    exit 1
}

configure_opencv() {
    local include_dir=""
    for candidate in /usr/include/opencv4 /usr/local/include/opencv4; do
        if [[ -f "$candidate/opencv2/imgcodecs.hpp" ]]; then
            include_dir=$candidate
            break
        fi
    done

    if [[ -n "$include_dir" ]]; then
        local imgcodecs_lib
        local imgproc_lib
        local core_lib
        imgcodecs_lib=$(find /usr/lib /usr/local/lib -name 'libopencv_imgcodecs.so*' -print 2>/dev/null | sort -V | tail -n 1)
        imgproc_lib=$(find /usr/lib /usr/local/lib -name 'libopencv_imgproc.so*' -print 2>/dev/null | sort -V | tail -n 1)
        core_lib=$(find /usr/lib /usr/local/lib -name 'libopencv_core.so*' -print 2>/dev/null | sort -V | tail -n 1)
        if [[ -n "$imgcodecs_lib" && -n "$imgproc_lib" && -n "$core_lib" ]]; then
            OPENCV_CXXFLAGS=(-I "$include_dir")
            OPENCV_LDFLAGS=("$imgcodecs_lib" "$imgproc_lib" "$core_lib")
            return
        fi
    fi

    if ! pkg-config --exists opencv4; then
        echo "OpenCV 4 is required to build resnet18_process image input support" >&2
        exit 1
    fi

    local opencv_cflags
    local opencv_ldflags
    opencv_cflags=$(pkg-config --cflags opencv4)
    opencv_ldflags=$(pkg-config --libs opencv4)
    append_env_words "$opencv_cflags" OPENCV_CXXFLAGS
    append_env_words "$opencv_ldflags" OPENCV_LDFLAGS
}

if [[ "$ENABLE_PYTORCH" == "1" ]]; then
    configure_pytorch
fi
configure_opencv

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
    src/config/cpu_affinity_config.cpp
    src/datatype/subgraph.cpp
    src/exam_client.cpp
)

DAEMON_SRCS=(
    apps/exam_daemon_example.cpp
    src/exam_daemon.cpp
    src/worker/worker.cpp
    src/worker/mock_worker.cpp
    src/worker/tensor_rt_gpu_worker.cpp
    "${PYTORCH_SRCS[@]}"
)

"$CXX" "${CXXFLAGS[@]}" "${PYTORCH_CXXFLAGS[@]}" "${DAEMON_SRCS[@]}" "${COMMON_SRCS[@]}" -o exam_daemon -lrt "${TENSOR_RT_LDFLAGS[@]}" "${PYTORCH_LDFLAGS[@]}"
"$CXX" "${CXXFLAGS[@]}" "${OPENCV_CXXFLAGS[@]}" apps/resnet18_process.cpp "${COMMON_SRCS[@]}" -o resnet18_process -lrt "${OPENCV_LDFLAGS[@]}"
if [[ "$ENABLE_PYTORCH" == "1" ]]; then
    "$CXX" "${CXXFLAGS[@]}" "${PYTORCH_CXXFLAGS[@]}" \
        apps/resnet18_pytorch_direct.cpp \
        src/thread_config.cpp \
        src/config/cpu_affinity_config.cpp \
        -o resnet18_pytorch_direct -lrt "${PYTORCH_LDFLAGS[@]}"
fi
"$CXX" "${CXXFLAGS[@]}" apps/shared_memory_region_sample.cpp src/ipc/shared_memory_region.cpp -o shared_memory_region_sample -lrt
