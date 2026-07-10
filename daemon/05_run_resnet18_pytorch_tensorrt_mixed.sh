#!/bin/bash
set -euo pipefail

# Usage: VAR=value ./05_run_resnet18_pytorch_tensorrt_mixed.sh
# Inputs:
#   ITERATIONS: measured execution count.
#   WARMUP: warmup execution count.
#   PYTORCH_NUM_THREADS: PyTorch intra-op thread count.
#   PYTORCH_INTEROP_THREADS: PyTorch inter-op thread count.
#   SCHED_FIFO: 1 enables sudo/SCHED_FIFO, 0 disables it.
#   PYTORCH_SG_SEQUENCE_PATH: PyTorch SG sequence JSON path.
#   TENSOR_RT_SG_SEQUENCE_PATH: TensorRT SG sequence JSON path.
#   MIXED_SG_SEQUENCE_PATH: mixed-worker SG sequence JSON path.
#   IMAGE_DIR: input image directory.
#   LABELS_PATH: ImageNet label text file path.
#   PYTORCH_DIR: TorchScript module directory.
#   CONDA_ENV_PREFIX: conda env prefix for PyTorch checks.
#   LOG_DIR: scenario log directory.
#   VERBOSE: 1 prints captured logs.
#   EXAM_CONFIG_PATH: config YAML path and daemon config.
#   LD_LIBRARY_PATH: runtime library search path.

cd "$(dirname "$0")"

source ./scripts/sudo_helpers.sh

default_config_path() {
    if [[ -n "${EXAM_CONFIG_PATH:-}" ]]; then
        echo "$EXAM_CONFIG_PATH"
        return
    fi

    if [[ -f config/examl.yaml ]]; then
        echo "config/examl.yaml"
        return
    fi

    if [[ -f config/exam.yaml ]]; then
        echo "config/exam.yaml"
        return
    fi
}

pytorch_worker_cpu_count() {
    local config_path
    config_path=$(default_config_path)
    if [[ -z "$config_path" || ! -f "$config_path" ]]; then
        echo 0
        return
    fi

    sed -n 's/^[[:space:]]*pytorch_worker:[[:space:]]*\[\(.*\)\][[:space:]]*$/\1/p' "$config_path" \
        | head -n 1 \
        | tr ',' '\n' \
        | awk '
            {
                gsub(/^[ \t]+|[ \t]+$/, "")
                if ($0 != "") {
                    count += 1
                }
            }
            END {
                print count + 0
            }
        '
}

default_pytorch_num_threads() {
    local cpu_count
    cpu_count=$(pytorch_worker_cpu_count)
    if [[ "$cpu_count" =~ ^[0-9]+$ && "$cpu_count" -gt 0 ]]; then
        echo "$cpu_count"
        return
    fi

    nproc
}

ITERATIONS=${ITERATIONS:-10}
WARMUP=${WARMUP:-5}
TOTAL_ITERATIONS=$((WARMUP + ITERATIONS))
PYTORCH_NUM_THREADS=${PYTORCH_NUM_THREADS:-$(default_pytorch_num_threads)}
PYTORCH_INTEROP_THREADS=${PYTORCH_INTEROP_THREADS:-1}
SCHED_FIFO=${SCHED_FIFO:-1}
PYTORCH_SG_SEQUENCE_PATH=${PYTORCH_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_pytorch.json}
TENSOR_RT_SG_SEQUENCE_PATH=${TENSOR_RT_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_tensor_rt_gpu.json}
MIXED_SG_SEQUENCE_PATH=${MIXED_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_mixed.json}
IMAGE_DIR=${IMAGE_DIR:-artifacts/image}
LABELS_PATH=${LABELS_PATH:-artifacts/image/imagenet_classes.txt}
PYTORCH_DIR=${PYTORCH_DIR:-artifacts/resnet18/pytorch}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/home/rubis/workspace/miniconda3/envs/exam}
LOG_DIR=${LOG_DIR:-/tmp/exam_resnet18_pytorch_tensorrt_mixed_logs}
VERBOSE=${VERBOSE:-0}

daemon_pid=""
daemon_wrapper_pid=""
daemon_pid_file=""
terminal_state=""
disable_sched_fifo=1
if [[ "$SCHED_FIFO" == "1" ]]; then
    exam_require_passwordless_sudo "[05]"
    disable_sched_fifo=0
fi
if [[ -t 0 ]]; then
    terminal_state=$(stty -g 2>/dev/null || true)
fi

wait_for_pid_exit() {
    local pid=$1
    for _ in $(seq 1 30); do
        if ! exam_pid_exists "$pid"; then
            return 0
        fi
        if ps -o stat= -p "$pid" 2>/dev/null | grep -q 'Z'; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

terminate_pid() {
    local pid=${1:-}
    local name=${2:-process}
    if [[ -z "$pid" ]]; then
        return
    fi
    if ! exam_pid_exists "$pid"; then
        wait "$pid" 2>/dev/null || true
        return
    fi

    exam_kill TERM "$pid"
    if ! wait_for_pid_exit "$pid"; then
        echo "[05] $name pid=$pid did not stop; sending SIGKILL" >&2
        exam_kill KILL "$pid"
        wait "$pid" 2>/dev/null || true
        return
    fi
    wait "$pid" 2>/dev/null || true
}

cleanup_shm() {
    exam_rm -f /dev/shm/exam_event_queue \
          /dev/shm/exam_daemon_global_shm \
          /dev/shm/exam_daemon_shm \
          /dev/shm/exam_request_queue \
          /dev/shm/exam_channel_* \
          /dev/shm/exam_client_control_*
}

stop_daemon() {
    terminate_pid "$daemon_pid" "exam_daemon"
    daemon_pid=""
    if [[ -n "$daemon_wrapper_pid" ]]; then
        wait "$daemon_wrapper_pid" 2>/dev/null || true
        daemon_wrapper_pid=""
    fi
    if [[ -n "$daemon_pid_file" ]]; then
        rm -f "$daemon_pid_file" 2>/dev/null || true
        daemon_pid_file=""
    fi
}

cleanup() {
    stop_daemon
    cleanup_shm
    if [[ -n "$terminal_state" ]]; then
        stty "$terminal_state" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ensure_exam_env() {
    if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
        echo "missing conda env python: $CONDA_ENV_PREFIX/bin/python" >&2
        exit 1
    fi
}

ensure_image_artifacts() {
    if [[ ! -d "$IMAGE_DIR" ]]; then
        echo "missing image dir: $IMAGE_DIR" >&2
        exit 1
    fi
    local image_count
    image_count=$(
        find "$IMAGE_DIR" -maxdepth 1 -type f \
            \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
            | wc -l
    )
    if [[ "$image_count" -eq 0 ]]; then
        echo "missing .png/.jpg/.jpeg images in: $IMAGE_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$LABELS_PATH" ]]; then
        echo "missing labels file: $LABELS_PATH" >&2
        exit 1
    fi
}

ensure_pytorch_artifacts() {
    local missing=0
    for artifact in sg1.pt sg2.pt sg3.pt; do
        if [[ ! -f "$PYTORCH_DIR/$artifact" ]]; then
            echo "missing PyTorch TorchScript artifact: $PYTORCH_DIR/$artifact" >&2
            missing=1
        fi
    done
    if [[ "$missing" -ne 0 ]]; then
        exit 1
    fi
}

has_pytorch_enabled_daemon() {
    [[ -x ./exam_daemon ]] && ldd ./exam_daemon 2>/dev/null | grep -q 'libtorch_cpu'
}

has_image_enabled_resnet18_process() {
    [[ -x ./resnet18_process ]] \
        && ldd ./resnet18_process 2>/dev/null | grep -q 'opencv_imgcodecs'
}

ensure_built() {
    if has_image_enabled_resnet18_process && has_pytorch_enabled_daemon; then
        echo "[05] build skipped: PyTorch/OpenCV-enabled binaries found"
        return
    fi

    echo "[05] building PyTorch/OpenCV-enabled binaries..."
    PATH="$CONDA_ENV_PREFIX/bin:$PATH" \
        PYTHONNOUSERSITE=1 \
        ENABLE_PYTORCH=1 \
        ./build.sh
}

wait_for_daemon_ready() {
    local daemon_log=$1
    for _ in $(seq 1 200); do
        if grep -q 'daemon: ready' "$daemon_log" 2>/dev/null; then
            return 0
        fi
        if ! exam_pid_exists "$daemon_pid"; then
            echo "daemon exited before becoming ready" >&2
            return 1
        fi
        sleep 0.05
    done

    echo "timed out waiting for daemon ready" >&2
    return 1
}

start_daemon() {
    local daemon_log=$1

    cleanup_shm
    daemon_pid=""
    daemon_wrapper_pid=""
    daemon_pid_file=""

    if [[ ${#EXAM_SUDO[@]} -eq 0 ]]; then
        env \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            EXAM_PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            EXAM_PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
            PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
            OMP_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            MKL_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            ./exam_daemon mock-fifo > "$daemon_log" 2>&1 &
        daemon_pid=$!
    else
        daemon_pid_file=$(mktemp /tmp/exam_daemon_pid.XXXXXX)
        rm -f "$daemon_pid_file"
        "${EXAM_SUDO[@]}" env \
            EXAM_DAEMON_PID_FILE="$daemon_pid_file" \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            EXAM_PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            EXAM_PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
            PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
            OMP_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            MKL_NUM_THREADS="$PYTORCH_NUM_THREADS" \
            bash -c 'echo $$ > "$EXAM_DAEMON_PID_FILE"; exec ./exam_daemon mock-fifo' \
            > "$daemon_log" 2>&1 &
        daemon_wrapper_pid=$!
        for _ in $(seq 1 100); do
            if [[ -s "$daemon_pid_file" ]]; then
                daemon_pid=$(cat "$daemon_pid_file")
                break
            fi
            sleep 0.01
        done
        if [[ -z "$daemon_pid" ]]; then
            echo "timed out waiting for daemon pid file" >&2
            return 1
        fi
    fi

    wait_for_daemon_ready "$daemon_log"
}

run_scenario() {
    local name=$1
    local sg_sequence_path=$2
    local process_log="$LOG_DIR/$name.log"
    local daemon_log="$LOG_DIR/$name.daemon.log"

    echo "[05] running $name scenario..."
    start_daemon "$daemon_log"

    set +e
    "${EXAM_SUDO[@]}" env \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        ./resnet18_process \
            1 \
            "$sg_sequence_path" \
            "" \
            "" \
            "$TOTAL_ITERATIONS" \
            "$IMAGE_DIR" \
            "$LABELS_PATH" \
            > "$process_log" 2>&1
    local status=$?
    set -e

    stop_daemon
    cleanup_shm

    if [[ "$status" -ne 0 ]]; then
        echo "[05] $name scenario failed with status=$status" >&2
        sed -n '1,220p' "$process_log" >&2
        sed -n '1,260p' "$daemon_log" >&2
        exit "$status"
    fi
}

print_summary() {
    LOG_DIR="$LOG_DIR" \
    ITERATIONS="$ITERATIONS" \
    WARMUP="$WARMUP" \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import math
import os
import re
import statistics
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
iterations = int(os.environ["ITERATIONS"])
warmup = int(os.environ["WARMUP"])
scenarios = ("pytorch", "tensor_rt", "mixed")
pattern = re.compile(
    r'^RESULT sample=(\d+) image="([^"]+)" .* '
    r'output_hash=([0-9a-f]+) .* latency_us=(\d+)'
)
classification_pattern = re.compile(
    r'^RESULT sample=(\d+) image="([^"]+)" .* latency_us=(\d+) '
    r'top1_index=(\d+) top1_label="([^"]+)" '
    r'top1_probability=([0-9.]+)'
)
sg_pattern = re.compile(
    r"daemon: (?:SG complete|request complete).* sg=(\d+) sg_latency_us=(\d+)"
)

def read_results(name):
    path = log_dir / f"{name}.log"
    results = []
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                results.append(
                    {
                        "sample": int(match.group(1)),
                        "image": match.group(2),
                        "hash": match.group(3),
                        "latency": int(match.group(4)),
                    }
                )
    if len(results) < warmup + iterations:
        raise SystemExit(
            f"{name}: got {len(results)} results, expected {warmup + iterations}"
        )
    return results[-iterations:]

def read_classifications(name):
    path = log_dir / f"{name}.log"
    rows = []
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            match = classification_pattern.search(line)
            if match:
                rows.append(
                    {
                        "sample": int(match.group(1)),
                        "image": match.group(2),
                        "latency": int(match.group(3)),
                        "index": int(match.group(4)),
                        "label": match.group(5),
                        "probability": float(match.group(6)),
                    }
                )
    return rows[-iterations:]

def percentile(values, ratio):
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * ratio) - 1)
    return sorted_values[index]

def read_sg_latencies(name):
    path = log_dir / f"{name}.daemon.log"
    values = {0: [], 1: [], 2: []}
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            match = sg_pattern.search(line)
            if not match:
                continue
            sg_id = int(match.group(1))
            if sg_id in values:
                values[sg_id].append(int(match.group(2)))

    measured = {}
    for sg_id in range(3):
        measured[sg_id] = values[sg_id][-iterations:]
        if len(measured[sg_id]) != iterations:
            raise SystemExit(
                f"{name}: got {len(measured[sg_id])} sg{sg_id} latencies, "
                f"expected {iterations}"
            )
    return measured

def stat(values, kind):
    if kind == "avg":
        return int(statistics.mean(values))
    if kind == "p50":
        return int(statistics.median(values))
    if kind == "p99":
        return percentile(values, 0.99)
    if kind == "max":
        return max(values)
    raise ValueError(kind)

data = {name: read_results(name) for name in scenarios}
sg_latencies = {name: read_sg_latencies(name) for name in scenarios}
reference_inputs = [(row["sample"], row["image"]) for row in data["pytorch"]]
for name in scenarios[1:]:
    inputs = [(row["sample"], row["image"]) for row in data[name]]
    if inputs != reference_inputs:
        raise SystemExit(f"{name}: measured images differ from pytorch")

print(
    "RESNET18_PYTORCH_TENSORRT_MIXED "
    f"warmup={warmup} iterations={iterations}"
)

print("SUMMARY_US")
print(
    f"{'scenario':<10} {'stat':<4} {'overall':>10} "
    f"{'sg0':>10} {'sg1':>10} {'sg2':>10}"
)
for name in scenarios:
    columns = {
        "overall": [row["latency"] for row in data[name]],
        "sg0": sg_latencies[name][0],
        "sg1": sg_latencies[name][1],
        "sg2": sg_latencies[name][2],
    }
    for kind in ("avg", "p50", "p99", "max"):
        print(
            f"{name:<10} {kind:<4} "
            + " ".join(f"{stat(columns[column], kind):>10}" for column in columns)
        )

classifications = {name: read_classifications(name) for name in scenarios}
if any(classifications.values()):
    print("")
    print("CLASSIFICATION_RESULTS")
    print(
        f"{'scenario':<10} {'sample':>6} {'image':<12} "
        f"{'top1':>5} {'prob':>8} label"
    )
    for name in scenarios:
        for row in classifications[name]:
            print(
                f"{name:<10} {row['sample']:>6} {row['image']:<12} "
                f"{row['index']:>5} {row['probability']:>8.5f} {row['label']}"
            )
PY
}

print_verbose_logs() {
    for name in pytorch tensor_rt mixed; do
        echo "--- $name process log: $LOG_DIR/$name.log ---"
        sed -n '1,220p' "$LOG_DIR/$name.log"
        echo "--- $name daemon log: $LOG_DIR/$name.daemon.log ---"
        sed -n '1,300p' "$LOG_DIR/$name.daemon.log"
    done
}

ensure_exam_env
ensure_image_artifacts
ensure_pytorch_artifacts
ensure_built

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"
cleanup_shm

run_scenario pytorch "$PYTORCH_SG_SEQUENCE_PATH"
run_scenario tensor_rt "$TENSOR_RT_SG_SEQUENCE_PATH"
run_scenario mixed "$MIXED_SG_SEQUENCE_PATH"

echo "SCHED_FIFO=$SCHED_FIFO"
echo "PYTORCH_THREADS worker=[intra=$PYTORCH_NUM_THREADS interop=$PYTORCH_INTEROP_THREADS]"
echo "IMAGE_DIR=$IMAGE_DIR"
mixed_migration_count=$(grep -c 'migration_source_worker=' "$LOG_DIR/mixed.daemon.log" || true)
echo "MIXED_MIGRATION_LAUNCHES=$mixed_migration_count"
print_summary

if [[ "$VERBOSE" == "1" ]]; then
    print_verbose_logs
fi
