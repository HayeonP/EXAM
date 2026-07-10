#!/bin/bash
set -euo pipefail

# Usage: VAR=value ./03_run_resnet18_pytorch_tensorrt_interleave_test.sh
# Inputs:
#   ITERATIONS: number of executions per process.
#   EXPECTED_SG_PER_REQUEST: expected SG count per request.
#   PROCESS_TIMEOUT: timeout for each process.
#   PYTORCH_INPUT_ID: PyTorch process input id.
#   TENSOR_RT_INPUT_ID: TensorRT process input id.
#   PYTORCH_SG_SEQUENCE_PATH: PyTorch SG sequence JSON path.
#   TENSOR_RT_SG_SEQUENCE_PATH: TensorRT SG sequence JSON path.
#   PYTORCH_DIR: TorchScript module directory.
#   CONDA_ENV_PREFIX: conda env prefix for PyTorch checks.
#   DAEMON_LOG: exam_daemon log file path.
#   LOG_DIR: process log directory.
#   READY_DIR: process ready marker directory.
#   GATE_FILE: process start gate file path.
#   SCHED_FIFO: 1 enables sudo/SCHED_FIFO, 0 disables it.
#   VERBOSE: 1 prints captured logs.
#   EXAM_CONFIG_PATH: daemon config YAML path.
#   LD_LIBRARY_PATH: runtime library search path.

cd "$(dirname "$0")"

source ./scripts/sudo_helpers.sh

ITERATIONS=${ITERATIONS:-5}
EXPECTED_SG_PER_REQUEST=${EXPECTED_SG_PER_REQUEST:-3}
PROCESS_TIMEOUT=${PROCESS_TIMEOUT:-120s}
PYTORCH_INPUT_ID=${PYTORCH_INPUT_ID:-100}
TENSOR_RT_INPUT_ID=${TENSOR_RT_INPUT_ID:-200}
PYTORCH_SG_SEQUENCE_PATH=${PYTORCH_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_pytorch.json}
TENSOR_RT_SG_SEQUENCE_PATH=${TENSOR_RT_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_tensor_rt_gpu.json}
PYTORCH_DIR=${PYTORCH_DIR:-artifacts/resnet18/pytorch}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/home/rubis/workspace/miniconda3/envs/exam}
DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_resnet18_pytorch_tensorrt_interleave.log}
LOG_DIR=${LOG_DIR:-/tmp/exam_resnet18_pytorch_tensorrt_interleave_logs}
READY_DIR=${READY_DIR:-/tmp/exam_resnet18_pytorch_tensorrt_ready}
GATE_FILE=${GATE_FILE:-/tmp/exam_resnet18_pytorch_tensorrt_gate}
SCHED_FIFO=${SCHED_FIFO:-1}
VERBOSE=${VERBOSE:-0}

daemon_pid=""
daemon_wrapper_pid=""
daemon_pid_file=""
process_pids=()
terminal_state=""
disable_sched_fifo=1
if [[ "$SCHED_FIFO" == "1" ]]; then
    exam_require_passwordless_sudo "[03]"
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
        echo "[03] $name pid=$pid did not stop; sending SIGKILL" >&2
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
    for pid in "${process_pids[@]}"; do
        terminate_pid "$pid" "resnet18_process"
    done
    stop_daemon
    cleanup_shm
    rm -rf "$READY_DIR"
    rm -f "$GATE_FILE"
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

has_pytorch_enabled_daemon() {
    [[ -x ./exam_daemon ]] && ldd ./exam_daemon 2>/dev/null | grep -q 'libtorch_cpu'
}

ensure_built() {
    if [[ -x ./resnet18_process ]] && has_pytorch_enabled_daemon; then
        echo "[03] build skipped: PyTorch-enabled binaries found"
        return
    fi

    echo "[03] building PyTorch-enabled binaries..."
    PATH="$CONDA_ENV_PREFIX/bin:$PATH" \
        PYTHONNOUSERSITE=1 \
        ENABLE_PYTORCH=1 \
        ./build.sh
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

wait_for_ready_processes() {
    local expected_count=$1
    for _ in $(seq 1 200); do
        local ready_count
        ready_count=$(find "$READY_DIR" -name 'input_*.ready' 2>/dev/null | wc -l)
        if [[ "$ready_count" -eq "$expected_count" ]]; then
            return 0
        fi
        sleep 0.05
    done

    echo "timed out waiting for ready processes" >&2
    return 1
}

wait_for_daemon_ready() {
    for _ in $(seq 1 200); do
        if grep -q 'daemon: ready' "$DAEMON_LOG" 2>/dev/null; then
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
    cleanup_shm
    daemon_pid=""
    daemon_wrapper_pid=""
    daemon_pid_file=""

    if [[ ${#EXAM_SUDO[@]} -eq 0 ]]; then
        env \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            ./exam_daemon mock-interleaving > "$DAEMON_LOG" 2>&1 < /dev/null &
        daemon_pid=$!
    else
        daemon_pid_file=$(mktemp /tmp/exam_daemon_pid.XXXXXX)
        rm -f "$daemon_pid_file"
        "${EXAM_SUDO[@]}" env \
            EXAM_DAEMON_PID_FILE="$daemon_pid_file" \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            bash -c 'echo $$ > "$EXAM_DAEMON_PID_FILE"; exec ./exam_daemon mock-interleaving' \
            > "$DAEMON_LOG" 2>&1 < /dev/null &
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

    wait_for_daemon_ready
}

start_timed_resnet18_process() {
    local log_file=$1
    shift

    "${EXAM_SUDO[@]}" env \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        timeout "$PROCESS_TIMEOUT" ./resnet18_process "$@" \
        > "$log_file" 2>&1 < /dev/null &
    process_pids+=("$!")
}

extract_output_hashes() {
    local log_file=$1
    sed -n 's/.* output_hash=\([0-9a-f][0-9a-f]*\) .*/\1/p' "$log_file" \
        | paste -sd' ' -
}

count_results() {
    local log_file=$1
    grep -c '^RESULT ' "$log_file" || true
}

count_launches() {
    local worker_name=$1
    grep -c "worker=$worker_name" "$DAEMON_LOG" || true
}

count_worker_switches() {
    awk '
        /daemon: launch SG/ {
            worker = ""
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^worker=/) {
                    split($i, parts, "=")
                    worker = parts[2]
                }
            }
            if (worker != "") {
                if (last != "" && worker != last) {
                    ++switches
                }
                last = worker
            }
        }
        END { print switches + 0 }
    ' "$DAEMON_LOG"
}

print_logs() {
    echo "--- launch order: $DAEMON_LOG ---"
    grep 'daemon: launch SG' "$DAEMON_LOG" || true
    echo "--- daemon log: $DAEMON_LOG ---"
    sed -n '1,360p' "$DAEMON_LOG"
    echo "--- pytorch process log: $LOG_DIR/pytorch.log ---"
    sed -n '1,220p' "$LOG_DIR/pytorch.log"
    echo "--- tensor_rt_gpu process log: $LOG_DIR/tensor_rt_gpu.log ---"
    sed -n '1,220p' "$LOG_DIR/tensor_rt_gpu.log"
}

ensure_exam_env
ensure_pytorch_artifacts
ensure_built

rm -rf "$LOG_DIR" "$READY_DIR"
mkdir -p "$LOG_DIR" "$READY_DIR"
rm -f "$GATE_FILE"
cleanup_shm

start_daemon

process_pids=()
start_timed_resnet18_process "$LOG_DIR/pytorch.log" \
    "$PYTORCH_INPUT_ID" \
    "$PYTORCH_SG_SEQUENCE_PATH" \
    "$READY_DIR" \
    "$GATE_FILE" \
    "$ITERATIONS"

start_timed_resnet18_process "$LOG_DIR/tensor_rt_gpu.log" \
    "$TENSOR_RT_INPUT_ID" \
    "$TENSOR_RT_SG_SEQUENCE_PATH" \
    "$READY_DIR" \
    "$GATE_FILE" \
    "$ITERATIONS"

wait_for_ready_processes 2
touch "$GATE_FILE"

statuses=()
for pid in "${process_pids[@]}"; do
    if wait "$pid"; then
        statuses+=(0)
    else
        statuses+=("$?")
    fi
done
process_pids=()

sleep 1
stop_daemon
cleanup_shm

pytorch_status=${statuses[0]:-1}
tensor_rt_status=${statuses[1]:-1}
pytorch_results=$(count_results "$LOG_DIR/pytorch.log")
tensor_rt_results=$(count_results "$LOG_DIR/tensor_rt_gpu.log")
pytorch_launches=$(count_launches pytorch-worker)
tensor_rt_launches=$(count_launches tensor-rt-gpu-worker)
worker_switches=$(count_worker_switches)
expected_launches=$((ITERATIONS * EXPECTED_SG_PER_REQUEST))

if [[ "$pytorch_status" -ne 0 || "$tensor_rt_status" -ne 0 ]]; then
    echo "MIXED_PROCESS_STATUSES pytorch=$pytorch_status tensor_rt_gpu=$tensor_rt_status"
    print_logs
    exit 1
fi

if [[ "$pytorch_results" -ne "$ITERATIONS" || "$tensor_rt_results" -ne "$ITERATIONS" ]]; then
    echo "unexpected result count" >&2
    echo "RESULT_COUNT pytorch=$pytorch_results tensor_rt_gpu=$tensor_rt_results expected=$ITERATIONS"
    print_logs
    exit 1
fi

if [[ "$pytorch_launches" -ne "$expected_launches" \
    || "$tensor_rt_launches" -ne "$expected_launches" ]]; then
    echo "unexpected launch count" >&2
    echo "LAUNCH_COUNT pytorch-worker=$pytorch_launches tensor-rt-gpu-worker=$tensor_rt_launches expected=$expected_launches"
    print_logs
    exit 1
fi

if [[ "$worker_switches" -lt 1 ]]; then
    echo "expected mixed worker launch order" >&2
    echo "WORKER_SWITCHES=$worker_switches"
    print_logs
    exit 1
fi

if grep -Eq 'has no previous SG output|previous SG output size does not match|worker loop stopped' "$DAEMON_LOG"; then
    echo "worker state error detected" >&2
    print_logs
    exit 1
fi

echo "MIXED_PROCESS_STATUSES pytorch=$pytorch_status tensor_rt_gpu=$tensor_rt_status"
echo "RESULT_COUNT pytorch=$pytorch_results tensor_rt_gpu=$tensor_rt_results"
echo "LAUNCH_COUNT pytorch-worker=$pytorch_launches tensor-rt-gpu-worker=$tensor_rt_launches"
echo "WORKER_SWITCHES=$worker_switches"
echo "PYTORCH_OUTPUT_HASHES=$(extract_output_hashes "$LOG_DIR/pytorch.log")"
echo "TENSORRT_OUTPUT_HASHES=$(extract_output_hashes "$LOG_DIR/tensor_rt_gpu.log")"
echo "MIXED_INTERLEAVING=PASS"

if [[ "$VERBOSE" == "1" ]]; then
    print_logs
fi
