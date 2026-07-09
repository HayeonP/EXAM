#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source ./scripts/realtime_sudo.sh
exam_require_passwordless_sudo "[03]"

ITERATIONS=${ITERATIONS:-5}
EXPECTED_SG_PER_REQUEST=${EXPECTED_SG_PER_REQUEST:-3}
PROCESS_TIMEOUT=${PROCESS_TIMEOUT:-120s}
PYTORCH_INPUT_ID=${PYTORCH_INPUT_ID:-100}
TENSOR_RT_INPUT_ID=${TENSOR_RT_INPUT_ID:-200}
PYTORCH_SG_SEQUENCE_PATH=${PYTORCH_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_pytorch.json}
TENSOR_RT_SG_SEQUENCE_PATH=${TENSOR_RT_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence.json}
PYTORCH_PICKLE_DIR=${PYTORCH_PICKLE_DIR:-artifacts/resnet18/pytorch}
PYTORCH_TS_DIR=${PYTORCH_TS_DIR:-artifacts/resnet18/pytorch_ts}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/home/rubis/workspace/miniconda3/envs/exam}
DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_resnet18_pytorch_tensorrt_interleave.log}
LOG_DIR=${LOG_DIR:-/tmp/exam_resnet18_pytorch_tensorrt_interleave_logs}
READY_DIR=${READY_DIR:-/tmp/exam_resnet18_pytorch_tensorrt_ready}
GATE_FILE=${GATE_FILE:-/tmp/exam_resnet18_pytorch_tensorrt_gate}
VERBOSE=${VERBOSE:-0}

daemon_pid=""
process_pids=()

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
          /dev/shm/exam_channel_* \
          /dev/shm/exam_client_control_*
}

cleanup() {
    for pid in "${process_pids[@]}"; do
        terminate_pid "$pid" "resnet18_process"
    done
    terminate_pid "$daemon_pid" "exam_daemon"
    cleanup_shm
    rm -rf "$READY_DIR"
    rm -f "$GATE_FILE"
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

ensure_pytorch_torchscript_artifacts() {
    if [[ -f "$PYTORCH_TS_DIR/sg1.pt" \
        && -f "$PYTORCH_TS_DIR/sg2.pt" \
        && -f "$PYTORCH_TS_DIR/sg3.pt" ]]; then
        return
    fi

    echo "[03] exporting PyTorch TorchScript SG artifacts..."
    PYTORCH_PICKLE_DIR="$PYTORCH_PICKLE_DIR" \
    PYTORCH_TS_DIR="$PYTORCH_TS_DIR" \
    PYTHONNOUSERSITE=1 \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import os
import torch

source_dir = os.environ["PYTORCH_PICKLE_DIR"]
target_dir = os.environ["PYTORCH_TS_DIR"]
samples = {
    "sg1": torch.randn(1, 3, 224, 224),
    "sg2": torch.randn(1, 64, 56, 56),
    "sg3": torch.randn(1, 256, 14, 14),
}

os.makedirs(target_dir, exist_ok=True)
for name, sample in samples.items():
    source = os.path.join(source_dir, f"{name}.pt")
    target = os.path.join(target_dir, f"{name}.pt")
    model = torch.load(source, map_location="cpu")
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, strict=False)
        traced = torch.jit.freeze(traced.eval())
        traced.save(target)
PY
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
ensure_pytorch_torchscript_artifacts
ensure_built

rm -rf "$LOG_DIR" "$READY_DIR"
mkdir -p "$LOG_DIR" "$READY_DIR"
rm -f "$GATE_FILE"
cleanup_shm

"${EXAM_SUDO[@]}" env EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    ./exam_daemon mock-interleaving > "$DAEMON_LOG" 2>&1 &
daemon_pid=$!
wait_for_daemon_ready

process_pids=()
"${EXAM_SUDO[@]}" timeout "$PROCESS_TIMEOUT" ./resnet18_process \
    "$PYTORCH_INPUT_ID" \
    "$PYTORCH_SG_SEQUENCE_PATH" \
    "$READY_DIR" \
    "$GATE_FILE" \
    "$ITERATIONS" \
    > "$LOG_DIR/pytorch.log" 2>&1 &
process_pids+=("$!")

"${EXAM_SUDO[@]}" timeout "$PROCESS_TIMEOUT" ./resnet18_process \
    "$TENSOR_RT_INPUT_ID" \
    "$TENSOR_RT_SG_SEQUENCE_PATH" \
    "$READY_DIR" \
    "$GATE_FILE" \
    "$ITERATIONS" \
    > "$LOG_DIR/tensor_rt_gpu.log" 2>&1 &
process_pids+=("$!")

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
terminate_pid "$daemon_pid" "exam_daemon"
daemon_pid=""
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
