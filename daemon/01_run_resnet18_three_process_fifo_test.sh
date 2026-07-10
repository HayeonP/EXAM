#!/bin/bash
set -euo pipefail

# Usage: VAR=value ./01_run_resnet18_three_process_fifo_test.sh
# Inputs:
#   DAEMON_LOG: exam_daemon log file path.
#   PROCESS_1_LOG: process 1 log file path.
#   PROCESS_2_LOG: process 2 log file path.
#   PROCESS_3_LOG: process 3 log file path.
#   SG_SEQUENCE_PATH: TensorRT SG sequence JSON path.
#   SCHED_FIFO: 1 enables sudo/SCHED_FIFO, 0 disables it.
#   VERBOSE: 1 prints captured logs.
#   EXAM_CONFIG_PATH: daemon config YAML path.
#   LD_LIBRARY_PATH: runtime library search path.

cd "$(dirname "$0")"

source ./scripts/sudo_helpers.sh

DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_resnet18_three_process.log}
PROCESS_1_LOG=${PROCESS_1_LOG:-/tmp/exam_resnet18_process_1.log}
PROCESS_2_LOG=${PROCESS_2_LOG:-/tmp/exam_resnet18_process_2.log}
PROCESS_3_LOG=${PROCESS_3_LOG:-/tmp/exam_resnet18_process_3.log}
SG_SEQUENCE_PATH=${SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_tensor_rt_gpu.json}
SCHED_FIFO=${SCHED_FIFO:-1}
VERBOSE=${VERBOSE:-0}
daemon_pid=""
daemon_wrapper_pid=""
daemon_pid_file=""
process_pids=()
terminal_state=""
disable_sched_fifo=1
if [[ "$SCHED_FIFO" == "1" ]]; then
    exam_require_passwordless_sudo "[01]"
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
        echo "[01] $name pid=$pid did not stop; sending SIGKILL" >&2
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
    if [[ -n "$terminal_state" ]]; then
        stty "$terminal_state" 2>/dev/null || true
    fi
}
trap cleanup EXIT

extract_output_hash() {
    local log_file=$1
    sed -n 's/.* output_hash=\([0-9a-f][0-9a-f]*\) .*/\1/p' "$log_file" | tail -1
}

print_logs() {
    echo "--- launch order: $DAEMON_LOG ---"
    grep 'daemon: launch SG' "$DAEMON_LOG" || true
    echo "--- daemon log: $DAEMON_LOG ---"
    sed -n '1,260p' "$DAEMON_LOG"
    echo "--- process 1 log: $PROCESS_1_LOG ---"
    sed -n '1,120p' "$PROCESS_1_LOG"
    echo "--- process 2 log: $PROCESS_2_LOG ---"
    sed -n '1,120p' "$PROCESS_2_LOG"
    echo "--- process 3 log: $PROCESS_3_LOG ---"
    sed -n '1,120p' "$PROCESS_3_LOG"
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

ensure_built() {
    if [[ -x ./exam_daemon && -x ./resnet18_process ]]; then
        echo "[01] build skipped: existing binaries found"
        return
    fi

    echo "[01] building..."
    ./build.sh
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
            ./exam_daemon mock-fifo > "$DAEMON_LOG" 2>&1 < /dev/null &
        daemon_pid=$!
    else
        daemon_pid_file=$(mktemp /tmp/exam_daemon_pid.XXXXXX)
        rm -f "$daemon_pid_file"
        "${EXAM_SUDO[@]}" env \
            EXAM_DAEMON_PID_FILE="$daemon_pid_file" \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            bash -c 'echo $$ > "$EXAM_DAEMON_PID_FILE"; exec ./exam_daemon mock-fifo' \
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

start_process() {
    local input_id=$1
    local log_file=$2

    "${EXAM_SUDO[@]}" env \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        ./resnet18_process "$input_id" "$SG_SEQUENCE_PATH" \
        > "$log_file" 2>&1 < /dev/null &
    process_pids+=("$!")
}

cleanup_shm

ensure_built

start_daemon

process_pids=()
start_process 1 "$PROCESS_1_LOG"
start_process 2 "$PROCESS_2_LOG"
start_process 3 "$PROCESS_3_LOG"

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

hash1=$(extract_output_hash "$PROCESS_1_LOG")
hash2=$(extract_output_hash "$PROCESS_2_LOG")
hash3=$(extract_output_hash "$PROCESS_3_LOG")

for status in "${statuses[@]}"; do
    if [[ "$status" -ne 0 ]]; then
        echo "PROCESS_STATUSES=${statuses[*]}"
        print_logs
        exit 1
    fi
done

if [[ -z "$hash1" || -z "$hash2" || -z "$hash3" ]]; then
    echo "missing output hash" >&2
    print_logs
    exit 1
fi

if [[ "$hash1" == "$hash2" || "$hash1" == "$hash3" || "$hash2" == "$hash3" ]]; then
    echo "expected different inputs to produce different output hashes" >&2
    echo "OUTPUT_HASH input_id=1 $hash1"
    echo "OUTPUT_HASH input_id=2 $hash2"
    echo "OUTPUT_HASH input_id=3 $hash3"
    print_logs
    exit 1
fi

echo "PROCESS_STATUSES=${statuses[*]}"
echo "OUTPUT_HASH input_id=1 $hash1"
echo "OUTPUT_HASH input_id=2 $hash2"
echo "OUTPUT_HASH input_id=3 $hash3"
echo "OUTPUT_UNIQUE=PASS"

if [[ "$VERBOSE" == "1" ]]; then
    print_logs
fi
