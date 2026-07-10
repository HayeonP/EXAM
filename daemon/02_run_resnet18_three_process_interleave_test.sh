#!/bin/bash
set -euo pipefail

# Usage: VAR=value ./02_run_resnet18_three_process_interleave_test.sh
# Inputs:
#   SG_SEQUENCE_PATH: TensorRT SG sequence JSON path.
#   BASELINE_DAEMON_LOG: baseline daemon log file path.
#   INTERLEAVE_DAEMON_LOG: interleaving daemon log file path.
#   BASELINE_LOG_DIR: baseline process log directory.
#   INTERLEAVE_LOG_DIR: interleaving process log directory.
#   READY_DIR: process ready marker directory.
#   GATE_FILE: process start gate file path.
#   SCHED_FIFO: 1 enables sudo/SCHED_FIFO, 0 disables it.
#   VERBOSE: 1 prints captured logs.
#   EXAM_CONFIG_PATH: daemon config YAML path.
#   LD_LIBRARY_PATH: runtime library search path.

cd "$(dirname "$0")"

source ./scripts/sudo_helpers.sh

SG_SEQUENCE_PATH=${SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_tensor_rt_gpu.json}
BASELINE_DAEMON_LOG=${BASELINE_DAEMON_LOG:-/tmp/exam_daemon_resnet18_baseline.log}
INTERLEAVE_DAEMON_LOG=${INTERLEAVE_DAEMON_LOG:-/tmp/exam_daemon_resnet18_interleave.log}
BASELINE_LOG_DIR=${BASELINE_LOG_DIR:-/tmp/exam_resnet18_baseline_logs}
INTERLEAVE_LOG_DIR=${INTERLEAVE_LOG_DIR:-/tmp/exam_resnet18_interleave_logs}
READY_DIR=${READY_DIR:-/tmp/exam_resnet18_ready}
GATE_FILE=${GATE_FILE:-/tmp/exam_resnet18_gate}
SCHED_FIFO=${SCHED_FIFO:-1}
VERBOSE=${VERBOSE:-0}

daemon_pid=""
daemon_wrapper_pid=""
daemon_pid_file=""
process_pids=()
terminal_state=""
disable_sched_fifo=1
if [[ "$SCHED_FIFO" == "1" ]]; then
    exam_require_passwordless_sudo "[02]"
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
        echo "[02] $name pid=$pid did not stop; sending SIGKILL" >&2
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

extract_output_hash() {
    local log_file=$1
    sed -n 's/.* output_hash=\([0-9a-f][0-9a-f]*\) .*/\1/p' "$log_file" | tail -1
}

start_daemon() {
    local policy=$1
    local log_file=$2

    cleanup_shm
    daemon_pid=""
    daemon_wrapper_pid=""
    daemon_pid_file=""

    if [[ ${#EXAM_SUDO[@]} -eq 0 ]]; then
        env \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            ./exam_daemon "$policy" > "$log_file" 2>&1 < /dev/null &
        daemon_pid=$!
    else
        daemon_pid_file=$(mktemp /tmp/exam_daemon_pid.XXXXXX)
        rm -f "$daemon_pid_file"
        "${EXAM_SUDO[@]}" env \
            EXAM_DAEMON_PID_FILE="$daemon_pid_file" \
            EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
            bash -c 'echo $$ > "$EXAM_DAEMON_PID_FILE"; exec ./exam_daemon "$@"' \
            bash "$policy" \
            > "$log_file" 2>&1 < /dev/null &
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

    wait_for_daemon_ready "$log_file"
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

run_resnet18_process() {
    "${EXAM_SUDO[@]}" env \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        ./resnet18_process "$@"
}

start_resnet18_process() {
    local log_file=$1
    shift

    run_resnet18_process "$@" > "$log_file" 2>&1 < /dev/null &
    process_pids+=("$!")
}

wait_for_daemon_ready() {
    local log_file=$1
    for _ in $(seq 1 200); do
        if grep -q 'daemon: ready' "$log_file" 2>/dev/null; then
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

validate_interleaving_launch_order() {
    local log_file=$1
    awk '
        /daemon: launch SG/ {
            sg = ""
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^sg=/) {
                    split($i, parts, "=")
                    sg = parts[2]
                }
            }

            if (sg == "0") {
                if (sg1 > 0 || sg2 > 0) {
                    bad = 1
                }
                ++sg0
            } else if (sg == "1") {
                if (sg0 < 3 || sg2 > 0) {
                    bad = 1
                }
                ++sg1
            } else if (sg == "2") {
                if (sg1 < 3) {
                    bad = 1
                }
                ++sg2
            }
        }
        END {
            if (sg0 != 3 || sg1 != 3 || sg2 != 3 || bad) {
                exit 1
            }
        }
    ' "$log_file"
}

print_logs() {
    echo "--- baseline daemon log: $BASELINE_DAEMON_LOG ---"
    sed -n '1,260p' "$BASELINE_DAEMON_LOG"
    for input_id in 1 2 3; do
        local log_file="$BASELINE_LOG_DIR/process_${input_id}.log"
        echo "--- baseline process $input_id log: $log_file ---"
        sed -n '1,120p' "$log_file"
    done

    echo "--- interleaving launch order: $INTERLEAVE_DAEMON_LOG ---"
    grep 'daemon: launch SG' "$INTERLEAVE_DAEMON_LOG" || true
    echo "--- interleaving daemon log: $INTERLEAVE_DAEMON_LOG ---"
    sed -n '1,320p' "$INTERLEAVE_DAEMON_LOG"
    for input_id in 1 2 3; do
        local log_file="$INTERLEAVE_LOG_DIR/process_${input_id}.log"
        echo "--- interleaving process $input_id log: $log_file ---"
        sed -n '1,120p' "$log_file"
    done
}

ensure_built() {
    if [[ -x ./exam_daemon && -x ./resnet18_process ]]; then
        echo "[02] build skipped: existing binaries found"
        return
    fi

    echo "[02] building..."
    ./build.sh
}

ensure_built

rm -rf "$BASELINE_LOG_DIR" "$INTERLEAVE_LOG_DIR" "$READY_DIR"
mkdir -p "$BASELINE_LOG_DIR" "$INTERLEAVE_LOG_DIR" "$READY_DIR"
rm -f "$GATE_FILE"

declare -A baseline_hashes
declare -A interleave_hashes

cleanup_shm
start_daemon mock-fifo "$BASELINE_DAEMON_LOG"
for input_id in 1 2 3; do
    log_file="$BASELINE_LOG_DIR/process_${input_id}.log"
    run_resnet18_process "$input_id" "$SG_SEQUENCE_PATH" \
        > "$log_file" 2>&1 < /dev/null
    baseline_hashes[$input_id]=$(extract_output_hash "$log_file")
    if [[ -z "${baseline_hashes[$input_id]}" ]]; then
        echo "missing baseline output hash for input_id=$input_id" >&2
        print_logs
        exit 1
    fi
done
stop_daemon
cleanup_shm

start_daemon mock-interleaving "$INTERLEAVE_DAEMON_LOG"
process_pids=()
for input_id in 1 2 3; do
    log_file="$INTERLEAVE_LOG_DIR/process_${input_id}.log"
    start_resnet18_process "$log_file" \
        "$input_id" "$SG_SEQUENCE_PATH" "$READY_DIR" "$GATE_FILE"
done

wait_for_ready_processes 3
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

for input_id in 1 2 3; do
    log_file="$INTERLEAVE_LOG_DIR/process_${input_id}.log"
    interleave_hashes[$input_id]=$(extract_output_hash "$log_file")
done

for status in "${statuses[@]}"; do
    if [[ "$status" -ne 0 ]]; then
        echo "INTERLEAVE_PROCESS_STATUSES=${statuses[*]}"
        print_logs
        exit 1
    fi
done

for input_id in 1 2 3; do
    if [[ -z "${interleave_hashes[$input_id]}" ]]; then
        echo "missing interleaved output hash for input_id=$input_id" >&2
        print_logs
        exit 1
    fi
    if [[ "${baseline_hashes[$input_id]}" != "${interleave_hashes[$input_id]}" ]]; then
        echo "output hash mismatch for input_id=$input_id" >&2
        echo "HASH_MATCH input_id=$input_id baseline=${baseline_hashes[$input_id]} interleaved=${interleave_hashes[$input_id]} result=FAIL"
        print_logs
        exit 1
    fi
done

if ! validate_interleaving_launch_order "$INTERLEAVE_DAEMON_LOG"; then
    echo "interleaving launch order check failed" >&2
    print_logs
    exit 1
fi

if grep -q 'tensor_rt worker has no previous SG output' "$INTERLEAVE_DAEMON_LOG"; then
    echo "missing previous SG output in TensorRT worker" >&2
    print_logs
    exit 1
fi

if grep -q 'previous SG output size does not match current SG input size' "$INTERLEAVE_DAEMON_LOG"; then
    echo "previous SG output size mismatch in TensorRT worker" >&2
    print_logs
    exit 1
fi

echo "INTERLEAVE_PROCESS_STATUSES=${statuses[*]}"
for input_id in 1 2 3; do
    echo "HASH_MATCH input_id=$input_id baseline=${baseline_hashes[$input_id]} interleaved=${interleave_hashes[$input_id]} result=PASS"
done
echo "INTERLEAVING_OUTPUT_MATCH=PASS"

if [[ "$VERBOSE" == "1" ]]; then
    print_logs
fi
