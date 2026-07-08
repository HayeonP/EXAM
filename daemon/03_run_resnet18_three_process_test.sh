#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_resnet18_three_process.log}
PROCESS_1_LOG=${PROCESS_1_LOG:-/tmp/exam_resnet18_process_1.log}
PROCESS_2_LOG=${PROCESS_2_LOG:-/tmp/exam_resnet18_process_2.log}
PROCESS_3_LOG=${PROCESS_3_LOG:-/tmp/exam_resnet18_process_3.log}
SG_SEQUENCE_PATH=${SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence.json}
VERBOSE=${VERBOSE:-0}
process_pids=()

cleanup() {
    for pid in "${process_pids[@]}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n "${daemon_pid:-}" ]]; then
        kill "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    rm -f /dev/shm/exam_event_queue \
          /dev/shm/exam_channel_* \
          /dev/shm/exam_client_control_*
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

rm -f /dev/shm/exam_event_queue \
      /dev/shm/exam_channel_* \
      /dev/shm/exam_client_control_*

./exam_daemon mock-fifo > "$DAEMON_LOG" 2>&1 &
daemon_pid=$!
sleep 1

process_pids=()
./resnet18_process 1 "$SG_SEQUENCE_PATH" > "$PROCESS_1_LOG" 2>&1 &
process_pids+=("$!")
./resnet18_process 2 "$SG_SEQUENCE_PATH" > "$PROCESS_2_LOG" 2>&1 &
process_pids+=("$!")
./resnet18_process 3 "$SG_SEQUENCE_PATH" > "$PROCESS_3_LOG" 2>&1 &
process_pids+=("$!")

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
kill "$daemon_pid" 2>/dev/null || true
wait "$daemon_pid" 2>/dev/null || true
unset daemon_pid

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
