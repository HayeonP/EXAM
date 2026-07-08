#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_two_process.log}
PROCESS_A_LOG=${PROCESS_A_LOG:-/tmp/exam_process_a.log}
PROCESS_B_LOG=${PROCESS_B_LOG:-/tmp/exam_process_b.log}
SG_SEQUENCE_PATH=${SG_SEQUENCE_PATH:-artifacts/sample_model/sg_sequence.json}

cleanup() {
    if [[ -n "${process_a_pid:-}" ]]; then
        kill "$process_a_pid" 2>/dev/null || true
        wait "$process_a_pid" 2>/dev/null || true
    fi
    if [[ -n "${process_b_pid:-}" ]]; then
        kill "$process_b_pid" 2>/dev/null || true
        wait "$process_b_pid" 2>/dev/null || true
    fi
    if [[ -n "${daemon_pid:-}" ]]; then
        kill "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    rm -f /dev/shm/exam_event_queue \
          /dev/shm/exam_channel_* \
          /dev/shm/exam_client_control_*
}
trap cleanup EXIT

rm -f /dev/shm/exam_event_queue \
      /dev/shm/exam_channel_* \
      /dev/shm/exam_client_control_*

echo "[02] building..."
./build.sh
echo "[02] generating sample model..."
./build_sample_model

echo "[02] starting daemon..."
./exam_daemon > "$DAEMON_LOG" 2>&1 &
daemon_pid=$!
sleep 1

echo "[02] running two processes with SG sequence: $SG_SEQUENCE_PATH"
./process 1 256 256 "$SG_SEQUENCE_PATH" > "$PROCESS_A_LOG" 2>&1 &
process_a_pid=$!
./process 1 256 256 "$SG_SEQUENCE_PATH" > "$PROCESS_B_LOG" 2>&1 &
process_b_pid=$!

set +e
wait "$process_a_pid"
process_a_status=$?
unset process_a_pid

wait "$process_b_pid"
process_b_status=$?
unset process_b_pid
set -e

sleep 1
kill "$daemon_pid" 2>/dev/null || true
wait "$daemon_pid" 2>/dev/null || true
unset daemon_pid

echo "[02] done"
echo "PROCESS_A_STATUS=$process_a_status"
echo "PROCESS_B_STATUS=$process_b_status"
echo "--- daemon log: $DAEMON_LOG ---"
sed -n '1,360p' "$DAEMON_LOG"
echo "--- process A log: $PROCESS_A_LOG ---"
sed -n '1,160p' "$PROCESS_A_LOG"
echo "--- process B log: $PROCESS_B_LOG ---"
sed -n '1,160p' "$PROCESS_B_LOG"

if [[ "$process_a_status" -ne 0 || "$process_b_status" -ne 0 ]]; then
    exit 1
fi
