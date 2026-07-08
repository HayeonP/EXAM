#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_single.log}
PROCESS_LOG=${PROCESS_LOG:-/tmp/exam_process_single.log}
SG_SEQUENCE_PATH=${SG_SEQUENCE_PATH:-artifacts/sample_model/sg_sequence.json}

cleanup() {
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

echo "[01] building..."
./build.sh

echo "[01] starting daemon..."
./exam_daemon > "$DAEMON_LOG" 2>&1 &
daemon_pid=$!
sleep 1

echo "[01] running process with SG sequence: $SG_SEQUENCE_PATH"
set +e
./process 1 256 256 "$SG_SEQUENCE_PATH" > "$PROCESS_LOG" 2>&1
process_status=$?
set -e
sleep 1

kill "$daemon_pid" 2>/dev/null || true
wait "$daemon_pid" 2>/dev/null || true
unset daemon_pid

echo "[01] done"
echo "PROCESS_STATUS=$process_status"
echo "--- daemon log: $DAEMON_LOG ---"
sed -n '1,260p' "$DAEMON_LOG"
echo "--- process log: $PROCESS_LOG ---"
sed -n '1,160p' "$PROCESS_LOG"

exit "$process_status"
