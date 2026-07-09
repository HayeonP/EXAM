#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODE=${1:---all}

usage() {
    cat <<'EOF'
Usage:
  ./terminate.sh              Kill EXAM daemon/client processes and clean shm.
  ./terminate.sh --all        Same as default.
  ./terminate.sh --daemon     Kill exam_daemon only and clean shm.
  ./terminate.sh --shm-only   Clean EXAM shared-memory files only.

Use sudo when previous runs left root-owned /dev/shm/exam_* files.
EOF
}

cleanup_shm() {
    local failed=0
    rm -f /dev/shm/exam_event_queue \
          /dev/shm/exam_channel_* \
          /dev/shm/exam_client_control_* 2>/dev/null || failed=1

    if compgen -G '/dev/shm/exam_event_queue' >/dev/null \
        || compgen -G '/dev/shm/exam_channel_*' >/dev/null \
        || compgen -G '/dev/shm/exam_client_control_*' >/dev/null; then
        failed=1
    fi

    if [[ "$failed" -ne 0 ]]; then
        echo "[terminate] shared-memory cleanup incomplete; rerun with sudo if files are root-owned" >&2
        ls -l /dev/shm/exam_event_queue \
              /dev/shm/exam_channel_* \
              /dev/shm/exam_client_control_* 2>/dev/null || true
        return 1
    fi
}

wait_for_pid_exit() {
    local pid=$1
    for _ in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
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
    local pid=$1
    local label=$2
    if ! kill -0 "$pid" 2>/dev/null; then
        return
    fi

    echo "[terminate] stopping $label pid=$pid"
    kill "$pid" 2>/dev/null || true
    if ! wait_for_pid_exit "$pid"; then
        echo "[terminate] $label pid=$pid did not stop; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        wait_for_pid_exit "$pid" || true
    fi
}

matching_pids() {
    local include_clients=$1
    ps -eo pid=,comm= | awk -v include_clients="$include_clients" '
        $2 == "exam_daemon" {
            print $1, $2
        }
        include_clients == "1" && ($2 == "resnet18_process" || $2 == "process") {
            print $1, $2
        }
    '
}

terminate_processes() {
    local include_clients=$1
    local found=0
    while read -r pid name; do
        if [[ -z "${pid:-}" ]]; then
            continue
        fi
        found=1
        terminate_pid "$pid" "$name"
    done < <(matching_pids "$include_clients")

    if [[ "$found" -eq 0 ]]; then
        echo "[terminate] no matching EXAM processes found"
    fi
}

case "$MODE" in
    --all)
        terminate_processes 1
        cleanup_shm || true
        ;;
    --daemon)
        terminate_processes 0
        cleanup_shm || true
        ;;
    --shm-only)
        cleanup_shm
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
