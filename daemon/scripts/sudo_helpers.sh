#!/bin/bash

EXAM_SUDO=()

exam_require_passwordless_sudo() {
    local label=${1:-[exam]}

    if [[ "$EUID" -eq 0 ]]; then
        EXAM_SUDO=()
        return
    fi

    if ! command -v sudo >/dev/null 2>&1; then
        echo "$label passwordless sudo is required, but sudo was not found" >&2
        exit 1
    fi

    if ! sudo -n true >/dev/null 2>&1; then
        echo "$label passwordless sudo is required for SCHED_FIFO; sudo -n failed" >&2
        echo "$label configure NOPASSWD sudo for these EXAM scripts/binaries, then rerun" >&2
        exit 1
    fi

    EXAM_SUDO=(sudo -n)
}

exam_pid_exists() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null || "${EXAM_SUDO[@]}" kill -0 "$pid" 2>/dev/null
}

exam_kill() {
    local signal=$1
    local pid=$2
    kill "-$signal" "$pid" 2>/dev/null \
        || "${EXAM_SUDO[@]}" kill "-$signal" "$pid" 2>/dev/null \
        || true
}

exam_rm() {
    rm "$@" 2>/dev/null || "${EXAM_SUDO[@]}" rm "$@" 2>/dev/null || true
}
