#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source ./scripts/realtime_sudo.sh

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

pytorch_worker_cpu_list() {
    local config_path
    config_path=$(default_config_path)
    if [[ -z "$config_path" || ! -f "$config_path" ]]; then
        echo "-"
        return
    fi

    local cpu_list
    cpu_list=$(sed -n 's/^[[:space:]]*pytorch_worker:[[:space:]]*\[\(.*\)\][[:space:]]*$/\1/p' "$config_path" \
        | head -n 1 \
        | tr -d '[:space:]')
    if [[ -z "$cpu_list" ]]; then
        echo "-"
        return
    fi
    echo "$cpu_list"
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
PYTORCH_NUM_THREADS=${PYTORCH_NUM_THREADS:-$(default_pytorch_num_threads)}
PYTORCH_INTEROP_THREADS=${PYTORCH_INTEROP_THREADS:-1}
SCHED_FIFO=${SCHED_FIFO:-1}
DIRECT_SCHED_FIFO_PRIORITY=${DIRECT_SCHED_FIFO_PRIORITY:-}
if [[ -z "$DIRECT_SCHED_FIFO_PRIORITY" ]]; then
    if [[ "$SCHED_FIFO" == "1" ]]; then
        DIRECT_SCHED_FIFO_PRIORITY=85
    else
        DIRECT_SCHED_FIFO_PRIORITY=0
    fi
fi
INPUT_ID=${INPUT_ID:-300}
PYTORCH_SG_SEQUENCE_PATH=${PYTORCH_SG_SEQUENCE_PATH:-artifacts/resnet18/sg_sequence_pytorch.json}
PYTORCH_PICKLE_DIR=${PYTORCH_PICKLE_DIR:-artifacts/resnet18/pytorch}
PYTORCH_TS_DIR=${PYTORCH_TS_DIR:-artifacts/resnet18/pytorch_ts}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/home/rubis/workspace/miniconda3/envs/exam}
DAEMON_LOG=${DAEMON_LOG:-/tmp/exam_daemon_pytorch_response_time.log}
DIRECT_LOG=${DIRECT_LOG:-/tmp/exam_pytorch_direct_response_time.log}
CPP_DIRECT_LOG=${CPP_DIRECT_LOG:-/tmp/exam_pytorch_cpp_direct_response_time.log}
WORKER_LOG=${WORKER_LOG:-/tmp/exam_pytorch_worker_response_time.log}
VERBOSE=${VERBOSE:-0}
RUN_ORDER=${RUN_ORDER:-direct-first}

daemon_pid=""
daemon_wrapper_pid=""
daemon_pid_file=""
terminal_state=""
disable_sched_fifo=1
if [[ "$SCHED_FIFO" == "1" ]]; then
    exam_require_passwordless_sudo "[04]"
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
        echo "[04] $name pid=$pid did not stop; sending SIGKILL" >&2
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
    terminate_pid "$daemon_pid" "exam_daemon"
    if [[ -n "$daemon_wrapper_pid" ]]; then
        wait "$daemon_wrapper_pid" 2>/dev/null || true
    fi
    if [[ -n "$daemon_pid_file" ]]; then
        rm -f "$daemon_pid_file" 2>/dev/null || true
    fi
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

ensure_pytorch_torchscript_artifacts() {
    if [[ -f "$PYTORCH_TS_DIR/sg1.pt" \
        && -f "$PYTORCH_TS_DIR/sg2.pt" \
        && -f "$PYTORCH_TS_DIR/sg3.pt" ]]; then
        return
    fi

    echo "[04] exporting PyTorch TorchScript SG artifacts..."
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

has_pytorch_enabled_daemon() {
    [[ -x ./exam_daemon ]] && ldd ./exam_daemon 2>/dev/null | grep -q 'libtorch_cpu'
}

has_pytorch_cpp_direct() {
    [[ -x ./resnet18_pytorch_direct ]] \
        && ldd ./resnet18_pytorch_direct 2>/dev/null | grep -q 'libtorch_cpu'
}

ensure_built() {
    if [[ -x ./resnet18_process ]] \
        && has_pytorch_enabled_daemon \
        && has_pytorch_cpp_direct; then
        echo "[04] build skipped: PyTorch-enabled binaries found"
        return
    fi

    echo "[04] building PyTorch-enabled binaries..."
    PATH="$CONDA_ENV_PREFIX/bin:$PATH" \
        PYTHONNOUSERSITE=1 \
        ENABLE_PYTORCH=1 \
        ./build.sh
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

run_direct() {
    echo "[04] running direct PyTorch measurement..."
    "${EXAM_SUDO[@]}" env \
        EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        EXAM_DIRECT_SCHED_FIFO_PRIORITY="$DIRECT_SCHED_FIFO_PRIORITY" \
        PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
        OMP_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        MKL_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        PYTHONNOUSERSITE=1 \
        "$CONDA_ENV_PREFIX/bin/python" \
        apps/resnet18_pytorch_direct.py \
        "$INPUT_ID" \
        "$PYTORCH_TS_DIR" \
        "$WARMUP" \
        "$ITERATIONS" \
        > "$DIRECT_LOG" 2>&1
}

run_cpp_direct() {
    echo "[04] running C++ direct LibTorch measurement..."
    "${EXAM_SUDO[@]}" env \
        EXAM_CONFIG_PATH="${EXAM_CONFIG_PATH:-}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        EXAM_DIRECT_SCHED_FIFO_PRIORITY="$DIRECT_SCHED_FIFO_PRIORITY" \
        EXAM_PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        EXAM_PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
        PYTORCH_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        PYTORCH_INTEROP_THREADS="$PYTORCH_INTEROP_THREADS" \
        OMP_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        MKL_NUM_THREADS="$PYTORCH_NUM_THREADS" \
        EXAM_TORCH_JIT_OPTIMIZE=1 \
        EXAM_TORCH_JIT_EXECUTOR_MODE=0 \
        EXAM_TORCH_JIT_PROFILING_MODE=0 \
        EXAM_TORCH_JIT_NUM_PROFILED_RUNS=1 \
        ./resnet18_pytorch_direct \
            "$INPUT_ID" \
            "$PYTORCH_TS_DIR" \
            "$WARMUP" \
            "$ITERATIONS" \
            > "$CPP_DIRECT_LOG" 2>&1
}

run_worker() {
    cleanup_shm
    echo "[04] running daemon-worker measurement..."
    daemon_pid_file=$(mktemp /tmp/exam_daemon_pid.XXXXXX)
    rm -f "$daemon_pid_file"
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
            ./exam_daemon mock-fifo > "$DAEMON_LOG" 2>&1 &
        daemon_pid=$!
    else
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
            > "$DAEMON_LOG" 2>&1 &
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

    "${EXAM_SUDO[@]}" env \
        EXAM_DISABLE_SCHED_FIFO="$disable_sched_fifo" \
        ./resnet18_process \
        "$INPUT_ID" \
        "$PYTORCH_SG_SEQUENCE_PATH" \
        "" \
        "" \
        "$((WARMUP + ITERATIONS))" \
        > "$WORKER_LOG" 2>&1

    terminate_pid "$daemon_pid" "exam_daemon"
    daemon_pid=""
    if [[ -n "$daemon_wrapper_pid" ]]; then
        wait "$daemon_wrapper_pid" 2>/dev/null || true
        daemon_wrapper_pid=""
    fi
    rm -f "$daemon_pid_file" 2>/dev/null || true
    daemon_pid_file=""
    cleanup_shm
}

extract_hashes() {
    local log_file=$1
    local prefix=$2
    sed -n "s/^$prefix.* output_hash=\\([0-9a-f][0-9a-f]*\\) .*/\\1/p" "$log_file" \
        | tail -n "$ITERATIONS" \
        | paste -sd' ' -
}

extract_latencies() {
    local log_file=$1
    local prefix=$2
    sed -n "s/^$prefix.* latency_us=\\([0-9][0-9]*\\).*/\\1/p" "$log_file" \
        | tail -n "$ITERATIONS" \
        | paste -sd' ' -
}

extract_field_values() {
    local log_file=$1
    local prefix=$2
    local field=$3
    sed -n "s/^$prefix.* $field=\\([0-9][0-9]*\\).*/\\1/p" "$log_file" \
        | tail -n "$ITERATIONS" \
        | paste -sd' ' -
}

extract_worker_sg_latencies() {
    DAEMON_LOG="$DAEMON_LOG" \
    ITERATIONS="$ITERATIONS" \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import os
import re

log_path = os.environ["DAEMON_LOG"]
iterations = int(os.environ["ITERATIONS"])
values = {0: [], 1: [], 2: []}
pattern = re.compile(r"daemon: (?:SG complete|request complete).* sg=(\d+) sg_latency_us=(\d+)")

with open(log_path, "r", encoding="utf-8") as log_file:
    for line in log_file:
        match = pattern.search(line)
        if not match:
            continue
        sg_id = int(match.group(1))
        if sg_id in values:
            values[sg_id].append(int(match.group(2)))

for sg_id in range(3):
    measured = values[sg_id][-iterations:]
    print(f"WORKER_SG{sg_id + 1}_US=" + " ".join(str(value) for value in measured))
PY
}

extract_worker_timing_latencies() {
    local field=$1
    DAEMON_LOG="$DAEMON_LOG" \
    ITERATIONS="$ITERATIONS" \
    FIELD="$field" \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import os
import re

log_path = os.environ["DAEMON_LOG"]
iterations = int(os.environ["ITERATIONS"])
field = os.environ["FIELD"]
values = {0: [], 1: [], 2: []}
pattern = re.compile(
    rf"pytorch-worker: timing .* sg=(\d+) .* {re.escape(field)}=(-?\d+)"
)

with open(log_path, "r", encoding="utf-8") as log_file:
    for line in log_file:
        match = pattern.search(line)
        if not match:
            continue
        sg_id = int(match.group(1))
        if sg_id in values:
            values[sg_id].append(int(match.group(2)))

field_name = field.removesuffix("_us").upper()
for sg_id in range(3):
    measured = values[sg_id][-iterations:]
    print(f"WORKER_{field_name}_SG{sg_id + 1}_US=" + " ".join(str(value) for value in measured))
PY
}

sum_sg_series() {
    SG1="$1" \
    SG2="$2" \
    SG3="$3" \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import os

series = [
    [int(value) for value in os.environ[name].split() if value]
    for name in ("SG1", "SG2", "SG3")
]
if any(len(values) != len(series[0]) for values in series):
    raise SystemExit("cannot sum SG series: latency series lengths differ")

summed = [
    sum(values[index] for values in series)
    for index in range(len(series[0]))
]
print(" ".join(str(value) for value in summed))
PY
}

subtract_sg_from_total() {
    TOTAL="$1" \
    SG1="$2" \
    SG2="$3" \
    SG3="$4" \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import os

total = [int(value) for value in os.environ["TOTAL"].split() if value]
sg_series = [
    [int(value) for value in os.environ[name].split() if value]
    for name in ("SG1", "SG2", "SG3")
]
if any(len(values) != len(total) for values in sg_series):
    raise SystemExit("cannot compute overhead: latency series lengths differ")

overhead = [
    total[index] - sum(values[index] for values in sg_series)
    for index in range(len(total))
]
print(" ".join(str(value) for value in overhead))
PY
}

stats_line() {
    local label=$1
    local latencies=$2
    LABEL="$label" LATENCIES="$latencies" "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import math
import os
import statistics

label = os.environ["LABEL"]
values = [int(value) for value in os.environ["LATENCIES"].split() if value]
if not values:
    raise SystemExit(f"{label}: no latencies")
print(
    f"{label}_STATS "
    f"count={len(values)} "
    f"min_us={min(values)} "
    f"avg_us={int(statistics.mean(values))} "
    f"p50_us={int(statistics.median(values))} "
    f"p99_us={sorted(values)[max(0, math.ceil(len(values) * 0.99) - 1)]} "
    f"max_us={max(values)}"
)
PY
}

ensure_series_count() {
    local label=$1
    local latencies=$2
    local count
    count=$(wc -w <<< "$latencies")
    if [[ "$count" -ne "$ITERATIONS" ]]; then
        echo "unexpected latency count $label=$count expected=$ITERATIONS" >&2
        print_logs
        exit 1
    fi
}

print_stats_row() {
    local label=$1
    local latencies=$2
    local stats
    stats=$(stats_line "$label" "$latencies")
    printf '%-10s %6s %10s %10s %10s %10s\n' \
        "$label" \
        "$(sed -n 's/.*count=\([0-9][0-9]*\).*/\1/p' <<< "$stats")" \
        "$(sed -n 's/.*min_us=\(-\?[0-9][0-9]*\).*/\1/p' <<< "$stats")" \
        "$(sed -n 's/.*avg_us=\(-\?[0-9][0-9]*\).*/\1/p' <<< "$stats")" \
        "$(sed -n 's/.*p50_us=\(-\?[0-9][0-9]*\).*/\1/p' <<< "$stats")" \
        "$(sed -n 's/.*max_us=\(-\?[0-9][0-9]*\).*/\1/p' <<< "$stats")"
}

print_compact_summary() {
    DIRECT_OVERALL="$direct_latencies" \
    DIRECT_SG1="$direct_sg1_latencies" \
    DIRECT_SG2="$direct_sg2_latencies" \
    DIRECT_SG3="$direct_sg3_latencies" \
    DIRECT_OVERHEAD="$direct_overhead_latencies" \
    CPP_OVERALL="$cpp_direct_latencies" \
    CPP_SG1="$cpp_direct_sg1_latencies" \
    CPP_SG2="$cpp_direct_sg2_latencies" \
    CPP_SG3="$cpp_direct_sg3_latencies" \
    CPP_OVERHEAD="$cpp_direct_overhead_latencies" \
    WORKER_OVERALL="$worker_latencies" \
    WORKER_SG1="$worker_forward_sg1_latencies" \
    WORKER_SG2="$worker_forward_sg2_latencies" \
    WORKER_SG3="$worker_forward_sg3_latencies" \
    WORKER_OVERHEAD="$worker_overhead_latencies" \
    "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import math
import os
import statistics

columns = ["overall", "sg1", "sg2", "sg3", "overhead"]
env_names = {
    "direct": {
        "overall": "DIRECT_OVERALL",
        "sg1": "DIRECT_SG1",
        "sg2": "DIRECT_SG2",
        "sg3": "DIRECT_SG3",
        "overhead": "DIRECT_OVERHEAD",
    },
    "cpp": {
        "overall": "CPP_OVERALL",
        "sg1": "CPP_SG1",
        "sg2": "CPP_SG2",
        "sg3": "CPP_SG3",
        "overhead": "CPP_OVERHEAD",
    },
    "worker": {
        "overall": "WORKER_OVERALL",
        "sg1": "WORKER_SG1",
        "sg2": "WORKER_SG2",
        "sg3": "WORKER_SG3",
        "overhead": "WORKER_OVERHEAD",
    },
}

def values(name):
    return [int(value) for value in os.environ[name].split() if value]

def stat(values, kind):
    if kind == "avg":
        return int(statistics.mean(values))
    if kind == "p50":
        return int(statistics.median(values))
    if kind == "p99":
        sorted_values = sorted(values)
        index = max(0, math.ceil(len(sorted_values) * 0.99) - 1)
        return sorted_values[index]
    if kind == "max":
        return max(values)
    raise ValueError(kind)

print("SUMMARY_US")
print(f"{'path':<8} {'stat':<4} " + " ".join(f"{column:>10}" for column in columns))
for path in ("python", "cpp", "worker"):
    source_path = "direct" if path == "python" else path
    for kind in ("avg", "p50", "p99", "max"):
        cells = [
            stat(values(env_names[source_path][column]), kind)
            for column in columns
        ]
        print(f"{path:<8} {kind:<4} " + " ".join(f"{cell:>10}" for cell in cells))
PY
}

print_logs() {
    echo "--- direct log: $DIRECT_LOG ---"
    sed -n '1,220p' "$DIRECT_LOG"
    echo "--- C++ direct log: $CPP_DIRECT_LOG ---"
    sed -n '1,220p' "$CPP_DIRECT_LOG"
    echo "--- daemon log: $DAEMON_LOG ---"
    sed -n '1,260p' "$DAEMON_LOG"
    echo "--- worker log: $WORKER_LOG ---"
    sed -n '1,220p' "$WORKER_LOG"
}

ensure_exam_env
ensure_pytorch_torchscript_artifacts
ensure_built
case "$RUN_ORDER" in
    direct-first)
        run_direct
        run_cpp_direct
        run_worker
        ;;
    worker-first)
        run_worker
        run_cpp_direct
        run_direct
        ;;
    *)
        echo "unknown RUN_ORDER=$RUN_ORDER; use direct-first or worker-first" >&2
        exit 1
        ;;
esac

direct_hashes=$(extract_hashes "$DIRECT_LOG" DIRECT_RESULT)
cpp_direct_hashes=$(extract_hashes "$CPP_DIRECT_LOG" CPP_DIRECT_RESULT)
worker_hashes=$(extract_hashes "$WORKER_LOG" RESULT)
direct_latencies=$(extract_latencies "$DIRECT_LOG" DIRECT_RESULT)
cpp_direct_latencies=$(extract_latencies "$CPP_DIRECT_LOG" CPP_DIRECT_RESULT)
worker_latencies=$(extract_latencies "$WORKER_LOG" RESULT)
direct_sg1_latencies=$(extract_field_values "$DIRECT_LOG" DIRECT_RESULT sg1_us)
direct_sg2_latencies=$(extract_field_values "$DIRECT_LOG" DIRECT_RESULT sg2_us)
direct_sg3_latencies=$(extract_field_values "$DIRECT_LOG" DIRECT_RESULT sg3_us)
cpp_direct_sg1_latencies=$(extract_field_values "$CPP_DIRECT_LOG" CPP_DIRECT_RESULT sg1_us)
cpp_direct_sg2_latencies=$(extract_field_values "$CPP_DIRECT_LOG" CPP_DIRECT_RESULT sg2_us)
cpp_direct_sg3_latencies=$(extract_field_values "$CPP_DIRECT_LOG" CPP_DIRECT_RESULT sg3_us)
worker_sg_latency_lines=$(extract_worker_sg_latencies)
worker_sg1_latencies=$(sed -n 's/^WORKER_SG1_US=//p' <<< "$worker_sg_latency_lines")
worker_sg2_latencies=$(sed -n 's/^WORKER_SG2_US=//p' <<< "$worker_sg_latency_lines")
worker_sg3_latencies=$(sed -n 's/^WORKER_SG3_US=//p' <<< "$worker_sg_latency_lines")
worker_forward_latency_lines=$(extract_worker_timing_latencies forward_us)
worker_forward_sg1_latencies=$(sed -n 's/^WORKER_FORWARD_SG1_US=//p' <<< "$worker_forward_latency_lines")
worker_forward_sg2_latencies=$(sed -n 's/^WORKER_FORWARD_SG2_US=//p' <<< "$worker_forward_latency_lines")
worker_forward_sg3_latencies=$(sed -n 's/^WORKER_FORWARD_SG3_US=//p' <<< "$worker_forward_latency_lines")

ensure_series_count direct "$direct_latencies"
ensure_series_count cpp_direct "$cpp_direct_latencies"
ensure_series_count worker "$worker_latencies"
ensure_series_count direct_sg1 "$direct_sg1_latencies"
ensure_series_count direct_sg2 "$direct_sg2_latencies"
ensure_series_count direct_sg3 "$direct_sg3_latencies"
ensure_series_count cpp_direct_sg1 "$cpp_direct_sg1_latencies"
ensure_series_count cpp_direct_sg2 "$cpp_direct_sg2_latencies"
ensure_series_count cpp_direct_sg3 "$cpp_direct_sg3_latencies"
direct_overhead_latencies=$(
    subtract_sg_from_total \
        "$direct_latencies" \
        "$direct_sg1_latencies" \
        "$direct_sg2_latencies" \
        "$direct_sg3_latencies"
)
ensure_series_count direct_overhead "$direct_overhead_latencies"
cpp_direct_overhead_latencies=$(
    subtract_sg_from_total \
        "$cpp_direct_latencies" \
        "$cpp_direct_sg1_latencies" \
        "$cpp_direct_sg2_latencies" \
        "$cpp_direct_sg3_latencies"
)
ensure_series_count cpp_direct_overhead "$cpp_direct_overhead_latencies"
ensure_series_count worker_sg1 "$worker_sg1_latencies"
ensure_series_count worker_sg2 "$worker_sg2_latencies"
ensure_series_count worker_sg3 "$worker_sg3_latencies"
ensure_series_count worker_forward_sg1 "$worker_forward_sg1_latencies"
ensure_series_count worker_forward_sg2 "$worker_forward_sg2_latencies"
ensure_series_count worker_forward_sg3 "$worker_forward_sg3_latencies"
worker_outside_sg_window_latencies=$(
    subtract_sg_from_total \
        "$worker_latencies" \
        "$worker_sg1_latencies" \
        "$worker_sg2_latencies" \
        "$worker_sg3_latencies"
)
ensure_series_count worker_outside_sg_window "$worker_outside_sg_window_latencies"
worker_sg_window_latencies=$(
    sum_sg_series \
        "$worker_sg1_latencies" \
        "$worker_sg2_latencies" \
        "$worker_sg3_latencies"
)
worker_forward_latencies=$(
    sum_sg_series \
        "$worker_forward_sg1_latencies" \
        "$worker_forward_sg2_latencies" \
        "$worker_forward_sg3_latencies"
)
worker_overhead_latencies=$(
    subtract_sg_from_total \
        "$worker_latencies" \
        "$worker_forward_sg1_latencies" \
        "$worker_forward_sg2_latencies" \
        "$worker_forward_sg3_latencies"
)
worker_inside_sg_nonforward_latencies=$(
    subtract_sg_from_total \
        "$worker_sg_window_latencies" \
        "$worker_forward_sg1_latencies" \
        "$worker_forward_sg2_latencies" \
        "$worker_forward_sg3_latencies"
)
ensure_series_count worker_sg_window "$worker_sg_window_latencies"
ensure_series_count worker_forward "$worker_forward_latencies"
ensure_series_count worker_overhead "$worker_overhead_latencies"
ensure_series_count worker_inside_sg_nonforward "$worker_inside_sg_nonforward_latencies"

if [[ "$direct_hashes" != "$cpp_direct_hashes" \
    || "$direct_hashes" != "$worker_hashes" ]]; then
    echo "output hash mismatch between Python direct, C++ direct, and daemon-worker runs" >&2
    echo "DIRECT_OUTPUT_HASHES=$direct_hashes"
    echo "CPP_DIRECT_OUTPUT_HASHES=$cpp_direct_hashes"
    echo "WORKER_OUTPUT_HASHES=$worker_hashes"
    print_logs
    exit 1
fi

direct_stats=$(stats_line DIRECT "$direct_latencies")
cpp_direct_stats=$(stats_line CPP_DIRECT "$cpp_direct_latencies")
worker_stats=$(stats_line WORKER "$worker_latencies")
direct_avg=$(sed -n 's/.*avg_us=\([0-9][0-9]*\).*/\1/p' <<< "$direct_stats")
cpp_direct_avg=$(sed -n 's/.*avg_us=\([0-9][0-9]*\).*/\1/p' <<< "$cpp_direct_stats")
worker_avg=$(sed -n 's/.*avg_us=\([0-9][0-9]*\).*/\1/p' <<< "$worker_stats")

ratios=$(DIRECT_AVG="$direct_avg" CPP_DIRECT_AVG="$cpp_direct_avg" WORKER_AVG="$worker_avg" "$CONDA_ENV_PREFIX/bin/python" <<'PY'
import os
direct = int(os.environ["DIRECT_AVG"])
cpp_direct = int(os.environ["CPP_DIRECT_AVG"])
worker = int(os.environ["WORKER_AVG"])
print(
    f"CPP_OVER_PYTHON={cpp_direct / direct:.3f} "
    f"WORKER_OVER_CPP={worker / cpp_direct:.3f} "
    f"WORKER_OVER_PYTHON={worker / direct:.3f}"
)
PY
)

echo "RESPONSE_TIME_COMPARE iterations=$ITERATIONS warmup=$WARMUP run_order=$RUN_ORDER"
echo "CPU_AFFINITY config=$(default_config_path) pytorch_worker=[$(pytorch_worker_cpu_list)]"
echo "SCHED_FIFO=$SCHED_FIFO"
echo "DIRECT_SCHED_FIFO_PRIORITY=$DIRECT_SCHED_FIFO_PRIORITY"
direct_load_us=$(sed -n 's/^DIRECT_LOAD_US=//p' "$DIRECT_LOG")
direct_threads=$(sed -n 's/^DIRECT_THREADS //p' "$DIRECT_LOG")
cpp_direct_load_us=$(sed -n 's/^CPP_DIRECT_LOAD_US=//p' "$CPP_DIRECT_LOG")
cpp_direct_threads=$(sed -n 's/^CPP_DIRECT_THREADS //p' "$CPP_DIRECT_LOG")
echo "DIRECT_LOAD_US=$direct_load_us"
echo "CPP_DIRECT_LOAD_US=$cpp_direct_load_us"
echo "PYTORCH_THREADS python=[$direct_threads] cpp=[$cpp_direct_threads] worker=[intra=$PYTORCH_NUM_THREADS interop=$PYTORCH_INTEROP_THREADS]"
echo "$ratios"
print_compact_summary
echo "OUTPUT_HASH_MATCH=PASS"

if [[ "$VERBOSE" == "1" ]]; then
    echo "DIRECT_LATENCIES_US=$direct_latencies"
    echo "DIRECT_SG1_US=$direct_sg1_latencies"
    echo "DIRECT_SG2_US=$direct_sg2_latencies"
    echo "DIRECT_SG3_US=$direct_sg3_latencies"
    echo "CPP_DIRECT_LATENCIES_US=$cpp_direct_latencies"
    echo "CPP_DIRECT_SG1_US=$cpp_direct_sg1_latencies"
    echo "CPP_DIRECT_SG2_US=$cpp_direct_sg2_latencies"
    echo "CPP_DIRECT_SG3_US=$cpp_direct_sg3_latencies"
    echo "CPP_DIRECT_OVERHEAD_US=$cpp_direct_overhead_latencies"
    echo "WORKER_LATENCIES_US=$worker_latencies"
    echo "WORKER_SG1_US=$worker_sg1_latencies"
    echo "WORKER_SG2_US=$worker_sg2_latencies"
    echo "WORKER_SG3_US=$worker_sg3_latencies"
    echo "WORKER_FORWARD_US=$worker_forward_latencies"
    echo "WORKER_FORWARD_SG1_US=$worker_forward_sg1_latencies"
    echo "WORKER_FORWARD_SG2_US=$worker_forward_sg2_latencies"
    echo "WORKER_FORWARD_SG3_US=$worker_forward_sg3_latencies"
    echo "WORKER_OVERHEAD_US=$worker_overhead_latencies"
    echo "WORKER_INSIDE_SG_NONFORWARD_US=$worker_inside_sg_nonforward_latencies"
    echo "WORKER_OUTSIDE_SG_WINDOW_US=$worker_outside_sg_window_latencies"
    echo "DIRECT_OUTPUT_HASHES=$direct_hashes"
    echo "CPP_DIRECT_OUTPUT_HASHES=$cpp_direct_hashes"
    echo "WORKER_OUTPUT_HASHES=$worker_hashes"
    print_logs
fi
