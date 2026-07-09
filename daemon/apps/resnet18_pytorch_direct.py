#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import time


RESNET18_INPUT_FLOATS = 1 * 3 * 224 * 224
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def cpu_list_to_string(cpu_ids: list[int]) -> str:
    if not cpu_ids:
        return "-"
    return ",".join(str(cpu_id) for cpu_id in cpu_ids)


def default_config_path() -> str:
    env_path = os.environ.get("EXAM_CONFIG_PATH")
    if env_path:
        return env_path

    for candidate in ("config/examl.yaml", "config/exam.yaml"):
        if os.path.exists(candidate):
            return candidate
    return ""


def parse_inline_cpu_list(value: str) -> list[int]:
    open_index = value.find("[")
    close_index = value.find("]", open_index + 1)
    if open_index < 0 or close_index < 0:
        return []

    cpu_ids = []
    for token in value[open_index + 1 : close_index].split(","):
        token = token.strip()
        if token:
            cpu_ids.append(int(token))
    return cpu_ids


def load_cpu_affinity_config() -> tuple[str, dict[str, list[int]]]:
    config_path = default_config_path()
    if not config_path:
        return "", {}

    cpu_affinity: dict[str, list[int]] = {}
    in_cpu_affinity = False
    cpu_affinity_indent = 0
    with open(config_path, "r", encoding="utf-8") as config_file:
        for raw_line in config_file:
            without_comment = raw_line.split("#", 1)[0].rstrip()
            stripped = without_comment.strip()
            if not stripped:
                continue

            indent = len(without_comment) - len(without_comment.lstrip(" "))
            if stripped == "cpu_affinity:":
                in_cpu_affinity = True
                cpu_affinity_indent = indent
                continue

            if in_cpu_affinity and indent <= cpu_affinity_indent:
                in_cpu_affinity = False
            if not in_cpu_affinity or ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            cpu_affinity[key.strip()] = parse_inline_cpu_list(value)

    return config_path, cpu_affinity


def apply_pytorch_worker_cpu_affinity() -> tuple[str, list[int]]:
    config_path, cpu_affinity = load_cpu_affinity_config()
    cpu_ids = cpu_affinity.get("pytorch_worker") or cpu_affinity.get("pytorch") or []
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpu_ids))
        except OSError as exc:
            print(
                "DIRECT_CPU_AFFINITY_WARNING "
                f"cpus=[{cpu_list_to_string(cpu_ids)}] error={exc}"
            )
    return config_path, cpu_ids


def apply_realtime_scheduler() -> None:
    disabled = os.environ.get("EXAM_DISABLE_SCHED_FIFO", "")
    if disabled and disabled not in ("0", "false", "False", "off", "OFF"):
        priority = positive_int_env("EXAM_DIRECT_SCHED_FIFO_PRIORITY", 0)
        if priority > 0:
            print(
                "DIRECT_SCHED_FIFO "
                f"priority={priority} result=skipped_by_EXAM_DISABLE_SCHED_FIFO"
            )
        return

    priority = positive_int_env("EXAM_DIRECT_SCHED_FIFO_PRIORITY", 0)
    if priority <= 0:
        return

    if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_FIFO"):
        print(
            "DIRECT_SCHED_FIFO_WARNING "
            f"priority={priority} error=unsupported_platform"
        )
        return

    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        print(f"DIRECT_SCHED_FIFO priority={priority} result=applied")
    except OSError as exc:
        print(
            "DIRECT_SCHED_FIFO_WARNING "
            f"priority={priority} error={exc}"
        )


def configure_pytorch_runtime() -> None:
    torch.set_num_threads(positive_int_env("PYTORCH_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_interop_threads(positive_int_env("PYTORCH_INTEROP_THREADS", 1))

    torch._C._set_graph_executor_optimize(True)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_num_profiled_runs(1)


def fnv1a64(data: bytes) -> str:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def make_input(input_id: int) -> torch.Tensor:
    state = (0x9E3779B9 ^ ((input_id * 0x85EBCA6B) & 0xFFFFFFFF)) & 0xFFFFFFFF
    values = np.empty(RESNET18_INPUT_FLOATS, dtype=np.float32)
    for index in range(RESNET18_INPUT_FLOATS):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values[index] = (
            np.float32((state >> 8) & 0xFFFF)
            / np.float32(65535.0)
            * np.float32(2.0)
            - np.float32(1.0)
        )
    return torch.from_numpy(values.reshape(1, 3, 224, 224))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_id", type=int, nargs="?", default=300)
    parser.add_argument("ts_dir", nargs="?", default="artifacts/resnet18/pytorch_ts")
    parser.add_argument("warmup", type=int, nargs="?", default=5)
    parser.add_argument("iterations", type=int, nargs="?", default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path, direct_cpu_ids = apply_pytorch_worker_cpu_affinity()
    apply_realtime_scheduler()

    global np
    global torch
    import numpy as np
    import torch

    print(
        "DIRECT_CPU_AFFINITY "
        f"config={config_path or '-'} "
        f"using=[{cpu_list_to_string(direct_cpu_ids)}]"
    )
    configure_pytorch_runtime()

    total = args.warmup + args.iterations
    load_start = time.perf_counter_ns()
    modules = [
        torch.jit.load(os.path.join(args.ts_dir, "sg1.pt"), map_location="cpu").eval(),
        torch.jit.load(os.path.join(args.ts_dir, "sg2.pt"), map_location="cpu").eval(),
        torch.jit.load(os.path.join(args.ts_dir, "sg3.pt"), map_location="cpu").eval(),
    ]
    load_us = (time.perf_counter_ns() - load_start) // 1000

    latencies = []
    sg_latencies = [[] for _ in modules]
    hashes = []
    with torch.inference_mode():
        for iteration in range(total):
            current_input_id = args.input_id + iteration
            tensor = make_input(current_input_id)
            start = time.perf_counter_ns()
            current_sg_latencies = []
            for module in modules:
                sg_start = time.perf_counter_ns()
                tensor = module(tensor)
                current_sg_latencies.append((time.perf_counter_ns() - sg_start) // 1000)
            latency_us = (time.perf_counter_ns() - start) // 1000
            output_hash = fnv1a64(tensor.detach().contiguous().cpu().numpy().tobytes())

            latencies.append(latency_us)
            for index, sg_latency_us in enumerate(current_sg_latencies):
                sg_latencies[index].append(sg_latency_us)
            hashes.append(output_hash)

            print(
                "DIRECT_RESULT "
                f"input_id={current_input_id} "
                f"iteration={iteration} "
                f"output_hash={output_hash} "
                f"latency_us={latency_us} "
                f"total_us={latency_us} "
                f"sg1_us={current_sg_latencies[0]} "
                f"sg2_us={current_sg_latencies[1]} "
                f"sg3_us={current_sg_latencies[2]}"
            )

    measured = latencies[args.warmup :]
    measured_hashes = hashes[args.warmup :]
    print(f"DIRECT_LOAD_US={load_us}")
    print(
        "DIRECT_THREADS "
        f"intra={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()}"
    )
    print("DIRECT_JIT graph_executor_optimize=on executor_mode=off profiling_mode=off num_profiled_runs=1")
    print("DIRECT_LATENCIES_US=" + " ".join(str(value) for value in measured))
    for index, values in enumerate(sg_latencies, start=1):
        measured_sg = values[args.warmup :]
        print(f"DIRECT_SG{index}_US=" + " ".join(str(value) for value in measured_sg))
    print("DIRECT_OUTPUT_HASHES=" + " ".join(measured_hashes))
    print(
        "DIRECT_STATS "
        f"count={len(measured)} "
        f"min_us={min(measured)} "
        f"avg_us={int(statistics.mean(measured))} "
        f"p50_us={int(statistics.median(measured))} "
        f"max_us={max(measured)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
