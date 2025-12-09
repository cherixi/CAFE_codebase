#!/usr/bin/env python3
"""
Auto GPU runner: monitor idle GPUs and launch a training command once enough GPUs
stay idle for a configured duration. Supports Linux and Windows (default linux).

Usage (examples):
  python auto_gpu_runner.py --train-cmd 'python train.py --batch 12 --num_frame 8'
  python auto_gpu_runner.py --train-cmd 'python train.py --batch 12 --num_frame 8 --device "4,5,6,7"' --needed-count 2

Key behavior:
- Monitors specified GPU indices (default 0-7).
- Idle if memory.used <= MEM_THRESHOLD and power.draw <= POWER_THRESHOLD and pstate == P8.
- Needs consecutive idle duration before launching.
- Automatically injects or replaces --device "g0,g1,..." in the provided training command.
- Single-line status updates (no log flooding); prints a detailed summary on launch.
"""

import argparse
import re
import shlex
import subprocess
import sys
import time
from typing import List, Optional, Tuple

# ===== Default configuration =====
DEFAULT_TARGET_GPUS = "0,1,2,3,4,5,6,7"
DEFAULT_NEEDED_COUNT = 4
DEFAULT_CHECK_INTERVAL_IDLE = 300  # seconds
DEFAULT_CHECK_INTERVAL_CONFIRM = 30  # seconds
DEFAULT_STABLE_DURATION = 180  # seconds of continuous satisfaction
# Defaults tuned for RTX 3090; adjust if needed
DEFAULT_MEM_THRESHOLD_MB = 1500
DEFAULT_POWER_THRESHOLD_W = 40
DEFAULT_PLATFORM = "linux"  # or "windows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor GPU idleness and launch training when ready (auto --device)."
    )
    parser.add_argument(
        "--train-cmd",
        required=True,
        help="Training command to run (will inject/replace --device).",
    )
    parser.add_argument(
        "--target-gpus",
        default=DEFAULT_TARGET_GPUS,
        help=f"Comma-separated GPU indices to monitor (default: {DEFAULT_TARGET_GPUS}).",
    )
    parser.add_argument(
        "--needed-count",
        type=int,
        default=DEFAULT_NEEDED_COUNT,
        help=f"Number of GPUs required (default: {DEFAULT_NEEDED_COUNT}).",
    )
    parser.add_argument(
        "--check-interval-idle",
        type=float,
        default=DEFAULT_CHECK_INTERVAL_IDLE,
        help=f"Polling interval when not yet satisfied (seconds, default: {DEFAULT_CHECK_INTERVAL_IDLE}).",
    )
    parser.add_argument(
        "--check-interval-confirm",
        type=float,
        default=DEFAULT_CHECK_INTERVAL_CONFIRM,
        help=f"Polling interval during confirmation (seconds, default: {DEFAULT_CHECK_INTERVAL_CONFIRM}).",
    )
    parser.add_argument(
        "--stable-duration",
        type=float,
        default=DEFAULT_STABLE_DURATION,
        help=f"Required continuous duration before launch (seconds, default: {DEFAULT_STABLE_DURATION}).",
    )
    parser.add_argument(
        "--mem-threshold",
        type=int,
        default=DEFAULT_MEM_THRESHOLD_MB,
        help=f"Idle memory threshold MB (default: {DEFAULT_MEM_THRESHOLD_MB}).",
    )
    parser.add_argument(
        "--power-threshold",
        type=int,
        default=DEFAULT_POWER_THRESHOLD_W,
        help=f"Idle power threshold W (default: {DEFAULT_POWER_THRESHOLD_W}).",
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default=DEFAULT_PLATFORM,
        help="Force platform-specific behavior; default linux.",
    )
    return parser.parse_args()


def parse_gpu_list(gpu_str: str) -> List[int]:
    parts = [p.strip() for p in gpu_str.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def run_nvidia_smi(platform: str) -> str:
    # Same query works on Linux/Windows; no sandboxing allowed here.
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used,power.draw,pstate", "--format=csv,noheader,nounits"]
    # Use shell on Windows for better PATH resolution; otherwise direct exec.
    if platform == "windows":
        completed = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
    else:
        completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return completed.stdout


def parse_gpu_status(raw: str) -> List[Tuple[int, int, float, str]]:
    """
    Returns list of tuples: (index, memory_used_mb, power_w, pstate)
    """
    results: List[Tuple[int, int, float, str]] = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            mem = int(float(parts[1]))
            power = float(parts[2])
            pstate = parts[3]
            results.append((idx, mem, power, pstate))
        except ValueError:
            continue
    return results


def find_idle_gpus(
    statuses: List[Tuple[int, int, float, str]],
    target_gpus: List[int],
    mem_threshold: int,
    power_threshold: int,
) -> List[int]:
    idle = []
    for idx, mem, power, pstate in statuses:
        if idx not in target_gpus:
            continue
        if mem <= mem_threshold and power <= power_threshold and pstate.upper() == "P8":
            idle.append(idx)
    return sorted(idle)


def inject_or_replace_device(train_cmd: str, device_str: str) -> str:
    """
    Replace existing --device value or append one.
    Handles forms:
      --device=X
      --device X
      --device "0,1"
    """
    device_pattern = re.compile(r"--device(?:\s+|=)(\"[^\"]*\"|'[^']*'|[^\\s]+)")
    replacement = f'--device "{device_str}"'
    if device_pattern.search(train_cmd):
        return device_pattern.sub(replacement, train_cmd, count=1)
    # If not found, append at the end (respect spacing)
    if train_cmd.endswith(" "):
        return f"{train_cmd}{replacement}"
    return f"{train_cmd} {replacement}"


def print_single_line(msg: str, last_len: int) -> int:
    # Overwrite previous line using carriage return and padding.
    out = "\r" + msg
    pad = max(0, last_len - len(msg))
    sys.stdout.write(out + (" " * pad))
    sys.stdout.flush()
    return len(msg)


def main() -> None:
    args = parse_args()
    target_gpus = parse_gpu_list(args.target_gpus)

    if args.needed_count <= 0:
        sys.exit("needed-count must be positive.")
    if args.needed_count > len(target_gpus):
        sys.exit("needed-count cannot exceed number of target-gpus.")

    # Precompute preview devices from the start of target list.
    preview_devices = ",".join(str(g) for g in target_gpus[: args.needed_count])
    preview_cmd = inject_or_replace_device(args.train_cmd, preview_devices)

    print("=== Auto GPU Runner ===")
    print(f"Platform: {args.platform}")
    print(f"Target GPUs: {target_gpus} (need {args.needed_count})")
    print(f"Intervals: idle={args.check_interval_idle}s, confirm={args.check_interval_confirm}s, stable={args.stable_duration}s")
    print(f"Idle thresholds: mem<={args.mem_threshold}MB, power<={args.power_threshold}W, pstate==P8")
    print(f"Training command preview (using first {args.needed_count} GPUs):")
    print(f"  {preview_cmd}")
    print("Monitoring... (Ctrl+C to exit)")

    stable_start: Optional[float] = None
    last_len = 0

    try:
        while True:
            now = time.time()
            try:
                raw = run_nvidia_smi(args.platform)
            except Exception as exc:
                sys.stdout.write("\n")
                sys.exit(f"Failed to query nvidia-smi: {exc}")

            statuses = parse_gpu_status(raw)
            idle = find_idle_gpus(statuses, target_gpus, args.mem_threshold, args.power_threshold)
            idle_count = len(idle)

            if idle_count >= args.needed_count:
                if stable_start is None:
                    stable_start = now
                stable_elapsed = now - stable_start
            else:
                stable_start = None
                stable_elapsed = 0.0

            # Single-line status
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            status_msg = (
                f"[{ts}] idle {idle_count}/{args.needed_count} GPUs "
                f"(idle: {','.join(map(str, idle)) or 'none'}) "
                f"stable_for={int(stable_elapsed)}s"
            )
            last_len = print_single_line(status_msg, last_len)

            # Check launch condition
            if stable_start is not None and (now - stable_start) >= args.stable_duration:
                chosen = idle[: args.needed_count]
                device_str = ",".join(str(g) for g in chosen)
                final_cmd = inject_or_replace_device(args.train_cmd, device_str)
                sys.stdout.write("\n")
                print("=== Launching training ===")
                print(f"Time: {ts}")
                print(f"Chosen GPUs: {chosen}")
                print(f"Command: {final_cmd}")
                # Use shell for compatibility with complex commands/quotes.
                subprocess.run(final_cmd, shell=True)
                return

            # Sleep according to phase
            interval = args.check_interval_confirm if stable_start is not None else args.check_interval_idle
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("Interrupted, exiting.")


if __name__ == "__main__":
    main()
