#!/usr/bin/env python3
"""
Launch multi-GPU CAFE object extraction in one command.

This launcher starts one extraction worker per GPU id, sets CUDA_VISIBLE_DEVICES
for each worker, injects --world_size/--rank automatically, and optionally merges
shards when all workers succeed.

Example:
  python scripts/launch_multi_gpu_extract.py --devices 0,1,2,3 \
    --backend hf \
    --hf_model_id /path/to/grounding-dino-base-snapshot \
    --hf_dtype auto \
    --hf_pack_batch \
    --hf_compile \
    --frame_batch_size 4 \
    --data_root /path/to/cafe \
    --person_tracks_pkl /path/to/cafe/gt_tracks.pkl \
    --output_pkl /path/to/cafe/object_tracks_gdino_swinb.pkl \
    --output_meta /path/to/cafe/object_tracks_gdino_swinb_meta.json \
    --overwrite
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, List, Sequence, Tuple


def parse_devices(devices_arg: str) -> List[str]:
    devices: List[str] = []
    seen = set()
    for part in devices_arg.split(","):
        dev = part.strip()
        if not dev:
            continue
        if dev in seen:
            continue
        seen.add(dev)
        devices.append(dev)
    if not devices:
        raise ValueError("--devices is empty. Example: --devices 0,1,2,3")
    return devices


def normalize_extract_args(raw_args: Sequence[str]) -> List[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    return args


def ensure_no_conflicting_args(extract_args: Sequence[str]) -> None:
    forbidden = {"--world_size", "--rank", "--merge_shards"}
    bad = [a for a in extract_args if a in forbidden]
    if bad:
        joined = ", ".join(sorted(set(bad)))
        raise ValueError(
            f"Do not pass {joined} in extractor args; launcher injects them automatically."
        )


def build_worker_cmd(
    python_bin: str,
    extract_script: Path,
    extract_args: Sequence[str],
    world_size: int,
    rank: int,
) -> List[str]:
    return [
        python_bin,
        str(extract_script),
        *extract_args,
        "--device",
        "cuda",
        "--world_size",
        str(world_size),
        "--rank",
        str(rank),
    ]


def build_merge_cmd(
    python_bin: str,
    extract_script: Path,
    extract_args: Sequence[str],
    world_size: int,
) -> List[str]:
    return [
        python_bin,
        str(extract_script),
        *extract_args,
        "--merge_shards",
        "--world_size",
        str(world_size),
        "--rank",
        "0",
    ]


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def run_merge(
    cmd: Sequence[str],
    logs_dir: Path,
    dry_run: bool,
) -> int:
    print("\n[merge] command:")
    print("  " + format_cmd(cmd))
    if dry_run:
        return 0
    merge_log = logs_dir / "merge.log"
    with open(merge_log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        print(f"[merge] failed. See log: {merge_log}")
    else:
        print(f"[merge] done. Log: {merge_log}")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch multi-GPU extraction workers and merge shard outputs."
    )
    parser.add_argument(
        "--devices",
        type=str,
        required=True,
        help="Comma-separated GPU ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run extraction workers.",
    )
    parser.add_argument(
        "--extract_script",
        type=str,
        default="scripts/extract_object_proposals_gdino_swinb.py",
        help="Path to extraction script.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="",
        help="Directory to write worker logs. Default: scripts/multi_gpu_logs/<timestamp>",
    )
    parser.add_argument(
        "--stagger_sec",
        type=float,
        default=3.0,
        help="Delay seconds between launching workers.",
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="Skip automatic merge after all workers finish.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only without executing.",
    )
    # Use parse_known_args so users can append extractor args directly without '--'.
    args, unknown = parser.parse_known_args()

    devices = parse_devices(args.devices)
    world_size = len(devices)
    extract_args = normalize_extract_args(unknown)
    ensure_no_conflicting_args(extract_args)

    extract_script = Path(args.extract_script).resolve()
    if not extract_script.exists():
        raise FileNotFoundError(f"extract_script not found: {extract_script}")

    if args.logs_dir:
        logs_dir = Path(args.logs_dir).resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        logs_dir = (Path("scripts") / "multi_gpu_logs" / ts).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Launcher configuration:")
    print(f"  devices      : {devices}")
    print(f"  world_size   : {world_size}")
    print(f"  python_bin   : {args.python_bin}")
    print(f"  extract_script: {extract_script}")
    print(f"  logs_dir     : {logs_dir}")
    print(f"  no_merge     : {args.no_merge}")
    print(f"  dry_run      : {args.dry_run}")

    workers: List[Tuple[int, str, List[str], Path, subprocess.Popen, IO[str]]] = []
    failed = False
    fail_msg = ""
    try:
        for rank, dev in enumerate(devices):
            cmd = build_worker_cmd(
                python_bin=args.python_bin,
                extract_script=extract_script,
                extract_args=extract_args,
                world_size=world_size,
                rank=rank,
            )
            log_path = logs_dir / f"rank{rank:02d}_gpu{dev}.log"

            print(f"\n[launch] rank={rank} gpu={dev}")
            print("  " + format_cmd(cmd))
            print(f"  log: {log_path}")

            if args.dry_run:
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = dev
            log_f = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            workers.append((rank, dev, cmd, log_path, proc, log_f))

            if args.stagger_sec > 0:
                time.sleep(args.stagger_sec)

        if not args.dry_run:
            remaining = {w[0]: w for w in workers}
            while remaining:
                done_ranks = []
                for rank, item in list(remaining.items()):
                    _, dev, _cmd, log_path, proc, _log_f = item
                    ret = proc.poll()
                    if ret is None:
                        continue
                    done_ranks.append(rank)
                    if ret == 0:
                        print(f"[done] rank={rank} gpu={dev} exit=0")
                    else:
                        failed = True
                        fail_msg = (
                            f"[fail] rank={rank} gpu={dev} exit={ret}. "
                            f"See log: {log_path}"
                        )
                        print(fail_msg)
                for rank in done_ranks:
                    remaining.pop(rank, None)

                if failed:
                    for _rank, dev, _cmd, log_path, proc, _log_f in remaining.values():
                        if proc.poll() is None:
                            proc.terminate()
                            print(
                                f"[terminate] rank={_rank} gpu={dev} due to previous failure. "
                                f"log: {log_path}"
                            )
                    break
                time.sleep(2.0)

            # Ensure child processes are reaped.
            for _rank, _dev, _cmd, _log_path, proc, _log_f in workers:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                try:
                    _log_f.close()
                except Exception:
                    pass

        if failed:
            print("\nLauncher failed.")
            if fail_msg:
                print(fail_msg)
            return 1

        if args.no_merge:
            print("\nAll workers finished. Merge skipped (--no_merge).")
            print(f"Logs: {logs_dir}")
            return 0

        merge_cmd = build_merge_cmd(
            python_bin=args.python_bin,
            extract_script=extract_script,
            extract_args=extract_args,
            world_size=world_size,
        )
        ret = run_merge(merge_cmd, logs_dir=logs_dir, dry_run=args.dry_run)
        if ret != 0:
            return ret

        print("\nAll workers finished and merge succeeded.")
        print(f"Logs: {logs_dir}")
        return 0

    finally:
        # Best effort close log files held by subprocess wrappers if any survive.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
