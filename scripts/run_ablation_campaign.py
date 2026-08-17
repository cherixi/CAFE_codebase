#!/usr/bin/env python
"""Chain supervised ablation stages without exceeding a fixed GPU slot."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_agile_experiment import VARIANTS


KIM_TARGET = {
    "group_mAP_1.0": 10.85,
    "group_mAP_0.5": 30.90,
    "outlier_mIoU": 63.84,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--initial_wait",
        action="append",
        default=[],
        help="existing supervisor as PID:status_json",
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--plateau_min_epoch", type=int, default=12)
    parser.add_argument("--plateau_patience", type=int, default=5)
    parser.add_argument("--gpu_cleanup_seconds", type=int, default=30)
    parser.add_argument("--gpu_cleanup_timeout", type=int, default=1800)
    parser.add_argument("--max_local_gpus", type=int, default=8)
    parser.add_argument("--min_free_gb", type=float, default=50.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def atomic_write_json(path, payload):
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def free_space_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if cmdline_path.exists():
        try:
            cmdline = cmdline_path.read_bytes()
        except OSError:
            return False
        return b"run_agile_experiment.py" in cmdline
    return True


def parse_wait_spec(spec, repo):
    pid_text, status_text = spec.split(":", 1)
    status_path = Path(status_text)
    if not status_path.is_absolute():
        status_path = repo / status_path
    return {"pid": int(pid_text), "status_path": status_path}


def validation_state(status_path, min_epoch, patience):
    status = read_json(status_path)
    if not status:
        return None
    result_dir = status.get("result_dir")
    if not result_dir:
        return None
    summary = read_json(Path(result_dir) / "summary.json")
    if not summary:
        return None
    history = summary.get("history", {}).get("val", [])
    if not history:
        return None

    latest_epoch = max(int(row["epoch"]) for row in history)
    qualified_rows = [
        row for row in history
        if all(float(row.get(key, float("-inf"))) >= target for key, target in KIM_TARGET.items())
    ]
    best_entries = summary.get("best", {})
    best_epochs = [
        int(entry.get("epoch", -1))
        for key, entry in best_entries.items()
        if key in KIM_TARGET and isinstance(entry, dict)
    ]
    last_best_epoch = max(best_epochs, default=-1)
    plateau = (
        bool(qualified_rows)
        and latest_epoch >= min_epoch
        and last_best_epoch >= 0
        and latest_epoch - last_best_epoch >= patience
    )
    return {
        "epoch": latest_epoch,
        "last_best_epoch": last_best_epoch,
        "qualified": bool(qualified_rows),
        "qualifying_epochs": [int(row["epoch"]) for row in qualified_rows],
        "plateau": plateau,
        "best": {key: best_entries.get(key) for key in KIM_TARGET},
    }


def monitor_supervisors(items, args, state_path, stage_name):
    plateau_stop_sent = False
    while True:
        alive = [item for item in items if process_alive(item["pid"])]
        states_by_pid = {
            item["pid"]: validation_state(
                item["status_path"],
                args.plateau_min_epoch,
                args.plateau_patience,
            )
            for item in items
        }
        atomic_write_json(
            state_path,
            {
                "state": "monitoring",
                "stage": stage_name,
                "supervisors": [
                    {
                        "pid": item["pid"],
                        "status_path": str(item["status_path"]),
                        "alive": process_alive(item["pid"]),
                    }
                    for item in items
                ],
                "validation": states_by_pid,
                "free_space_gb": free_space_gb(args.repo),
            },
        )
        if not alive:
            qualified = all(
                states_by_pid[item["pid"]]
                and states_by_pid[item["pid"]]["qualified"]
                for item in items
            )
            return {
                "qualified": qualified,
                "validation": states_by_pid,
            }

        # Stop a matched stage together only when every still-running member
        # has both crossed Kim and exhausted the same conservative patience.
        alive_states = [states_by_pid[item["pid"]] for item in alive]
        if alive_states and all(state and state["plateau"] for state in alive_states):
            print(f"PLATEAU_STOP stage={stage_name} pids={[x['pid'] for x in alive]}", flush=True)
            for item in alive:
                try:
                    os.kill(item["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            plateau_stop_sent = True

        if plateau_stop_sent:
            time.sleep(min(args.poll_seconds, 30))
        else:
            time.sleep(args.poll_seconds)


def validate_stage(stage, max_local_gpus):
    if not stage.get("name"):
        raise ValueError("Every stage needs a non-empty name")
    experiments = stage.get("experiments", [])
    if not experiments:
        raise ValueError(f"Stage {stage['name']} has no experiments")
    used = []
    for experiment in experiments:
        missing = {"variant", "devices", "tag"} - set(experiment)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {experiment}")
        if experiment["variant"] not in VARIANTS:
            raise ValueError(f"Unknown variant {experiment['variant']}")
        devices = [int(value) for value in experiment["devices"].split(",")]
        if len(devices) != len(set(devices)):
            raise ValueError(f"Duplicate devices in {experiment}")
        used.extend(devices)
    if len(used) != len(set(used)):
        raise ValueError(f"Overlapping devices in stage {stage.get('name')}")
    if len(used) > max_local_gpus:
        raise ValueError(f"Stage uses {len(used)} GPUs, limit is {max_local_gpus}")
    return used


def gpu_compute_processes():
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid_to_index = {}
    for line in gpu_result.stdout.splitlines():
        index, gpu_uuid = (value.strip() for value in line.split(",", 1))
        uuid_to_index[gpu_uuid] = int(index)

    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = {index: [] for index in uuid_to_index.values()}
    for line in process_result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 1)]
        if len(fields) != 2 or fields[0] not in uuid_to_index:
            continue
        processes[uuid_to_index[fields[0]]].append(int(fields[1]))
    return processes


def wait_for_devices_idle(devices, args, state_path, stage_name):
    deadline = time.time() + args.gpu_cleanup_timeout
    while True:
        occupied = {
            device: pids
            for device, pids in gpu_compute_processes().items()
            if device in devices and pids
        }
        if not occupied:
            return
        atomic_write_json(
            state_path,
            {
                "state": "waiting_for_gpus",
                "stage": stage_name,
                "devices": devices,
                "occupied": occupied,
            },
        )
        if time.time() >= deadline:
            raise TimeoutError(f"GPUs remained occupied for {stage_name}: {occupied}")
        time.sleep(min(args.poll_seconds, 30))


def launch_stage(stage, repo, run_root):
    items = []
    for experiment in stage.get("experiments", []):
        tag = experiment["tag"]
        command = [
            sys.executable,
            "scripts/run_agile_experiment.py",
            "--variant", experiment["variant"],
            "--devices", experiment["devices"],
            "--tag", tag,
            "--random_seed", str(experiment.get("random_seed", 1)),
        ]
        supervisor_log = run_root / f"{tag}.supervisor.log"
        pid_path = run_root / f"{tag}.supervisor.pid"
        if pid_path.exists():
            try:
                old_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                old_pid = -1
            if old_pid > 0 and process_alive(old_pid):
                raise RuntimeError(f"Supervisor for {tag} is already running: {old_pid}")
        with supervisor_log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=repo,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(process.pid), encoding="utf-8")
        items.append({
            "pid": process.pid,
            "status_path": run_root / f"{tag}.status.json",
        })
        print(f"LAUNCHED stage={stage.get('name')} tag={tag} pid={process.pid}", flush=True)
    return items


def main():
    args = parse_args()
    repo = args.repo.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = read_json(plan_path.resolve())
    if not plan or not isinstance(plan.get("stages"), list):
        raise ValueError("Plan must contain a stages list")

    tags = []
    for stage in plan["stages"]:
        validate_stage(stage, args.max_local_gpus)
        tags.extend(experiment["tag"] for experiment in stage["experiments"])
    if len(tags) != len(set(tags)):
        raise ValueError("Experiment tags must be unique across the campaign")

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    run_root = repo / "agile_runs"
    run_root.mkdir(exist_ok=True)
    state_path = run_root / f"campaign-{args.campaign}.status.json"

    current = [parse_wait_spec(spec, repo) for spec in args.initial_wait]
    if current:
        outcome = monitor_supervisors(current, args, state_path, "initial")
        if not outcome["qualified"]:
            atomic_write_json(
                state_path,
                {
                    "state": "blocked_unqualified",
                    "stage": "initial",
                    "validation": outcome["validation"],
                },
            )
            return

    for stage in plan["stages"]:
        devices = validate_stage(stage, args.max_local_gpus)
        time.sleep(args.gpu_cleanup_seconds)
        available_gb = free_space_gb(repo)
        if available_gb < args.min_free_gb:
            atomic_write_json(
                state_path,
                {
                    "state": "blocked_low_disk",
                    "stage": stage["name"],
                    "free_space_gb": available_gb,
                    "min_free_gb": args.min_free_gb,
                },
            )
            return
        wait_for_devices_idle(devices, args, state_path, stage["name"])
        current = launch_stage(stage, repo, run_root)
        outcome = monitor_supervisors(current, args, state_path, stage["name"])
        if not outcome["qualified"]:
            atomic_write_json(
                state_path,
                {
                    "state": "blocked_unqualified",
                    "stage": stage["name"],
                    "validation": outcome["validation"],
                },
            )
            return

    atomic_write_json(state_path, {"state": "completed", "stage": None})


if __name__ == "__main__":
    main()
