#!/usr/bin/env python
"""Launch one controlled CAFE experiment and stop clearly weak runs early."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


KIM_TARGET = {
    "group_mAP_1.0": 10.85,
    "group_mAP_0.5": 30.90,
    "outlier_mIoU": 63.84,
}


VARIANTS = {
    "baseline_nols": {
        "hoi_mode": "none",
        "mae_fusion": "static_pool",
        "label_smoothing": 0.0,
    },
    "bias_pool": {
        "hoi_mode": "bias",
        "mae_fusion": "static_pool",
        "label_smoothing": 0.05,
    },
    "bias_pool_nols": {
        "hoi_mode": "bias",
        "mae_fusion": "static_pool",
        "label_smoothing": 0.0,
    },
    "hardmask_pool": {
        "hoi_mode": "hard_mask",
        "mae_fusion": "static_pool",
        "label_smoothing": 0.05,
        "hoi_hard_thresh": 0.2,
    },
    "bias_add": {
        "hoi_mode": "bias",
        "mae_fusion": "static_add",
        "label_smoothing": 0.05,
    },
    "bias_nome": {
        "hoi_mode": "bias",
        "mae_fusion": "none",
        "label_smoothing": 0.05,
        "no_mae": True,
    },
    "actor_motion_bias_pool": {
        "hoi_mode": "bias",
        "mae_fusion": "static_pool",
        "label_smoothing": 0.05,
        "use_sdtp": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--devices", required=True, help="four comma-separated GPU ids")
    parser.add_argument("--tag", default="")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_command(args, tag):
    cfg = VARIANTS[args.variant]
    command = [
        sys.executable,
        "train.py",
        "--split", "place",
        "--data_path", "/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/",
        "--experiment_tag", tag,
        "--backbone", "dinov2_vitb14",
        "--unfreeze_blocks", "2",
        "--frozen_batch_norm",
        "--batch", "16",
        "--num_frame", "8",
        "--test_batch", "52",
        "--device", args.devices,
        "--videomae_feats_path", "./videomae_features_giant",
        "--mae_fusion", cfg["mae_fusion"],
        "--hoi_mode", cfg["hoi_mode"],
        "--no_olic",
        "--no_pairwise_refiner",
        "--temporal_agg_mode", "learned_pool",
        "--label_smoothing", str(cfg["label_smoothing"]),
        "--skip_test_epochs", "0",
        "--test_freq", "1",
        "--random_seed", "1",
        "--no_sdtp",
    ]
    if cfg.get("no_mae"):
        command.append("--no_mae")
    if "hoi_hard_thresh" in cfg:
        command.extend(["--hoi_hard_thresh", str(cfg["hoi_hard_thresh"])])
    if cfg.get("use_sdtp"):
        command[-1] = "--use_sdtp"
        command.extend([
            "--sdtp_scope", "actor",
            "--sdtp_dynamic_scale_init", "0.02",
            "--sdtp_dynamic_scale_max", "0.1",
        ])
    return command


def read_summary(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def best_metrics(summary):
    history = summary.get("history", {}).get("val", [])
    if not history:
        return 0, {key: float("-inf") for key in KIM_TARGET}
    latest_epoch = max(int(row["epoch"]) for row in history)
    best = {
        key: max(float(row.get(key, float("-inf"))) for row in history)
        for key in KIM_TARGET
    }
    return latest_epoch, best


def stop_reason(epoch, best):
    # These gates are deliberately lenient relative to successful historical
    # curves. They catch collapse, not ordinary metric noise.
    if epoch >= 6 and best["group_mAP_1.0"] < 2.0 and best["group_mAP_0.5"] < 10.0:
        return "collapse_at_epoch_6"
    if epoch >= 8 and best["group_mAP_1.0"] < 6.0:
        return "below_viable_map1_at_epoch_8"
    if epoch >= 12 and (
        best["group_mAP_1.0"] < 8.5 or best["group_mAP_0.5"] < 25.0
    ):
        return "below_historical_curve_at_epoch_12"
    if epoch >= 18 and (
        best["group_mAP_1.0"] < KIM_TARGET["group_mAP_1.0"]
        or best["group_mAP_0.5"] < KIM_TARGET["group_mAP_0.5"] - 2.0
        or best["outlier_mIoU"] < KIM_TARGET["outlier_mIoU"] - 2.0
    ):
        return "unlikely_to_beat_kim_at_epoch_18"
    return None


def terminate_group(process, reason):
    if process.poll() is not None:
        return
    print(f"EARLY_STOP reason={reason} pid={process.pid}", flush=True)
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def write_status(path, **payload):
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    args = parse_args()
    repo = args.repo.resolve()
    host = socket.gethostname().split(".")[0]
    raw_tag = args.tag or f"agile-{args.variant}-{host}"
    tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in raw_tag).strip("-_")
    command = build_command(args, tag)
    if args.dry_run:
        print(json.dumps({"tag": tag, "command": command}, indent=2))
        return

    run_root = repo / "agile_runs"
    run_root.mkdir(exist_ok=True)
    wrapper_log = run_root / f"{tag}.log"
    status_path = run_root / f"{tag}.status.json"
    result_root = repo / "result"
    started_at = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with wrapper_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def handle_signal(signum, _frame):
        terminate_group(process, f"supervisor_signal_{signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    write_status(
        status_path,
        state="starting",
        variant=args.variant,
        tag=tag,
        devices=args.devices,
        train_pid=process.pid,
        command=command,
    )

    result_dir = None
    last_epoch = -1
    qualified = False
    while process.poll() is None:
        if result_dir is None:
            candidates = [
                path for path in result_root.iterdir()
                if path.is_dir()
                and path.name.endswith(f"_[{tag}]")
                and path.stat().st_mtime >= started_at - 5
            ]
            if candidates:
                result_dir = max(candidates, key=lambda path: path.stat().st_mtime)

        if result_dir is not None:
            summary = read_summary(result_dir / "summary.json")
            if summary:
                epoch, best = best_metrics(summary)
                if epoch != last_epoch:
                    last_epoch = epoch
                    qualified = all(best[key] >= value for key, value in KIM_TARGET.items())
                    state = "qualified" if qualified else "running"
                    print(
                        f"MONITOR epoch={epoch} best={best} state={state}",
                        flush=True,
                    )
                    write_status(
                        status_path,
                        state=state,
                        variant=args.variant,
                        tag=tag,
                        devices=args.devices,
                        train_pid=process.pid,
                        result_dir=str(result_dir),
                        epoch=epoch,
                        best=best,
                    )
                    reason = stop_reason(epoch, best)
                    if reason:
                        terminate_group(process, reason)
                        write_status(
                            status_path,
                            state="early_stopped",
                            reason=reason,
                            variant=args.variant,
                            tag=tag,
                            devices=args.devices,
                            train_pid=process.pid,
                            result_dir=str(result_dir),
                            epoch=epoch,
                            best=best,
                        )
                        return
        time.sleep(args.poll_seconds)

    return_code = process.returncode
    write_status(
        status_path,
        state="completed" if return_code == 0 else "failed",
        return_code=return_code,
        variant=args.variant,
        tag=tag,
        devices=args.devices,
        train_pid=process.pid,
        result_dir=str(result_dir) if result_dir else None,
        epoch=last_epoch,
        qualified=qualified,
    )
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
