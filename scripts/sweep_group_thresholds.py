import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import util.misc as utils
from test import build_parser as build_test_parser
from test import collate_fn
from util.test_eval_utils import (
    build_eval_metrics,
    build_eval_model_and_criterion,
    build_test_loader,
    collect_prediction_rows,
    create_metric_logger,
    evaluate_prediction_file,
    get_name_to_vid,
    resolve_eval_args,
    run_eval_batch,
    setup_eval_environment,
    update_metric_logger,
    write_prediction_rows,
)


METRIC_KEYS = ("group_mAP_1.0", "group_mAP_0.5", "outlier_mIoU")


def parse_thresholds(raw) -> List[float]:
    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(v) for v in raw)
    values = []
    for part in str(raw).replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def format_threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace("-", "m").replace(".", "p")


def load_saved_args(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    args_path = run_dir / "args.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return summary.get("args", {})
    if args_path.exists():
        with args_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_summary_best(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary.get("best", {})


def coerce_value(value, current):
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def build_eval_args(cli_args, checkpoint_path: Path):
    parser = build_test_parser()
    eval_args = parser.parse_args([])
    saved_args = load_saved_args(Path(cli_args.run_dir))

    for key, value in saved_args.items():
        if hasattr(eval_args, key):
            setattr(eval_args, key, coerce_value(value, getattr(eval_args, key)))

    eval_args.model_path = str(checkpoint_path)
    eval_args.result_path = str(cli_args.output_dir)
    eval_args.group_threshold = float(cli_args.thresholds[0])

    for key in (
        "device",
        "data_path",
        "tracks_source",
        "tracks_pkl_path",
        "object_tracks_pkl",
        "test_batch",
        "num_frame",
        "num_object_boxes",
        "pmr_anchor_source",
        "eval_image_width",
        "eval_image_height",
        "eval_num_workers",
        "eval_persistent_workers",
        "eval_prefetch_factor",
        "groundtruth",
        "labelmap",
    ):
        value = getattr(cli_args, key, None)
        if value is not None:
            setattr(eval_args, key, value)

    eval_args.groundtruth = ensure_file_obj(eval_args.groundtruth, "r")
    eval_args.labelmap = ensure_file_obj(eval_args.labelmap, "r")
    return resolve_eval_args(eval_args)


def ensure_file_obj(value, mode: str):
    if hasattr(value, "read"):
        try:
            value.seek(0)
        except Exception:
            pass
        return value
    return open(value, mode)


def checkpoint_metric_hint(checkpoint_name: str) -> Optional[str]:
    name = checkpoint_name.lower()
    if "map_1_0" in name or "map@1.0" in name:
        return "group_mAP_1.0"
    if "map_0_5" in name or "map@0.5" in name:
        return "group_mAP_0.5"
    if "outlier" in name or "miou" in name:
        return "outlier_mIoU"
    if "loss" in name:
        return "loss"
    return None


def original_value_for_checkpoint(best: Dict, checkpoint_name: str):
    key = checkpoint_metric_hint(checkpoint_name)
    if key is None or key not in best:
        return None, None
    item = best[key]
    if isinstance(item, dict):
        return key, item.get("value")
    return key, item


@torch.no_grad()
def run_checkpoint(checkpoint_path: Path, cli_args, thresholds: List[float], summary_best: Dict):
    eval_args = build_eval_args(cli_args, checkpoint_path)
    setup_eval_environment(eval_args)

    test_set, test_loader = build_test_loader(eval_args, collate_fn)
    model, criterion = build_eval_model_and_criterion(eval_args)
    model.eval()
    criterion.eval()

    metrics = build_eval_metrics(eval_args)
    metric_logger = create_metric_logger()
    name_to_vid = get_name_to_vid()
    rows = []

    total_batches = len(test_loader)
    print(f"  Processing {total_batches} batches for {checkpoint_path.name}...")
    for i, (images, targets, infos) in enumerate(test_loader):
        if i % max(1, total_batches // 10) == 0:
            print(f"    Progress: {i}/{total_batches} batches ({100 * i // max(1, total_batches)}%)")
        clean_boxes, outputs, loss_dict = run_eval_batch(model, criterion, images, targets, infos, eval_args)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        update_metric_logger(metric_logger, loss_dict_reduced, criterion.weight_dict, outputs, eval_args)
        rows.extend(collect_prediction_rows(clean_boxes, infos, outputs, eval_args, name_to_vid=name_to_vid))
    print(f"    Progress: {total_batches}/{total_batches} batches (100%)")

    metric_logger.synchronize_between_processes()
    logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    ckpt_out_dir = Path(cli_args.output_dir) / checkpoint_path.stem
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    original_key, original_value = original_value_for_checkpoint(summary_best, checkpoint_path.name)
    results = []
    for threshold in thresholds:
        pred_path = ckpt_out_dir / f"pred_group_threshold_{format_threshold_tag(threshold)}.txt"
        write_prediction_rows(rows, str(pred_path), threshold, eval_args)
        result = evaluate_prediction_file(metrics, str(pred_path))
        row = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "threshold": threshold,
            "original_metric": original_key or "",
            "original_value": original_value if original_value is not None else "",
            "loss": logger_stats.get("loss", ""),
            "group_class_error": logger_stats.get("group_class_error", ""),
        }
        for key in METRIC_KEYS:
            row[key] = float(result[key])
        if original_key in row and original_value is not None:
            row["delta_vs_original"] = float(row[original_key]) - float(original_value)
        else:
            row["delta_vs_original"] = ""
        results.append(row)

    del model
    torch.cuda.empty_cache()
    return results


def write_outputs(rows: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "threshold_sweep_results.json"
    csv_path = output_dir / "threshold_sweep_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "checkpoint",
        "threshold",
        "group_mAP_1.0",
        "group_mAP_0.5",
        "outlier_mIoU",
        "original_metric",
        "original_value",
        "delta_vs_original",
        "loss",
        "group_class_error",
        "checkpoint_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all best checkpoints in a run directory with multiple group_threshold values."
    )
    parser.add_argument("--run_dir", required=True, type=Path, help="Training result directory containing best*.pth.")
    parser.add_argument("--thresholds", nargs="+", default=["0.45,0.50,0.55,0.60,0.65,0.70"])
    parser.add_argument("--checkpoint_glob", default="best*.pth", help="Checkpoint glob under run_dir.")
    parser.add_argument("--output_dir", default=None, type=Path)

    parser.add_argument("--device", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--tracks_source", default=None, choices=["gt", "pred"])
    parser.add_argument("--tracks_pkl_path", default=None)
    parser.add_argument("--object_tracks_pkl", default=None)
    parser.add_argument("--test_batch", default=None, type=int)
    parser.add_argument("--num_frame", default=None, type=int)
    parser.add_argument("--num_object_boxes", default=None, type=int)
    parser.add_argument("--pmr_anchor_source", default=None, choices=["auto", "gdino", "yolo"])
    parser.add_argument("--eval_image_width", default=None, type=int)
    parser.add_argument("--eval_image_height", default=None, type=int)
    parser.add_argument("--eval_num_workers", default=0, type=int)
    parser.add_argument("--eval_persistent_workers", default=None, action="store_true")
    parser.add_argument("--eval_prefetch_factor", default=None, type=int)
    parser.add_argument("--groundtruth", default=None)
    parser.add_argument("--labelmap", default=None)
    args = parser.parse_args()
    args.thresholds = parse_thresholds(args.thresholds)

    args.run_dir = args.run_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.run_dir / "threshold_sweep"
    else:
        args.output_dir = args.output_dir.resolve()

    checkpoints = sorted(args.run_dir.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob!r} under {args.run_dir}")

    summary_best = load_summary_best(args.run_dir)
    all_rows = []
    for checkpoint in checkpoints:
        print(f"[sweep] Evaluating {checkpoint.name} with thresholds {args.thresholds}")
        all_rows.extend(run_checkpoint(checkpoint, args, args.thresholds, summary_best))

    json_path, csv_path = write_outputs(all_rows, args.output_dir)

    print("\ncheckpoint, threshold, mAP@1.0, mAP@0.5, outlier_mIoU, delta")
    for row in sorted(all_rows, key=lambda x: (x["checkpoint"], -x["group_mAP_1.0"], -x["outlier_mIoU"])):
        print(
            f"{row['checkpoint']}, {row['threshold']:.2f}, "
            f"{row['group_mAP_1.0']:.2f}, {row['group_mAP_0.5']:.2f}, "
            f"{row['outlier_mIoU']:.2f}, {row['delta_vs_original']}"
        )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
