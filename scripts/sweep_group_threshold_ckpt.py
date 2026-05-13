import csv
import json
import os
import time

import torch

from test import build_parser, collate_fn
from util import experiment
import util.misc as utils
from util.test_eval_utils import (
    build_eval_metrics,
    build_eval_model_and_criterion,
    build_test_loader,
    collect_prediction_rows,
    create_metric_logger,
    evaluate_prediction_file,
    format_threshold_tag,
    get_name_to_vid,
    metric_logger_to_dict,
    resolve_eval_args,
    run_eval_batch,
    setup_eval_environment,
    update_metric_logger,
    write_prediction_rows,
)


def build_sweep_parser():
    parser = build_parser()
    parser.description = "Sweep group_threshold for a single checkpoint"
    parser.add_argument('--threshold_center', default=0.5, type=float,
                        help='center threshold used for automatic symmetric sweep')
    parser.add_argument('--threshold_step', default=0.05, type=float,
                        help='step size used for automatic symmetric sweep')
    parser.add_argument('--steps_before', default=3, type=int,
                        help='how many thresholds to scan below center')
    parser.add_argument('--steps_after', default=3, type=int,
                        help='how many thresholds to scan above center')
    parser.add_argument('--thresholds', nargs='*', type=float, default=None,
                        help='explicit threshold list; if provided, overrides center/step sweep')
    return parser


def build_threshold_grid(args):
    if args.thresholds:
        vals = [float(v) for v in args.thresholds]
    else:
        vals = [
            float(args.threshold_center) + float(args.threshold_step) * offset
            for offset in range(-int(args.steps_before), int(args.steps_after) + 1)
        ]
    vals = sorted({round(v, 6) for v in vals if 0.0 <= float(v) <= 1.0})
    if not vals:
        raise ValueError("No valid thresholds remain after clipping to [0, 1].")
    return vals


def build_output_dir(args):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_GAD_threshold_sweep_[%s]' % (args.dataset, time_str)
    output_dir = os.path.join(args.result_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    return exp_name, output_dir


def save_summary_csv(rows, csv_path):
    fieldnames = ["threshold", "group_mAP_1.0", "group_mAP_0.5", "outlier_mIoU", "pred_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def summarize_best(rows, key):
    best = max(rows, key=lambda item: float(item[key]))
    return {
        "threshold": float(best["threshold"]),
        key: float(best[key]),
        "pred_file": best["pred_file"],
    }


@torch.no_grad()
def collect_prediction_cache(test_loader, model, criterion, args):
    model.eval()
    criterion.eval()

    metric_logger = create_metric_logger()
    header = 'Threshold Sweep Inference: '
    print_freq = len(test_loader)
    name_to_vid = get_name_to_vid()
    rows = []

    for i, (images, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        if i % max(1, len(test_loader) // 10) == 0:
            print(f"  Progress: {i}/{len(test_loader)} batches ({100 * i // len(test_loader)}%)")

        clean_boxes, outputs, loss_dict = run_eval_batch(model, criterion, images, targets, infos, args)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        update_metric_logger(metric_logger, loss_dict_reduced, criterion.weight_dict, outputs, args)
        rows.extend(collect_prediction_rows(clean_boxes, infos, outputs, args, name_to_vid=name_to_vid))

    metric_logger.synchronize_between_processes()
    return rows, metric_logger_to_dict(metric_logger)


def main(argv=None):
    parser = build_sweep_parser()
    args = parser.parse_args(argv)
    resolve_eval_args(args)
    setup_eval_environment(args)
    thresholds = build_threshold_grid(args)
    exp_name, output_dir = build_output_dir(args)

    experiment.save_args(os.path.join(output_dir, "args.json"), vars(args))

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Output path: {output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Thresholds: {', '.join(f'{t:.2f}' for t in thresholds)}")
    print("=" * 60)

    print("\n[1/6] Setting random seed...")
    print("[2/6] Loading dataset...")
    test_set, test_loader = build_test_loader(args, collate_fn)
    print(f"    Dataset loaded: {len(test_set)} test samples")
    print("[3/6] Creating data loader...")
    print(f"    Data loader created: {len(test_loader)} batches")

    print("[4/6] Building model...")
    model, criterion = build_eval_model_and_criterion(args)
    print("    Model built and moved to GPU")
    print("[5/6] Initializing evaluation metrics...")
    metrics = build_eval_metrics(args)
    print("    Metrics initialized")
    print("[6/6] Running single forward pass and caching predictions...")
    pred_rows, forward_stats = collect_prediction_cache(test_loader, model, criterion, args)
    print(f"    Cached {len(pred_rows)} prediction rows")

    cache_path = os.path.join(output_dir, "pred_cache.pt")
    torch.save(
        {
            "rows": pred_rows,
            "thresholds": thresholds,
            "model_path": args.model_path,
            "split": args.split,
        },
        cache_path,
    )

    results = []
    for threshold in thresholds:
        thr_tag = format_threshold_tag(threshold)
        pred_path = os.path.join(output_dir, f"pred_group_test_{args.split}_thr{thr_tag}.txt")
        write_prediction_rows(pred_rows, pred_path, threshold, args)
        result = evaluate_prediction_file(metrics, pred_path)
        row = {
            "threshold": float(threshold),
            "group_mAP_1.0": float(result["group_mAP_1.0"]),
            "group_mAP_0.5": float(result["group_mAP_0.5"]),
            "outlier_mIoU": float(result["outlier_mIoU"]),
            "pred_file": pred_path,
        }
        results.append(row)
        print(
            "threshold=%.2f | mAP@1.0=%.2f | mAP@0.5=%.2f | outlier_mIoU=%.2f"
            % (row["threshold"], row["group_mAP_1.0"], row["group_mAP_0.5"], row["outlier_mIoU"])
        )

    summary = {
        "forward_stats": forward_stats,
        "thresholds": thresholds,
        "results": results,
        "best_group_mAP_1.0": summarize_best(results, "group_mAP_1.0"),
        "best_group_mAP_0.5": summarize_best(results, "group_mAP_0.5"),
        "best_outlier_mIoU": summarize_best(results, "outlier_mIoU"),
        "pred_cache": cache_path,
    }

    summary_json = os.path.join(output_dir, "sweep_summary.json")
    summary_csv = os.path.join(output_dir, "sweep_summary.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_summary_csv(results, summary_csv)

    print("\nBest thresholds:")
    print("  group mAP@1.0 : %.2f @ %.2f" % (
        summary["best_group_mAP_1.0"]["group_mAP_1.0"],
        summary["best_group_mAP_1.0"]["threshold"],
    ))
    print("  group mAP@0.5 : %.2f @ %.2f" % (
        summary["best_group_mAP_0.5"]["group_mAP_0.5"],
        summary["best_group_mAP_0.5"]["threshold"],
    ))
    print("  outlier mIoU  : %.2f @ %.2f" % (
        summary["best_outlier_mIoU"]["outlier_mIoU"],
        summary["best_outlier_mIoU"]["threshold"],
    ))
    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary CSV : {summary_csv}")
    print(f"Saved pred cache  : {cache_path}")


if __name__ == "__main__":
    main()
