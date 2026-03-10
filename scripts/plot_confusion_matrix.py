import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on PYTHONPATH when running as `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.cafe_eval import read_text_file, make_groups, read_labelmap, cal_group_IoU


def extract_group_records(groups_ids, groups_activity, groups_scores, class_ids, min_members=2):
    records_by_clip = {}
    for clip_key in groups_ids.keys():
        ids_map = groups_ids[clip_key][0]
        act_map = groups_activity[clip_key][0]
        score_map = groups_scores[clip_key][0] if groups_scores is not None else {}

        clip_records = []
        for gid, members in ids_map.items():
            if len(members) < min_members:
                continue

            if gid not in act_map or len(act_map[gid]) == 0:
                continue
            cls_id = next(iter(act_map[gid]))
            if cls_id not in class_ids:
                continue

            score = 1.0
            if groups_scores is not None and gid in score_map and len(score_map[gid]) > 0:
                raw = next(iter(score_map[gid]))
                score = 0.0 if raw is None else float(raw)

            clip_records.append(
                {
                    "group_id": gid,
                    "members": members,
                    "class_id": int(cls_id),
                    "score": float(score),
                }
            )
        records_by_clip[clip_key] = clip_records
    return records_by_clip


def match_groups_and_build_cm(gt_by_clip, pred_by_clip, class_ids, iou_thresh=0.5, include_bg=True):
    class_ids = sorted(class_ids)
    class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
    num_classes = len(class_ids)
    bg_idx = num_classes if include_bg else None
    size = num_classes + 1 if include_bg else num_classes
    cm = np.zeros((size, size), dtype=np.int64)

    clips = sorted(set(gt_by_clip.keys()) | set(pred_by_clip.keys()))

    for clip_key in clips:
        gt_groups = gt_by_clip.get(clip_key, [])
        pred_groups = sorted(pred_by_clip.get(clip_key, []), key=lambda x: x["score"], reverse=True)

        used_gt = set()
        used_pred = set()

        for p_idx, pred in enumerate(pred_groups):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_groups):
                if g_idx in used_gt:
                    continue
                iou = cal_group_IoU(
                    [clip_key, pred["group_id"], pred["members"]],
                    [clip_key, gt["group_id"], gt["members"]],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx >= 0 and best_iou >= iou_thresh:
                used_gt.add(best_gt_idx)
                used_pred.add(p_idx)
                gt_cls = gt_groups[best_gt_idx]["class_id"]
                pred_cls = pred["class_id"]
                cm[class_to_idx[gt_cls], class_to_idx[pred_cls]] += 1

        if include_bg:
            for g_idx, gt in enumerate(gt_groups):
                if g_idx not in used_gt:
                    cm[class_to_idx[gt["class_id"]], bg_idx] += 1
            for p_idx, pred in enumerate(pred_groups):
                if p_idx not in used_pred:
                    cm[bg_idx, class_to_idx[pred["class_id"]]] += 1

    return cm


def save_cm_csv(cm, labels, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + row.tolist())


def plot_cm(cm, labels, out_path, normalize=False, title="Group Confusion Matrix (IoU=0.5)"):
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        cm_vis = np.divide(cm, np.maximum(row_sum, 1.0))
    else:
        cm_vis = cm.astype(np.float64)

    fig_w = max(8, 0.9 * len(labels))
    fig_h = max(6, 0.75 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm_vis, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Ground Truth",
        xlabel="Prediction",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    thresh = cm_vis.max() / 2.0 if cm_vis.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text_val = f"{cm[i, j]}\n({cm_vis[i, j]:.2f})"
            else:
                text_val = str(cm[i, j])
            ax.text(
                j,
                i,
                text_val,
                ha="center",
                va="center",
                color="white" if cm_vis[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot group-level confusion matrix at IoU=0.5.")
    parser.add_argument("--pred_txt", required=True, type=str, help="Prediction txt path")
    parser.add_argument("--groundtruth", default="./evaluation/gt_tracks.txt", type=str, help="GT txt path")
    parser.add_argument("--labelmap", default="./label_map/group_action_list.pbtxt", type=str, help="Label map path")
    parser.add_argument("--eval_type", default="gt_base", choices=["gt_base", "detect_base"], type=str)
    parser.add_argument("--iou_thresh", default=0.5, type=float, help="Group IoU threshold (default: 0.5)")
    parser.add_argument("--min_members", default=2, type=int, help="Minimum group size to evaluate")
    parser.add_argument("--exclude_bg", action="store_true", help="Exclude background row/column")
    parser.add_argument("--normalize", action="store_true", help="Show normalized matrix values in plot")
    parser.add_argument("--out_dir", default="", type=str, help="Output folder (default: pred txt folder)")
    args = parser.parse_args()

    pred_path = Path(args.pred_txt)
    if not pred_path.exists():
        raise FileNotFoundError(f"pred txt not found: {pred_path}")
    gt_path = Path(args.groundtruth)
    if not gt_path.exists():
        raise FileNotFoundError(f"groundtruth not found: {gt_path}")
    labelmap_path = Path(args.labelmap)
    if not labelmap_path.exists():
        raise FileNotFoundError(f"labelmap not found: {labelmap_path}")

    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labelmap, "r", encoding="utf-8") as f:
        categories, class_ids = read_labelmap(f)
    class_ids = sorted(class_ids)
    id_to_name = {int(c["id"]): c["name"] for c in categories}

    with open(args.groundtruth, "r", encoding="utf-8") as f:
        gt_boxes, gt_g_labels, gt_act_labels, _, gt_g_scores = read_text_file(f, args.eval_type, mode="gt")
    with open(args.pred_txt, "r", encoding="utf-8") as f:
        pred_boxes, pred_g_labels, pred_act_labels, _, pred_g_scores = read_text_file(f, args.eval_type, mode="pred")

    gt_groups_ids, gt_groups_activity, gt_groups_scores = make_groups(
        gt_boxes, gt_g_labels, gt_act_labels, gt_g_scores
    )
    pred_groups_ids, pred_groups_activity, pred_groups_scores = make_groups(
        pred_boxes, pred_g_labels, pred_act_labels, pred_g_scores
    )

    gt_by_clip = extract_group_records(
        gt_groups_ids, gt_groups_activity, gt_groups_scores, class_ids, min_members=args.min_members
    )
    pred_by_clip = extract_group_records(
        pred_groups_ids, pred_groups_activity, pred_groups_scores, class_ids, min_members=args.min_members
    )

    include_bg = not args.exclude_bg
    cm = match_groups_and_build_cm(
        gt_by_clip, pred_by_clip, class_ids, iou_thresh=args.iou_thresh, include_bg=include_bg
    )

    labels = [id_to_name[cid] for cid in class_ids]
    if include_bg:
        labels.append("BG")

    npy_path = out_dir / "confusion_matrix_iou0.5.npy"
    csv_path = out_dir / "confusion_matrix_iou0.5.csv"
    png_path = out_dir / "confusion_matrix_iou0.5.png"
    np.save(npy_path, cm)
    save_cm_csv(cm, labels, csv_path)
    plot_cm(
        cm,
        labels,
        png_path,
        normalize=args.normalize,
        title=f"Group Confusion Matrix (IoU={args.iou_thresh:.2f})",
    )

    print("Saved confusion matrix:")
    print(f"  NPY: {npy_path}")
    print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
