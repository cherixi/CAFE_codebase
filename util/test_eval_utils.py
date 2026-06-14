import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

import evaluation.cafe_eval as evaluation
import util.logger as loggers
import util.misc as utils
from dataloader.dataloader import read_dataset
from models import build_model
from util.box_noise import apply_box_noise


SEQS_CAFE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


def resolve_eval_args(args):
    if getattr(args, "pmr_anchor_source", "auto") == "auto":
        object_source_hint = str(getattr(args, "object_tracks_pkl", "") or "").lower()
        args.pmr_anchor_source = "yolo" if "yolo" in object_source_hint else "gdino"

    if getattr(args, "no_mae", False):
        args.mae_fusion = "none"
    args.use_mae = (not getattr(args, "no_mae", False)) and getattr(args, "mae_fusion", "none") != "none"
    args.mae_dim = 1408 if getattr(args, "mae_version", "v2") == "v2" else 768
    if not args.use_mae:
        args.mae_dim = 0

    if getattr(args, "olic_dropout", -1.0) < 0:
        args.olic_dropout = args.drop_rate

    return args


def setup_eval_environment(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_test_loader(args, collate_fn):
    _, test_set = read_dataset(args)

    if args.distributed:
        sampler_test = data.DistributedSampler(test_set, shuffle=False)
    else:
        sampler_test = data.RandomSampler(test_set)

    num_workers = int(getattr(args, "eval_num_workers", 2))
    loader_kwargs = {
        "drop_last": False,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(getattr(args, "eval_persistent_workers", True)),
            prefetch_factor=int(getattr(args, "eval_prefetch_factor", 2)),
        )

    test_loader = data.DataLoader(
        test_set,
        args.test_batch,
        sampler=sampler_test,
        **loader_kwargs,
    )
    return test_set, test_loader


def build_eval_model_and_criterion(args):
    model, criterion = build_model(args)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model, criterion


def build_eval_metrics(args):
    return evaluation.GAD_Evaluation(args)


def create_metric_logger():
    return loggers.MetricLogger(mode="test", delimiter="  ")


def get_name_to_vid():
    return {name: i + 1 for i, name in enumerate(SEQS_CAFE)}


def update_metric_logger(metric_logger, loss_dict_reduced, weight_dict, outputs, args):
    loss_dict_reduced_scaled = {
        k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
    }
    loss_dict_reduced_unscaled = {
        f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
    }
    metric_logger.update(
        loss=sum(loss_dict_reduced_scaled.values()),
        **loss_dict_reduced_scaled,
        **loss_dict_reduced_unscaled,
    )

    metric_logger.update(group_class_error=loss_dict_reduced["group_class_error"])
    if args.use_pairwise_refiner and "pairwise_refine_delta_mean" in outputs:
        metric_logger.update(
            pairwise_refine_delta_mean=float(outputs["pairwise_refine_delta_mean"].mean().item()),
            qpmr_support_abs_mean=float(outputs["qpmr_support_abs_mean"].mean().item()),
            membership_entropy=float(outputs["membership_entropy"].mean().item()),
            query_conditioned_pmr=float(outputs["query_conditioned_pmr"].mean().item()),
            type_aware_object_token=float(outputs["type_aware_object_token"].mean().item()),
            group_olic_disabled=float(outputs["group_olic_disabled"].mean().item()),
        )
        if "pair_pos_mean" in loss_dict_reduced:
            metric_logger.update(
                pair_pos_mean=float(loss_dict_reduced["pair_pos_mean"].item()),
                pair_neg_mean=float(loss_dict_reduced["pair_neg_mean"].item()),
                pair_gap=float(loss_dict_reduced["pair_gap"].item()),
            )
        if "qpair_pos_mean" in loss_dict_reduced:
            metric_logger.update(
                query_pairwise_loss=float(loss_dict_reduced["loss_query_pairwise_group"].item()),
                qpair_pos_mean=float(loss_dict_reduced["qpair_pos_mean"].item()),
                qpair_neg_mean=float(loss_dict_reduced["qpair_neg_mean"].item()),
                qpair_gap=float(loss_dict_reduced["qpair_gap"].item()),
                qpair_matched_query_count=float(loss_dict_reduced["qpair_matched_query_count"].item()),
                qpair_active_pair_count=float(loss_dict_reduced["qpair_active_pair_count"].item()),
            )
    if args.use_olic and "small_valid_obj_per_actor" in outputs:
        metric_logger.update(
            small_valid_obj_per_actor=float(outputs["small_valid_obj_per_actor"].mean().item()),
            anchor_valid_obj_per_actor=float(outputs["anchor_valid_obj_per_actor"].mean().item()),
            shared_table_mean=float(outputs["shared_table_mean"].mean().item()),
            shared_service_mean=float(outputs["shared_service_mean"].mean().item()),
        )
    if getattr(args, "use_attach_head", False) and "attach_pos_mean" in loss_dict_reduced:
        metric_logger.update(
            attach_loss=float(loss_dict_reduced["loss_attach"].item()),
            attach_pos_mean=float(loss_dict_reduced["attach_pos_mean"].item()),
            attach_neg_mean=float(loss_dict_reduced["attach_neg_mean"].item()),
            attach_gap=float(loss_dict_reduced["attach_gap"].item()),
            attach_acc=float(loss_dict_reduced["attach_acc"].item()),
        )
    if getattr(args, "membership_margin_loss_coef", 0.0) > 0.0 and "loss_membership_margin" in loss_dict_reduced:
        metric_logger.update(
            membership_margin_loss=float(loss_dict_reduced["loss_membership_margin"].item()),
            member_margin_loss=float(loss_dict_reduced["member_margin_loss"].item()),
            outlier_margin_loss=float(loss_dict_reduced["outlier_margin_loss"].item()),
            member_margin_active=float(loss_dict_reduced["member_margin_active"].item()),
            outlier_margin_active=float(loss_dict_reduced["outlier_margin_active"].item()),
        )


def compute_keep_score(membership_max, attach_prob, args):
    mode = getattr(args, "attach_infer_mode", "membership_only")
    if not getattr(args, "use_attach_head", False) or attach_prob is None:
        mode = "membership_only"

    if mode == "attach_only":
        return attach_prob
    if mode == "joint":
        return torch.sqrt((attach_prob * membership_max).clamp(min=0.0))
    return membership_max


@torch.no_grad()
def run_eval_batch(model, criterion, images, targets, infos, args):
    images = images.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

    clean_boxes = torch.stack([t["boxes"] for t in targets])
    boxes = apply_box_noise(clean_boxes, infos=infos, args=args, phase="infer")
    dummy_mask = torch.stack([t["actions"] == args.num_class + 1 for t in targets]).squeeze()

    mae_feats = None
    if args.use_mae and "mae_feats" in targets[0]:
        mae_feats = torch.stack([t["mae_feats"] for t in targets])

    object_boxes_xyxy = None
    object_valid_mask = None
    object_scores = None
    object_token_id = None
    object_family_id = None
    if args.use_olic and "object_boxes_xyxy" in targets[0]:
        object_boxes_xyxy = torch.stack([t["object_boxes_xyxy"] for t in targets])
        object_valid_mask = torch.stack([t["object_valid_mask"] for t in targets])
        object_scores = torch.stack([t["object_scores"] for t in targets])
        if "object_token_id" in targets[0]:
            object_token_id = torch.stack([t["object_token_id"] for t in targets])
        if "object_family_id" in targets[0]:
            object_family_id = torch.stack([t["object_family_id"] for t in targets])

    outputs = model(
        images, boxes, dummy_mask, mae_feats,
        object_boxes_xyxy=object_boxes_xyxy,
        object_valid_mask=object_valid_mask,
        object_scores=object_scores,
        object_family_id=object_family_id,
        object_token_id=object_token_id,
        olic_warmup_scale=1.0,
    )

    loss_dict = criterion(outputs, targets)
    return clean_boxes, outputs, loss_dict


def collect_prediction_rows(clean_boxes, infos, outputs, args, name_to_vid=None):
    if name_to_vid is None:
        name_to_vid = get_name_to_vid()

    rows: List[Dict] = []
    image_w = args.eval_image_width
    image_h = args.eval_image_height
    no_group_class_idx = outputs["pred_activities"].shape[-1] - 1

    for b in range(clean_boxes.shape[0]):
        pred_group_actions = F.softmax(outputs["pred_activities"][b], dim=1).detach().cpu()
        members = outputs["membership"][b].detach().cpu()
        pred_membership = torch.argmax(members.transpose(0, 1), dim=1)
        membership_max = members.transpose(0, 1).max(-1).values
        attach_prob = None
        if "attach_prob" in outputs:
            attach_prob = outputs["attach_prob"][b].detach().cpu()
        keep_score = compute_keep_score(membership_max, attach_prob, args)
        pred_group_action = torch.argmax(pred_group_actions, dim=1)

        for t in range(clean_boxes.shape[1]):
            for box_idx in range(clean_boxes.shape[2]):
                x, y, w, h = clean_boxes[b][t][box_idx].tolist()
                x1 = (x - w / 2.0) * image_w
                y1 = (y - h / 2.0) * image_h
                x2 = (x + w / 2.0) * image_w
                y2 = (y + h / 2.0) * image_h
                is_valid = not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0)

                pred_group_id = int(pred_membership[box_idx].item())
                pred_group_action_idx = int(pred_group_action[pred_group_id].item())
                pred_group_action_prob = float(
                    pred_group_actions[pred_group_id][pred_group_action_idx].item()
                )

                rows.append(
                    {
                        "vid": int(name_to_vid[infos[b]["vid"]]),
                        "sid": int(infos[b]["sid"]),
                        "fid": int(infos[b]["fid"][t]),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "pred_group_id": pred_group_id,
                        "pred_group_action_idx": pred_group_action_idx,
                        "pred_group_action_prob": pred_group_action_prob,
                        "membership_max": float(membership_max[box_idx].item()),
                        "attach_prob": float(attach_prob[box_idx].item()) if attach_prob is not None else 1.0,
                        "keep_score": float(keep_score[box_idx].item()),
                        "is_valid": bool(is_valid),
                        "is_no_group_query": bool(pred_group_action_idx == no_group_class_idx),
                    }
                )
    return rows


def format_threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace("-", "m").replace(".", "p")


def write_prediction_rows(rows, file_path: str, threshold: float, args):
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            if not row["is_valid"]:
                continue

            pred_group_id = int(row["pred_group_id"])
            pred_group_action_idx = int(row["pred_group_action_idx"])
            score = float(row.get("keep_score", row["membership_max"]))
            if (not row["is_no_group_query"]) and not (score > float(threshold)):
                pred_group_id = -1
                pred_group_action_idx = int(args.num_class)

            pred_list = [
                int(row["vid"]),
                int(row["sid"]),
                int(row["fid"]),
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
                int(pred_group_id),
                int(pred_group_action_idx) + 1,
                float(row["pred_group_action_prob"]),
            ]
            f.write(" ".join(str(v) for v in pred_list) + "\r\n")


def evaluate_prediction_file(metrics, file_path: str):
    with open(file_path, "r", encoding="utf-8") as detections:
        return metrics.evaluate(detections)


def metric_logger_to_dict(metric_logger):
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
