import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.dataloader import read_dataset
from models import build_model
from test import build_parser
from util.box_noise import apply_box_noise


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack([image for image in batch[0]])
    return tuple(batch)


def prepare_args(args):
    if args.no_mae:
        args.mae_fusion = "none"
    args.use_mae = (not args.no_mae) and args.mae_fusion != "none"
    args.mae_dim = 1408 if args.use_mae and args.mae_version == "v2" else (768 if args.use_mae else 0)
    if args.olic_dropout < 0:
        args.olic_dropout = args.drop_rate
    if args.olic_warmup_epochs < 0:
        args.olic_warmup_epochs = 0
    if args.olic_attn_tau <= 0:
        args.olic_attn_tau = 1.0
    if args.olic_geom_scale_max <= 0:
        args.olic_geom_scale_max = 2.0
    if args.pmr_anchor_source == "auto":
        object_source_hint = str(args.object_tracks_pkl or "").lower()
        args.pmr_anchor_source = "yolo" if "yolo" in object_source_hint else "gdino"
    return args


def tensor_summary(x):
    x = x.detach()
    return {
        "shape": list(x.shape),
        "finite": bool(torch.isfinite(x).all().item()),
        "mean": float(x.float().mean().item()),
        "std": float(x.float().std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.float().min().item()),
        "max": float(x.float().max().item()),
    }


def require(name, cond):
    if not cond:
        raise RuntimeError(f"Smoke check failed: {name}")


def main():
    parser = build_parser()
    parser.add_argument("--smoke_split", default="train", choices=["train", "test"])
    parser.add_argument("--smoke_batch", default=2, type=int)
    parser.add_argument("--smoke_backward", action="store_true")
    parser.add_argument("--smoke_output_json", default="", type=str)
    args = prepare_args(parser.parse_args())

    require("CUDA is available because build_model moves criterion to cuda", torch.cuda.is_available())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    train_set, test_set = read_dataset(args)
    dataset = train_set if args.smoke_split == "train" else test_set
    loader = data.DataLoader(
        dataset,
        batch_size=max(2, int(args.smoke_batch)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    model, criterion = build_model(args)
    model = model.cuda()
    model.train(args.smoke_backward)
    criterion.train(args.smoke_backward)

    images, targets, infos = next(iter(loader))
    images = images.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

    clean_boxes = torch.stack([t["boxes"] for t in targets])
    boxes = apply_box_noise(clean_boxes, infos=infos, args=args, phase="train")
    dummy_mask = torch.stack([t["actions"] == args.num_class + 1 for t in targets]).squeeze(1)

    mae_feats = None
    if args.use_mae and "mae_feats" in targets[0]:
        mae_feats = torch.stack([t["mae_feats"] for t in targets])

    object_boxes_xyxy = None
    object_valid_mask = None
    object_scores = None
    object_family_id = None
    object_token_id = None
    if args.use_olic and "object_boxes_xyxy" in targets[0]:
        object_boxes_xyxy = torch.stack([t["object_boxes_xyxy"] for t in targets])
        object_valid_mask = torch.stack([t["object_valid_mask"] for t in targets])
        object_scores = torch.stack([t["object_scores"] for t in targets])
        object_family_id = torch.stack([t["object_family_id"] for t in targets]) if "object_family_id" in targets[0] else None
        object_token_id = torch.stack([t["object_token_id"] for t in targets]) if "object_token_id" in targets[0] else None

    outputs = model(
        images,
        boxes,
        dummy_mask,
        mae_feats,
        object_boxes_xyxy=object_boxes_xyxy,
        object_valid_mask=object_valid_mask,
        object_scores=object_scores,
        object_family_id=object_family_id,
        object_token_id=object_token_id,
        olic_warmup_scale=1.0,
    )
    loss_dict = criterion(outputs, targets)
    total_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)

    if args.smoke_backward:
        total_loss.backward()

    bs = images.shape[0]
    n = args.num_boxes
    k = args.num_group_tokens

    require("membership_logits_base exists", "membership_logits_base" in outputs)
    require("membership_logits_refined exists", "membership_logits_refined" in outputs)
    require("membership_logits_base shape", list(outputs["membership_logits_base"].shape) == [bs, k, n])
    require("membership_logits_refined shape", list(outputs["membership_logits_refined"].shape) == [bs, k, n])
    require("membership logits finite", torch.isfinite(outputs["membership_logits_refined"]).all().item())

    summary = {
        "total_loss": float(total_loss.detach().item()),
        "membership_logits_base": tensor_summary(outputs["membership_logits_base"]),
        "membership_logits_refined": tensor_summary(outputs["membership_logits_refined"]),
        "pairwise_refine_delta_mean": float(outputs.get("pairwise_refine_delta_mean", torch.tensor(0.0)).mean().item()),
        "membership_entropy": float(outputs.get("membership_entropy", torch.tensor(0.0)).mean().item()),
        "loss_keys": sorted(loss_dict.keys()),
    }

    if args.use_pairwise_refiner:
        require("pairwise_affinity_logits exists", "pairwise_affinity_logits" in outputs)
        require("pairwise_affinity_logits finite", torch.isfinite(outputs["pairwise_affinity_logits"]).all().item())
        summary["pairwise_affinity_logits"] = tensor_summary(outputs["pairwise_affinity_logits"])
        for key in ("pair_pos_mean", "pair_neg_mean", "pair_gap"):
            require(f"{key} in loss_dict", key in loss_dict)
            summary[key] = float(loss_dict[key].detach().item())

    if args.use_attach_head:
        require("attach_logits exists", "attach_logits" in outputs)
        require("attach_prob exists", "attach_prob" in outputs)
        require("attach_logits shape", list(outputs["attach_logits"].shape) == [bs, n])
        require("attach_prob shape", list(outputs["attach_prob"].shape) == [bs, n])
        require("attach_logits finite", torch.isfinite(outputs["attach_logits"]).all().item())
        require("attach_prob finite", torch.isfinite(outputs["attach_prob"]).all().item())
        require("attach_prob range", ((outputs["attach_prob"] >= 0.0) & (outputs["attach_prob"] <= 1.0)).all().item())
        require("loss_attach in loss_dict", "loss_attach" in loss_dict)
        summary["attach_logits"] = tensor_summary(outputs["attach_logits"])
        summary["attach_prob"] = tensor_summary(outputs["attach_prob"])
        for key in ("loss_attach", "attach_pos_mean", "attach_neg_mean", "attach_gap", "attach_acc"):
            summary[key] = float(loss_dict[key].detach().item())
    else:
        require("attach_logits absent when disabled", "attach_logits" not in outputs)
        require("attach_prob absent when disabled", "attach_prob" not in outputs)
        require("loss_attach absent when disabled", "loss_attach" not in loss_dict)

    if args.membership_margin_loss_coef > 0.0:
        require("loss_membership_margin in loss_dict", "loss_membership_margin" in loss_dict)
        require("loss_membership_margin finite", torch.isfinite(loss_dict["loss_membership_margin"]).item())
        for key in ("loss_membership_margin", "member_margin_loss", "outlier_margin_loss", "member_margin_active", "outlier_margin_active"):
            summary[key] = float(loss_dict[key].detach().item())
    else:
        require("loss_membership_margin absent when disabled", "loss_membership_margin" not in loss_dict)

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.smoke_output_json:
        out_path = Path(args.smoke_output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
