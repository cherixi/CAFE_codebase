#!/usr/bin/env python3
"""
Offline object proposal extraction for CAFE using GroundingDINO-B (Swin-B).

Pipeline (per frame):
1) Run three semantic prompt packs independently.
2) Map predicted phrases to stable semantic families.
3) Class-aware NMS (by family).
4) Family-level caps.
5) Global capped selection with family quota (default M=10).

Output format (pickle):
    tracks[(vid, cid)][fid] = np.ndarray shape [M, 10], dtype=float32
    row = [obj_id, x1, y1, x2, y2, score, family_id, pack_id, token_id, valid_mask]
       padded rows are all-zeros when frame proposals < M.
    valid_mask is 1.0 for valid rows, 0.0 for padded rows.

Coordinates are normalized xyxy in [0, 1].
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from tqdm import tqdm

try:
    from groundingdino.util.inference import load_image, load_model, predict
except ImportError as exc:
    raise ImportError(
        "Failed to import GroundingDINO. Install and activate its environment first."
    ) from exc


FAMILY_ORDER = [
    "phone-family",
    "study-family",
    "dining-family",
    "table-family",
    "seat-family",
    "service-family",
]
FAMILY_TO_ID = {name: idx + 1 for idx, name in enumerate(FAMILY_ORDER)}
ID_TO_FAMILY = {v: k for k, v in FAMILY_TO_ID.items()}

TOKEN_ORDER = [
    "phone",
    "smartphone",
    "laptop",
    "tablet",
    "book",
    "notebook",
    "paper",
    "cup",
    "mug",
    "bottle",
    "tray",
    "plate",
    "bowl",
    "food container",
    "table",
    "desk",
    "chair",
    "seat",
    "counter",
    "cashier counter",
    "service counter",
    "pickup counter",
]
TOKEN_TO_ID = {name: idx + 1 for idx, name in enumerate(TOKEN_ORDER)}
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}

TOKEN_TO_FAMILY = {
    "phone": "phone-family",
    "smartphone": "phone-family",
    "laptop": "study-family",
    "tablet": "study-family",
    "book": "study-family",
    "notebook": "study-family",
    "paper": "study-family",
    "cup": "dining-family",
    "mug": "dining-family",
    "bottle": "dining-family",
    "tray": "dining-family",
    "plate": "dining-family",
    "bowl": "dining-family",
    "food container": "dining-family",
    "table": "table-family",
    "desk": "table-family",
    "chair": "seat-family",
    "seat": "seat-family",
    "counter": "service-family",
    "cashier counter": "service-family",
    "service counter": "service-family",
    "pickup counter": "service-family",
}

# Phrase aliases -> canonical token
ALIAS_TO_TOKEN = {
    "cell phone": "phone",
    "phone": "phone",
    "smartphone": "smartphone",
    "laptop": "laptop",
    "tablet": "tablet",
    "book": "book",
    "notebook": "notebook",
    "paper": "paper",
    "cup": "cup",
    "mug": "mug",
    "bottle": "bottle",
    "tray": "tray",
    "plate": "plate",
    "bowl": "bowl",
    "food container": "food container",
    "table": "table",
    "desk": "desk",
    "chair": "chair",
    "seat": "seat",
    "counter": "counter",
    "cashier counter": "cashier counter",
    "service counter": "service counter",
    "pickup counter": "pickup counter",
}
ALIAS_KEYS_SORTED = sorted(ALIAS_TO_TOKEN.keys(), key=len, reverse=True)


@dataclass(frozen=True)
class PromptPack:
    name: str
    pack_id: int
    caption: str
    box_threshold: float
    text_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract offline object proposals from CAFE using GroundingDINO-B."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"D:\Cafe_Dataset\Cafe_Dataset\Dataset\cafe",
        help="Path to CAFE directory containing video folders and ann/images.",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="",
        help="Output pickle path. Default: <data_root>/object_tracks_gdino_swinb.pkl",
    )
    parser.add_argument(
        "--output_meta",
        type=str,
        default="",
        help="Output meta json path. Default: <data_root>/object_tracks_gdino_swinb_meta.json",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="GroundingDINO config path (e.g., GroundingDINO_SwinB_cfg.py).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="GroundingDINO checkpoint path (e.g., groundingdino_swinb_cogcoor.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: cuda or cpu.",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=5e-5,
        help="Drop boxes with normalized area below this threshold.",
    )
    parser.add_argument(
        "--global_topk",
        type=int,
        default=10,
        help="Final per-frame proposal cap (M).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output pkl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used for visualization frame sampling.",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="",
        help="Optional visualization output directory.",
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=0,
        help="Number of random frames to dump visualization for.",
    )

    # Per-pack thresholds
    parser.add_argument("--box_th_a", type=float, default=0.25)
    parser.add_argument("--box_th_b", type=float, default=0.27)
    parser.add_argument("--box_th_c", type=float, default=0.35)
    parser.add_argument("--text_th_a", type=float, default=0.20)
    parser.add_argument("--text_th_b", type=float, default=0.22)
    parser.add_argument("--text_th_c", type=float, default=0.25)

    # Family caps
    parser.add_argument("--cap_phone", type=int, default=4)
    parser.add_argument("--cap_study", type=int, default=6)
    parser.add_argument("--cap_dining", type=int, default=6)
    parser.add_argument("--cap_table", type=int, default=4)
    parser.add_argument("--cap_seat", type=int, default=6)
    parser.add_argument("--cap_service", type=int, default=2)

    # Global family quota (used before final score fill-up)
    parser.add_argument("--quota_phone", type=int, default=1)
    parser.add_argument("--quota_study", type=int, default=2)
    parser.add_argument("--quota_dining", type=int, default=2)
    parser.add_argument("--quota_table", type=int, default=1)
    parser.add_argument("--quota_seat", type=int, default=1)
    parser.add_argument("--quota_service", type=int, default=1)

    # Family-specific NMS IoU
    parser.add_argument("--nms_iou_phone", type=float, default=0.50)
    parser.add_argument("--nms_iou_study", type=float, default=0.50)
    parser.add_argument("--nms_iou_dining", type=float, default=0.50)
    parser.add_argument("--nms_iou_table", type=float, default=0.50)
    parser.add_argument("--nms_iou_seat", type=float, default=0.40)
    parser.add_argument("--nms_iou_service", type=float, default=0.40)

    parser.add_argument(
        "--allow_unknown_phrase",
        action="store_true",
        help="Keep unmatched phrases as unknown and skip family mapping constraints. Disabled by default.",
    )
    return parser.parse_args()


def normalize_phrase(phrase: str) -> str:
    phrase = phrase.lower().strip()
    phrase = phrase.strip(".")
    phrase = re.sub(r"[,_;:]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def map_phrase_to_token(phrase: str) -> Optional[str]:
    if phrase in ALIAS_TO_TOKEN:
        return ALIAS_TO_TOKEN[phrase]
    for alias in ALIAS_KEYS_SORTED:
        if alias in phrase:
            return ALIAS_TO_TOKEN[alias]
    return None


def boxes_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def parse_fid_from_name(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Failed to parse fid from filename: {path.name}")
    return int(numbers[-1])


def sorted_numeric_dirs(root: Path) -> List[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def collect_clips(data_root: Path) -> Tuple[List[Dict], List[Tuple[int, int, int]]]:
    clips: List[Dict] = []
    frame_keys: List[Tuple[int, int, int]] = []
    for vid_dir in sorted_numeric_dirs(data_root):
        vid = int(vid_dir.name)
        for cid_dir in sorted_numeric_dirs(vid_dir):
            cid = int(cid_dir.name)
            image_dir = cid_dir / "images"
            if not image_dir.exists():
                continue

            frame_files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                frame_files.extend(image_dir.glob(ext))
            if not frame_files:
                continue

            fid_to_path = {}
            for path in frame_files:
                fid = parse_fid_from_name(path)
                fid_to_path[fid] = path
                frame_keys.append((vid, cid, fid))

            ordered_frames = sorted(fid_to_path.items(), key=lambda x: x[0])
            max_fid = ordered_frames[-1][0]
            clips.append(
                {
                    "vid": vid,
                    "cid": cid,
                    "frames": ordered_frames,
                    "max_fid": max_fid,
                }
            )
    return clips, frame_keys


def build_prompt_packs(args: argparse.Namespace) -> List[PromptPack]:
    return [
        PromptPack(
            name="A",
            pack_id=1,
            caption="cell phone . smartphone . laptop . tablet . book . notebook . paper .",
            box_threshold=args.box_th_a,
            text_threshold=args.text_th_a,
        ),
        PromptPack(
            name="B",
            pack_id=2,
            caption="cup . mug . bottle . tray . plate . bowl . food container . table . desk .",
            box_threshold=args.box_th_b,
            text_threshold=args.text_th_b,
        ),
        PromptPack(
            name="C",
            pack_id=3,
            caption="chair . seat . counter . cashier counter . service counter . pickup counter .",
            box_threshold=args.box_th_c,
            text_threshold=args.text_th_c,
        ),
    ]


def get_family_caps(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "phone-family": args.cap_phone,
        "study-family": args.cap_study,
        "dining-family": args.cap_dining,
        "table-family": args.cap_table,
        "seat-family": args.cap_seat,
        "service-family": args.cap_service,
    }


def get_family_quota(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "phone-family": args.quota_phone,
        "study-family": args.quota_study,
        "dining-family": args.quota_dining,
        "table-family": args.quota_table,
        "seat-family": args.quota_seat,
        "service-family": args.quota_service,
    }


def get_family_nms_iou(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "phone-family": args.nms_iou_phone,
        "study-family": args.nms_iou_study,
        "dining-family": args.nms_iou_dining,
        "table-family": args.nms_iou_table,
        "seat-family": args.nms_iou_seat,
        "service-family": args.nms_iou_service,
    }


def run_packs_for_frame(
    model,
    image_path: Path,
    prompt_packs: Sequence[PromptPack],
    min_area: float,
    allow_unknown_phrase: bool,
) -> Tuple[np.ndarray, List[dict]]:
    image_source, image_tensor = load_image(str(image_path))
    proposals: List[dict] = []
    for pack in prompt_packs:
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=pack.caption,
            box_threshold=pack.box_threshold,
            text_threshold=pack.text_threshold,
        )
        if boxes is None or len(boxes) == 0:
            continue

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)

        boxes = boxes.detach().cpu().float()
        logits = logits.detach().cpu().float()
        boxes_xyxy = boxes_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)

        for idx in range(boxes_xyxy.shape[0]):
            phrase = normalize_phrase(str(phrases[idx]))
            token = map_phrase_to_token(phrase)
            if token is None:
                if allow_unknown_phrase:
                    family = "unknown-family"
                    family_id = 0
                    token_id = 0
                    token_name = "unknown"
                else:
                    continue
            else:
                family = TOKEN_TO_FAMILY.get(token)
                if family is None:
                    continue
                family_id = FAMILY_TO_ID[family]
                token_id = TOKEN_TO_ID[token]
                token_name = token

            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[idx].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            proposals.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(logits[idx].item()),
                    "family": family,
                    "family_id": family_id,
                    "pack_id": pack.pack_id,
                    "pack_name": pack.name,
                    "token": token_name,
                    "token_id": token_id,
                    "raw_phrase": phrase,
                }
            )
    return image_source, proposals


def apply_class_aware_nms(
    proposals: Sequence[dict],
    family_nms_iou: Dict[str, float],
) -> List[dict]:
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for proposal in proposals:
        by_family[proposal["family"]].append(proposal)

    kept: List[dict] = []
    for family, items in by_family.items():
        if len(items) == 1:
            kept.append(items[0])
            continue
        boxes = torch.tensor(
            [[p["x1"], p["y1"], p["x2"], p["y2"]] for p in items], dtype=torch.float32
        )
        scores = torch.tensor([p["score"] for p in items], dtype=torch.float32)
        iou_thr = family_nms_iou.get(family, 0.5)
        keep_idx = nms(boxes, scores, iou_thr).tolist()
        keep_idx = sorted(keep_idx, key=lambda i: items[i]["score"], reverse=True)
        for i in keep_idx:
            kept.append(items[i])

    kept.sort(key=lambda p: p["score"], reverse=True)
    return kept


def apply_family_caps(
    proposals: Sequence[dict],
    family_caps: Dict[str, int],
) -> List[dict]:
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for proposal in proposals:
        by_family[proposal["family"]].append(proposal)

    kept: List[dict] = []
    for family, items in by_family.items():
        cap = family_caps.get(family, len(items))
        items_sorted = sorted(items, key=lambda p: p["score"], reverse=True)
        kept.extend(items_sorted[: max(cap, 0)])
    kept.sort(key=lambda p: p["score"], reverse=True)
    return kept


def apply_global_quota_and_topk(
    proposals: Sequence[dict],
    family_quota: Dict[str, int],
    global_topk: int,
) -> List[dict]:
    if global_topk <= 0:
        return sorted(proposals, key=lambda p: p["score"], reverse=True)

    by_family: Dict[str, List[dict]] = defaultdict(list)
    for proposal in proposals:
        by_family[proposal["family"]].append(proposal)
    for family in by_family.keys():
        by_family[family].sort(key=lambda p: p["score"], reverse=True)

    selected: List[dict] = []
    leftovers: List[dict] = []

    for family in FAMILY_ORDER:
        items = by_family.get(family, [])
        quota = max(family_quota.get(family, 0), 0)
        remaining = max(global_topk - len(selected), 0)
        take_n = min(quota, len(items), remaining)
        selected.extend(items[:take_n])
        leftovers.extend(items[take_n:])

    known = set(FAMILY_ORDER)
    for family, items in by_family.items():
        if family not in known:
            leftovers.extend(items)

    if len(selected) < global_topk and leftovers:
        leftovers.sort(key=lambda p: p["score"], reverse=True)
        selected.extend(leftovers[: global_topk - len(selected)])

    if len(selected) > global_topk:
        selected = sorted(selected, key=lambda p: p["score"], reverse=True)[:global_topk]

    selected.sort(key=lambda p: p["score"], reverse=True)
    return selected


def proposals_to_fixed_array(
    selected: Sequence[dict],
    fixed_len: int,
) -> np.ndarray:
    if fixed_len < 0:
        raise ValueError(f"fixed_len must be >= 0, got {fixed_len}")
    if fixed_len == 0:
        return np.zeros((0, 10), dtype=np.float32)

    out = np.zeros((fixed_len, 10), dtype=np.float32)

    take_n = min(len(selected), fixed_len)
    for idx in range(take_n):
        proposal = selected[idx]
        out[idx] = np.asarray(
            [
                float(idx + 1),
                float(proposal["x1"]),
                float(proposal["y1"]),
                float(proposal["x2"]),
                float(proposal["y2"]),
                float(proposal["score"]),
                float(proposal["family_id"]),
                float(proposal["pack_id"]),
                float(proposal["token_id"]),
                1.0,
            ],
            dtype=np.float32,
        )
    return out


def draw_visualization(
    image_source: np.ndarray,
    selected: Sequence[dict],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_source)
    draw = ImageDraw.Draw(image)
    w, h = image.size

    colors = {
        "phone-family": (255, 102, 102),
        "study-family": (102, 178, 255),
        "dining-family": (255, 153, 51),
        "table-family": (153, 102, 255),
        "seat-family": (51, 204, 153),
        "service-family": (255, 204, 0),
    }

    for proposal in selected:
        x1 = int(proposal["x1"] * w)
        y1 = int(proposal["y1"] * h)
        x2 = int(proposal["x2"] * w)
        y2 = int(proposal["y2"] * h)
        family = proposal["family"]
        color = colors.get(family, (255, 255, 255))
        label = f"{proposal['token']}:{proposal['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)

    image.save(save_path)


def ensure_outputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    data_root = Path(args.data_root)
    if args.output_pkl:
        output_pkl = Path(args.output_pkl)
    else:
        output_pkl = data_root / "object_tracks_gdino_swinb.pkl"

    if args.output_meta:
        output_meta = Path(args.output_meta)
    else:
        output_meta = data_root / "object_tracks_gdino_swinb_meta.json"

    for path in (output_pkl, output_meta):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {path}. Use --overwrite to replace it."
            )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    return output_pkl, output_meta


def main() -> None:
    args = parse_args()
    if args.global_topk <= 0:
        raise ValueError("--global_topk must be > 0 for fixed-length export.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_pkl, output_meta = ensure_outputs(args)
    prompt_packs = build_prompt_packs(args)
    family_caps = get_family_caps(args)
    family_quota = get_family_quota(args)
    family_nms_iou = get_family_nms_iou(args)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    print(f"[1/5] Loading GroundingDINO model on device={args.device} ...")
    model = load_model(args.config_path, args.checkpoint_path, device=args.device)

    print(f"[2/5] Scanning clips under: {data_root}")
    clips, frame_keys = collect_clips(data_root)
    if not clips:
        raise RuntimeError(f"No valid clips found under: {data_root}")
    print(f"  Found clips: {len(clips)}")
    print(f"  Found frames: {len(frame_keys)}")

    vis_key_set = set()
    if args.vis_dir and args.vis_samples > 0 and frame_keys:
        sampled = random.sample(frame_keys, k=min(args.vis_samples, len(frame_keys)))
        vis_key_set = set(sampled)
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Visualization samples: {len(vis_key_set)}")

    print("[3/5] Running extraction ...")
    tracks: Dict[Tuple[int, int], List[np.ndarray]] = {}

    stats = {
        "total_clips": len(clips),
        "total_frames": 0,
        "frames_no_raw": 0,
        "frames_no_final": 0,
        "raw_props_total": 0,
        "nms_props_total": 0,
        "capped_props_total": 0,
        "final_props_total": 0,
        "final_valid_total": 0,
        "errors": 0,
    }
    family_counter = Counter()
    token_counter = Counter()
    pack_counter = Counter()

    clip_bar = tqdm(clips, desc="Clips", ncols=100)
    for clip in clip_bar:
        vid = clip["vid"]
        cid = clip["cid"]
        max_fid = clip["max_fid"]
        clip_tracks = [
            np.zeros((args.global_topk, 10), dtype=np.float32) for _ in range(max_fid + 1)
        ]

        for fid, image_path in clip["frames"]:
            stats["total_frames"] += 1
            try:
                image_source, raw_props = run_packs_for_frame(
                    model=model,
                    image_path=image_path,
                    prompt_packs=prompt_packs,
                    min_area=args.min_area,
                    allow_unknown_phrase=args.allow_unknown_phrase,
                )
            except Exception:
                stats["errors"] += 1
                raw_props = []
                image_source = None

            if not raw_props:
                stats["frames_no_raw"] += 1
                stats["frames_no_final"] += 1
                continue

            stats["raw_props_total"] += len(raw_props)
            for p in raw_props:
                pack_counter[p["pack_name"]] += 1

            nms_props = apply_class_aware_nms(raw_props, family_nms_iou)
            stats["nms_props_total"] += len(nms_props)

            capped_props = apply_family_caps(nms_props, family_caps)
            stats["capped_props_total"] += len(capped_props)

            final_props = apply_global_quota_and_topk(
                capped_props, family_quota, args.global_topk
            )
            stats["final_props_total"] += len(final_props)
            if not final_props:
                stats["frames_no_final"] += 1
                continue

            for p in final_props:
                family_counter[p["family"]] += 1
                token_counter[p["token"]] += 1

            fixed_rows = proposals_to_fixed_array(
                final_props, fixed_len=args.global_topk
            )
            clip_tracks[fid] = fixed_rows
            stats["final_valid_total"] += int(fixed_rows[:, 9].sum())

            frame_key = (vid, cid, fid)
            if image_source is not None and frame_key in vis_key_set:
                vis_path = Path(args.vis_dir) / f"{vid}_{cid}_{fid}.jpg"
                draw_visualization(image_source, final_props, vis_path)

        tracks[(vid, cid)] = clip_tracks

    print(f"[4/5] Saving outputs ...")
    with open(output_pkl, "wb") as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

    avg_raw = (
        float(stats["raw_props_total"]) / max(stats["total_frames"], 1)
    )
    avg_final = (
        float(stats["final_props_total"]) / max(stats["total_frames"], 1)
    )
    avg_valid = (
        float(stats["final_valid_total"]) / max(stats["total_frames"], 1)
    )

    meta = {
        "format": {
            "row": [
                "obj_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "score",
                "family_id",
                "pack_id",
                "token_id",
                "valid_mask",
            ],
            "coords": "normalized_xyxy",
            "fixed_len_M": args.global_topk,
            "valid_mask_dtype": "float32(0_or_1)",
            "padding_row": "all-zero",
        },
        "paths": {
            "data_root": str(data_root),
            "output_pkl": str(output_pkl),
            "output_meta": str(output_meta),
            "vis_dir": str(args.vis_dir) if args.vis_dir else "",
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
        },
        "prompt_packs": [
            {
                "name": pack.name,
                "pack_id": pack.pack_id,
                "caption": pack.caption,
                "box_threshold": pack.box_threshold,
                "text_threshold": pack.text_threshold,
            }
            for pack in prompt_packs
        ],
        "family_caps": family_caps,
        "family_quota": family_quota,
        "family_nms_iou": family_nms_iou,
        "family_to_id": {**{"unknown-family": 0}, **FAMILY_TO_ID},
        "id_to_family": {**{0: "unknown-family"}, **ID_TO_FAMILY},
        "token_to_id": {**{"unknown": 0}, **TOKEN_TO_ID},
        "id_to_token": {**{0: "unknown"}, **ID_TO_TOKEN},
        "token_to_family": TOKEN_TO_FAMILY,
        "stats": {
            **stats,
            "avg_raw_props_per_frame": avg_raw,
            "avg_final_props_per_frame": avg_final,
            "avg_valid_rows_per_frame": avg_valid,
            "final_family_counter": dict(family_counter),
            "final_token_counter": dict(token_counter),
            "raw_pack_counter": dict(pack_counter),
        },
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[5/5] Done.")
    print(f"  Saved pkl : {output_pkl}")
    print(f"  Saved meta: {output_meta}")
    if args.vis_dir and args.vis_samples > 0:
        print(f"  Saved vis : {args.vis_dir}")
    print("  Summary:")
    print(f"    Total clips               : {stats['total_clips']}")
    print(f"    Total frames              : {stats['total_frames']}")
    print(f"    Frames with no raw props  : {stats['frames_no_raw']}")
    print(f"    Frames with no final props: {stats['frames_no_final']}")
    print(f"    Avg raw props / frame     : {avg_raw:.3f}")
    print(f"    Avg final props / frame   : {avg_final:.3f}")
    print(f"    Avg valid rows / frame    : {avg_valid:.3f} (M={args.global_topk})")


if __name__ == "__main__":
    main()
