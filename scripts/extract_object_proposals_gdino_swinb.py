#!/usr/bin/env python3
"""
Offline object proposal extraction for CAFE using GroundingDINO-B (Swin-B).

Pipeline (per frame):
1) Run four semantic prompt packs independently.
2) Map predicted phrases to stable semantic families.
3) Keep only proposals intersecting expanded person boxes from gt_tracks.pkl.
4) Class-aware NMS (by family).
5) Containment suppression for nested duplicate boxes.
6) Family-level caps.
7) Cross-token dedup for near-identical boxes (e.g., phone vs book overlap).
8) Global capped selection with family quota (default M=10), with optional
   family-aware score multipliers to avoid furniture domination.

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
    "service-family",
]
FAMILY_TO_ID = {name: idx + 1 for idx, name in enumerate(FAMILY_ORDER)}
ID_TO_FAMILY = {v: k for k, v in FAMILY_TO_ID.items()}

TOKEN_ORDER = [
    "phone",
    "laptop",
    "book",
    "notebook",
    "paper",
    "cup",
    "bottle",
    "tray",
    "plate",
    "bowl",
    "food container",
    "counter",
    "cashier counter",
    "pickup counter",
    "service counter",
]
TOKEN_TO_ID = {name: idx + 1 for idx, name in enumerate(TOKEN_ORDER)}
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}

TOKEN_TO_FAMILY = {
    "phone": "phone-family",
    "laptop": "study-family",
    "book": "study-family",
    "notebook": "study-family",
    "paper": "study-family",
    "cup": "dining-family",
    "bottle": "dining-family",
    "tray": "dining-family",
    "plate": "dining-family",
    "bowl": "dining-family",
    "food container": "dining-family",
    "counter": "service-family",
    "cashier counter": "service-family",
    "pickup counter": "service-family",
    "service counter": "service-family",
}

# Phrase aliases -> canonical token
ALIAS_TO_TOKEN = {
    "handheld phone": "phone",
    "phone in hand": "phone",
    "raised phone": "phone",
    "selfie phone": "phone",
    "small handheld phone device": "phone",
    "mobile phone": "phone",
    "phone": "phone",
    "open laptop on table": "laptop",
    "laptop in use": "laptop",
    "laptop": "laptop",
    "open book": "book",
    "book in hand": "book",
    "book": "book",
    "study material": "book",
    "notebook on table": "notebook",
    "notebook": "notebook",
    "paper on table": "paper",
    "paper": "paper",
    "cup on table": "cup",
    "cup": "cup",
    "bottle on table": "bottle",
    "bottle": "bottle",
    "tray on table": "tray",
    "tray": "tray",
    "plate on table": "plate",
    "plate": "plate",
    "bowl on table": "bowl",
    "bowl": "bowl",
    "food container on table": "food container",
    "dining item on table": "food container",
    "food container": "food container",
    "ordering counter": "counter",
    "cashier area": "cashier counter",
    "pickup area": "pickup counter",
    "service desk": "service counter",
    "counter": "counter",
    "cashier counter": "cashier counter",
    "pickup counter": "pickup counter",
    "service counter": "service counter",
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
        "--person_tracks_pkl",
        type=str,
        default="",
        help="Path to person track pkl. Default: <data_root>/gt_tracks.pkl",
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
        "--phone_max_area",
        type=float,
        default=0.06,
        help="Drop phone-family boxes above this normalized area to avoid person-level boxes.",
    )
    parser.add_argument(
        "--person_expand_ratio",
        type=float,
        default=1.2,
        help="Expand each person bbox around center by this ratio before overlap filtering.",
    )
    parser.add_argument(
        "--person_min_intersection",
        type=float,
        default=0.0,
        help="Minimum intersection area with any expanded person box to keep object box.",
    )
    parser.add_argument(
        "--slot_iou_thr",
        type=float,
        default=0.88,
        help="During final truncation, proposals with IoU >= this threshold share one slot.",
    )
    parser.add_argument(
        "--subset_containment_thr",
        type=float,
        default=0.98,
        help="Containment threshold for subset rule: inter / smaller_area.",
    )
    parser.add_argument(
        "--subset_area_ratio_thr",
        type=float,
        default=1.03,
        help="Minimum area ratio (larger/smaller) for subset rule to trigger.",
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
    parser.add_argument("--box_th_b", type=float, default=0.25)
    parser.add_argument("--box_th_c", type=float, default=0.27)
    parser.add_argument("--box_th_d", type=float, default=0.35)
    parser.add_argument("--text_th_a", type=float, default=0.20)
    parser.add_argument("--text_th_b", type=float, default=0.20)
    parser.add_argument("--text_th_c", type=float, default=0.22)
    parser.add_argument("--text_th_d", type=float, default=0.25)

    # Family caps
    parser.add_argument("--cap_phone", type=int, default=4)
    parser.add_argument("--cap_study", type=int, default=6)
    parser.add_argument("--cap_dining", type=int, default=6)
    parser.add_argument("--cap_service", type=int, default=2)

    # Global family quota (used before final score fill-up)
    parser.add_argument("--quota_phone", type=int, default=1)
    parser.add_argument("--quota_study", type=int, default=2)
    parser.add_argument("--quota_dining", type=int, default=2)
    parser.add_argument("--quota_service", type=int, default=1)

    # Family-specific NMS IoU
    parser.add_argument("--nms_iou_phone", type=float, default=0.50)
    parser.add_argument("--nms_iou_study", type=float, default=0.50)
    parser.add_argument("--nms_iou_dining", type=float, default=0.50)
    parser.add_argument("--nms_iou_service", type=float, default=0.40)
    parser.add_argument(
        "--containment_ratio_thr",
        type=float,
        default=0.90,
        help="If intersection / smaller_area >= threshold, treat as nested duplicate.",
    )
    parser.add_argument(
        "--containment_area_ratio",
        type=float,
        default=1.8,
        help="Require larger_area >= smaller_area * ratio for containment suppression.",
    )
    parser.add_argument(
        "--containment_score_margin",
        type=float,
        default=0.05,
        help="For non-phone classes, remove larger box only when score_big <= score_small + margin.",
    )
    parser.add_argument(
        "--cross_token_iou_thr",
        type=float,
        default=0.82,
        help="Cross-token dedup IoU threshold for near-identical boxes.",
    )
    parser.add_argument(
        "--cross_token_area_ratio_thr",
        type=float,
        default=0.65,
        help="Cross-token dedup area similarity threshold: min(area)/max(area).",
    )
    parser.add_argument(
        "--cross_token_center_dist_thr",
        type=float,
        default=0.08,
        help="Cross-token dedup center-distance threshold in normalized image space.",
    )

    # Family score multipliers for final ranking (used before quota+topk).
    # <1 suppresses dominant families, >1 boosts sparse ones.
    parser.add_argument("--score_mul_phone", type=float, default=1.05)
    parser.add_argument("--score_mul_study", type=float, default=1.10)
    parser.add_argument("--score_mul_dining", type=float, default=1.10)
    parser.add_argument("--score_mul_service", type=float, default=1.00)
    parser.add_argument("--token_mul_phone", type=float, default=1.35)
    parser.add_argument("--token_mul_study", type=float, default=1.20)
    parser.add_argument("--token_mul_dining", type=float, default=1.10)
    parser.add_argument("--token_mul_service", type=float, default=1.05)

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


def resolve_person_tracks_path(args: argparse.Namespace, data_root: Path) -> Path:
    if args.person_tracks_pkl:
        return Path(args.person_tracks_pkl)
    return data_root / "gt_tracks.pkl"


def load_person_tracks(person_tracks_path: Path) -> Dict:
    if not person_tracks_path.exists():
        raise FileNotFoundError(f"person_tracks_pkl does not exist: {person_tracks_path}")
    with open(person_tracks_path, "rb") as f:
        tracks = pickle.load(f)
    if not isinstance(tracks, dict):
        raise TypeError(
            f"person_tracks_pkl should be dict, got {type(tracks)} from {person_tracks_path}"
        )
    return tracks


def _clip_frame_tracks_to_array(clip_tracks, fid: int) -> np.ndarray:
    if isinstance(clip_tracks, dict):
        frame_tracks = clip_tracks.get(fid, [])
    elif isinstance(clip_tracks, list):
        if fid < 0 or fid >= len(clip_tracks):
            frame_tracks = []
        else:
            frame_tracks = clip_tracks[fid]
    else:
        frame_tracks = []

    if frame_tracks is None or len(frame_tracks) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.asarray(frame_tracks, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return np.zeros((0, 5), dtype=np.float32)
    return arr[:, :5]


def _sanitize_xyxy(x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float, float, float]]:
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_expanded_person_boxes_xyxy(
    person_tracks: Dict,
    vid: int,
    cid: int,
    fid: int,
    expand_ratio: float,
) -> List[Tuple[float, float, float, float]]:
    clip_tracks = person_tracks.get((vid, cid))
    if clip_tracks is None:
        return []
    tracks_arr = _clip_frame_tracks_to_array(clip_tracks, fid=fid)
    if tracks_arr.shape[0] == 0:
        return []

    ratio = max(float(expand_ratio), 0.0)
    boxes: List[Tuple[float, float, float, float]] = []
    for row in tracks_arr:
        # row format in gt_tracks.pkl: [id, x1, y1, x2, y2], normalized xyxy.
        parsed = _sanitize_xyxy(float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        if parsed is None:
            continue
        x1, y1, x2, y2 = parsed
        if ratio <= 1.0:
            boxes.append((x1, y1, x2, y2))
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1) * ratio
        h = (y2 - y1) * ratio
        ex1 = max(0.0, cx - 0.5 * w)
        ey1 = max(0.0, cy - 0.5 * h)
        ex2 = min(1.0, cx + 0.5 * w)
        ey2 = min(1.0, cy + 0.5 * h)
        expanded = _sanitize_xyxy(ex1, ey1, ex2, ey2)
        if expanded is not None:
            boxes.append(expanded)
    return boxes


def build_prompt_packs(args: argparse.Namespace) -> List[PromptPack]:
    return [
        PromptPack(
            name="A",
            pack_id=1,
            caption="phone in hand . handheld phone . raised phone . selfie phone . small handheld phone device .",
            box_threshold=float(getattr(args, "box_th_a", 0.25)),
            text_threshold=float(getattr(args, "text_th_a", 0.20)),
        ),
        PromptPack(
            name="B",
            pack_id=2,
            caption="open laptop on table . laptop in use . open book . book in hand . notebook on table . paper on table . study material .",
            box_threshold=float(getattr(args, "box_th_b", 0.25)),
            text_threshold=float(getattr(args, "text_th_b", 0.20)),
        ),
        PromptPack(
            name="C",
            pack_id=3,
            caption="cup on table . bottle on table . tray on table . plate on table . bowl on table . food container on table . dining item on table .",
            box_threshold=float(getattr(args, "box_th_c", 0.27)),
            text_threshold=float(getattr(args, "text_th_c", 0.22)),
        ),
        PromptPack(
            name="D",
            pack_id=4,
            caption="ordering counter . cashier area . pickup area . service desk .",
            box_threshold=float(getattr(args, "box_th_d", 0.35)),
            text_threshold=float(getattr(args, "text_th_d", 0.25)),
        ),
    ]


def get_family_caps(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "phone-family": int(getattr(args, "cap_phone", 4)),
        "study-family": int(getattr(args, "cap_study", 6)),
        "dining-family": int(getattr(args, "cap_dining", 6)),
        "service-family": int(getattr(args, "cap_service", 2)),
    }


def get_family_quota(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "phone-family": int(getattr(args, "quota_phone", 1)),
        "study-family": int(getattr(args, "quota_study", 2)),
        "dining-family": int(getattr(args, "quota_dining", 2)),
        "service-family": int(getattr(args, "quota_service", 1)),
    }


def get_family_nms_iou(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "phone-family": float(getattr(args, "nms_iou_phone", 0.50)),
        "study-family": float(getattr(args, "nms_iou_study", 0.50)),
        "dining-family": float(getattr(args, "nms_iou_dining", 0.50)),
        "service-family": float(getattr(args, "nms_iou_service", 0.40)),
    }


def get_family_score_multipliers(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "phone-family": float(getattr(args, "score_mul_phone", 1.05)),
        "study-family": float(getattr(args, "score_mul_study", 1.10)),
        "dining-family": float(getattr(args, "score_mul_dining", 1.10)),
        "service-family": float(getattr(args, "score_mul_service", 1.00)),
    }


def get_token_score_multipliers(args: argparse.Namespace) -> Dict[str, float]:
    study_mul = float(getattr(args, "token_mul_study", 1.20))
    dining_mul = float(getattr(args, "token_mul_dining", 1.10))
    service_mul = float(getattr(args, "token_mul_service", 1.05))
    return {
        "phone": float(getattr(args, "token_mul_phone", 1.35)),
        "laptop": study_mul,
        "book": study_mul,
        "notebook": study_mul,
        "paper": study_mul,
        "cup": dining_mul,
        "bottle": dining_mul,
        "tray": dining_mul,
        "plate": dining_mul,
        "bowl": dining_mul,
        "food container": dining_mul,
        "counter": service_mul,
        "cashier counter": service_mul,
        "pickup counter": service_mul,
        "service counter": service_mul,
    }


def run_packs_for_frame(
    model,
    image_path: Path,
    prompt_packs: Sequence[PromptPack],
    min_area: float,
    allow_unknown_phrase: bool,
    device: str,
    phone_max_area: float = 0.06,
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
            device=device,
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
            if (
                family == "phone-family"
                and phone_max_area > 0.0
                and area > phone_max_area
            ):
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


def _intersection_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def filter_proposals_by_person_overlap(
    proposals: Sequence[dict],
    person_boxes_xyxy: Sequence[Tuple[float, float, float, float]],
    min_intersection: float,
) -> List[dict]:
    if not proposals:
        return []
    if not person_boxes_xyxy:
        return []
    thr = max(float(min_intersection), 0.0)

    kept: List[dict] = []
    for p in proposals:
        box = (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"]))
        matched = False
        for person_box in person_boxes_xyxy:
            inter = _intersection_xyxy(box, person_box)
            if inter > thr:
                matched = True
                break
        if matched:
            kept.append(p)
    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept


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


def _intersection_area(a: dict, b: dict) -> float:
    ix1 = max(float(a["x1"]), float(b["x1"]))
    iy1 = max(float(a["y1"]), float(b["y1"]))
    ix2 = min(float(a["x2"]), float(b["x2"]))
    iy2 = min(float(a["y2"]), float(b["y2"]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _box_area(p: dict) -> float:
    return max(0.0, float(p["x2"]) - float(p["x1"])) * max(
        0.0, float(p["y2"]) - float(p["y1"])
    )


def _iou(a: dict, b: dict) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    area_a = _box_area(a)
    area_b = _box_area(b)
    union = max(area_a + area_b - inter, 1e-12)
    return inter / union


def _center_dist(a: dict, b: dict) -> float:
    ax = 0.5 * (float(a["x1"]) + float(a["x2"]))
    ay = 0.5 * (float(a["y1"]) + float(a["y2"]))
    bx = 0.5 * (float(b["x1"]) + float(b["x2"]))
    by = 0.5 * (float(b["y1"]) + float(b["y2"]))
    dx = ax - bx
    dy = ay - by
    return float((dx * dx + dy * dy) ** 0.5)


def _is_counter_or_table_token(token: str) -> bool:
    t = str(token).strip().lower()
    return ("counter" in t) or (t == "table")


def _subset_relation(
    a: dict,
    b: dict,
    containment_thr: float = 0.98,
    area_ratio_thr: float = 1.05,
) -> Tuple[bool, int]:
    area_a = _box_area(a)
    area_b = _box_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return False, -1
    inter = _intersection_area(a, b)
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    if smaller <= 0.0:
        return False, -1
    containment = inter / smaller
    if containment < containment_thr:
        return False, -1
    if larger < smaller * area_ratio_thr:
        return False, -1
    # return index of the smaller box: 0 means a is smaller, 1 means b is smaller.
    return True, (0 if area_a <= area_b else 1)


def apply_containment_suppression(
    proposals: Sequence[dict],
    containment_ratio_thr: float,
    containment_area_ratio: float,
    containment_score_margin: float,
) -> List[dict]:
    if not proposals:
        return []
    if containment_ratio_thr <= 0.0 or containment_area_ratio <= 1.0:
        return sorted(proposals, key=lambda p: p["score"], reverse=True)

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for p in proposals:
        grouped[(p.get("family", ""), p.get("token", ""))].append(p)

    kept: List[dict] = []
    for (family, token_name), items in grouped.items():
        if len(items) <= 1:
            kept.extend(items)
            continue
        to_remove = set()
        areas = [
            max(0.0, float(p["x2"]) - float(p["x1"])) * max(0.0, float(p["y2"]) - float(p["y1"]))
            for p in items
        ]

        for i in range(len(items)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(items)):
                if j in to_remove:
                    continue
                ai = areas[i]
                aj = areas[j]
                if ai <= 0.0 or aj <= 0.0:
                    continue
                inter = _intersection_area(items[i], items[j])
                smaller = min(ai, aj)
                larger = max(ai, aj)
                if smaller <= 0.0:
                    continue
                containment = inter / smaller
                if containment < containment_ratio_thr:
                    continue
                if larger < smaller * containment_area_ratio:
                    continue

                big_idx = i if ai >= aj else j
                small_idx = j if big_idx == i else i
                score_big = float(items[big_idx]["score"])
                score_small = float(items[small_idx]["score"])
                token_exempt = _is_counter_or_table_token(token_name)
                is_subset_pair, small_pos = _subset_relation(
                    items[i], items[j], containment_thr=0.98, area_ratio_thr=1.05
                )

                if is_subset_pair and not token_exempt:
                    # For strict subset duplicate, always keep the smaller box.
                    if small_pos == 0:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                    continue

                if family == "phone-family":
                    to_remove.add(big_idx)
                elif score_big <= score_small + containment_score_margin:
                    to_remove.add(big_idx)

        for idx, p in enumerate(items):
            if idx not in to_remove:
                kept.append(p)

    kept.sort(key=lambda p: p["score"], reverse=True)
    return kept


def apply_cross_token_dedup(
    proposals: Sequence[dict],
    iou_thr: float,
    area_ratio_thr: float,
    center_dist_thr: float,
) -> List[dict]:
    if not proposals:
        return []
    if iou_thr <= 0.0 or area_ratio_thr <= 0.0 or center_dist_thr <= 0.0:
        return sorted(
            proposals, key=lambda p: float(p.get("rank_score", p["score"])), reverse=True
        )

    sorted_props = sorted(
        proposals, key=lambda p: float(p.get("rank_score", p["score"])), reverse=True
    )
    kept: List[dict] = []

    for cand in sorted_props:
        is_dup = False
        area_c = _box_area(cand)
        if area_c <= 0.0:
            continue

        for ref in kept:
            if cand.get("token") == ref.get("token"):
                continue
            iou = _iou(cand, ref)
            if iou < iou_thr:
                continue
            area_r = _box_area(ref)
            if area_r <= 0.0:
                continue
            area_ratio = min(area_c, area_r) / max(area_c, area_r)
            if area_ratio < area_ratio_thr:
                continue
            if _center_dist(cand, ref) > center_dist_thr:
                continue
            is_dup = True
            break

        if not is_dup:
            kept.append(cand)

    kept.sort(key=lambda p: float(p.get("rank_score", p["score"])), reverse=True)
    return kept


def apply_global_subset_suppression(
    proposals: Sequence[dict],
    subset_containment_thr: float,
    subset_area_ratio_thr: float,
) -> List[dict]:
    if not proposals:
        return []
    if subset_containment_thr <= 0.0 or subset_area_ratio_thr <= 1.0:
        return sorted(proposals, key=lambda p: float(p.get("rank_score", p["score"])), reverse=True)

    items = list(proposals)
    to_remove = set()

    for i in range(len(items)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(items)):
            if j in to_remove:
                continue
            pi = items[i]
            pj = items[j]
            if _is_counter_or_table_token(pi.get("token", "")) or _is_counter_or_table_token(
                pj.get("token", "")
            ):
                continue

            is_subset_pair, small_pos = _subset_relation(
                pi,
                pj,
                containment_thr=subset_containment_thr,
                area_ratio_thr=subset_area_ratio_thr,
            )
            if not is_subset_pair:
                continue

            # Always keep the smaller box for subset duplicates.
            if small_pos == 0:
                to_remove.add(j)
            else:
                to_remove.add(i)
                break

    kept = [p for idx, p in enumerate(items) if idx not in to_remove]
    kept.sort(key=lambda p: float(p.get("rank_score", p["score"])), reverse=True)
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


def apply_family_score_multipliers(
    proposals: Sequence[dict],
    family_score_mult: Dict[str, float],
) -> List[dict]:
    out: List[dict] = []
    for proposal in proposals:
        p = dict(proposal)
        mult = float(family_score_mult.get(p["family"], 1.0))
        p["rank_score"] = float(p["score"]) * mult
        out.append(p)
    out.sort(key=lambda x: x.get("rank_score", x["score"]), reverse=True)
    return out


def apply_token_score_multipliers(
    proposals: Sequence[dict],
    token_score_mult: Dict[str, float],
) -> List[dict]:
    out: List[dict] = []
    for proposal in proposals:
        p = dict(proposal)
        cur = float(p.get("rank_score", p["score"]))
        mult = float(token_score_mult.get(p.get("token", ""), 1.0))
        p["rank_score"] = cur * mult
        out.append(p)
    out.sort(key=lambda x: x.get("rank_score", x["score"]), reverse=True)
    return out


def apply_global_quota_and_topk(
    proposals: Sequence[dict],
    family_quota: Dict[str, int],
    global_topk: int,
    slot_iou_thr: float = 0.0,
    subset_containment_thr: float = 0.98,
    subset_area_ratio_thr: float = 1.03,
) -> List[dict]:
    def rank_score(item: dict) -> float:
        return float(item.get("rank_score", item["score"]))

    def try_add_with_slot_policy(candidate: dict, selected_items: List[dict]) -> bool:
        # Returns True only when candidate consumes a new slot.
        if slot_iou_thr <= 0.0:
            selected_items.append(candidate)
            return True

        for idx, ref in enumerate(selected_items):
            cand_exempt = _is_counter_or_table_token(candidate.get("token", ""))
            ref_exempt = _is_counter_or_table_token(ref.get("token", ""))
            is_subset_pair, small_pos = _subset_relation(
                candidate,
                ref,
                containment_thr=subset_containment_thr,
                area_ratio_thr=subset_area_ratio_thr,
            )

            # For strict subset duplicates, keep smaller one unless token is exempt.
            if is_subset_pair and not (cand_exempt or ref_exempt):
                if small_pos == 0:
                    selected_items[idx] = candidate
                return False

            if slot_iou_thr <= 0.0:
                continue
            if _iou(candidate, ref) < slot_iou_thr:
                continue

            # Non-subset or exempt tokens: share one slot, keep existing selected one.
            return False

        selected_items.append(candidate)
        return True

    if global_topk <= 0:
        return sorted(proposals, key=rank_score, reverse=True)

    by_family: Dict[str, List[dict]] = defaultdict(list)
    for proposal in proposals:
        by_family[proposal["family"]].append(proposal)
    for family in by_family.keys():
        by_family[family].sort(key=rank_score, reverse=True)

    selected: List[dict] = []
    leftovers: List[dict] = []

    for family in FAMILY_ORDER:
        items = by_family.get(family, [])
        quota = max(family_quota.get(family, 0), 0)
        taken = 0
        for item in items:
            if len(selected) >= global_topk:
                leftovers.append(item)
                continue
            if taken >= quota:
                leftovers.append(item)
                continue
            if try_add_with_slot_policy(item, selected):
                taken += 1

    known = set(FAMILY_ORDER)
    for family, items in by_family.items():
        if family not in known:
            leftovers.extend(items)

    if len(selected) < global_topk and leftovers:
        leftovers.sort(key=rank_score, reverse=True)
        for item in leftovers:
            if len(selected) >= global_topk:
                break
            try_add_with_slot_policy(item, selected)

    if len(selected) > global_topk:
        selected = sorted(selected, key=rank_score, reverse=True)[:global_topk]

    selected.sort(key=rank_score, reverse=True)
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
    family_score_mult = get_family_score_multipliers(args)
    token_score_mult = get_token_score_multipliers(args)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    person_tracks_path = resolve_person_tracks_path(args, data_root)
    print(f"[1/6] Loading person tracks from: {person_tracks_path}")
    person_tracks = load_person_tracks(person_tracks_path)

    print(f"[2/6] Loading GroundingDINO model on device={args.device} ...")
    model = load_model(args.config_path, args.checkpoint_path, device=args.device)

    print(f"[3/6] Scanning clips under: {data_root}")
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

    print("[4/6] Running extraction ...")
    tracks: Dict[Tuple[int, int], List[np.ndarray]] = {}

    stats = {
        "total_clips": len(clips),
        "total_frames": 0,
        "frames_no_raw": 0,
        "frames_no_person_overlap": 0,
        "frames_no_final": 0,
        "raw_props_total": 0,
        "person_boxes_total": 0,
        "person_gate_props_total": 0,
        "person_filtered_out_total": 0,
        "nms_props_total": 0,
        "contain_props_total": 0,
        "capped_props_total": 0,
        "subset_props_total": 0,
        "cross_dedup_props_total": 0,
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
                    device=args.device,
                    phone_max_area=args.phone_max_area,
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

            person_boxes_xyxy = get_expanded_person_boxes_xyxy(
                person_tracks=person_tracks,
                vid=vid,
                cid=cid,
                fid=fid,
                expand_ratio=args.person_expand_ratio,
            )
            stats["person_boxes_total"] += len(person_boxes_xyxy)

            person_gated_props = filter_proposals_by_person_overlap(
                raw_props,
                person_boxes_xyxy=person_boxes_xyxy,
                min_intersection=args.person_min_intersection,
            )
            stats["person_gate_props_total"] += len(person_gated_props)
            stats["person_filtered_out_total"] += max(
                len(raw_props) - len(person_gated_props), 0
            )
            if not person_gated_props:
                stats["frames_no_person_overlap"] += 1
                stats["frames_no_final"] += 1
                continue

            nms_props = apply_class_aware_nms(person_gated_props, family_nms_iou)
            stats["nms_props_total"] += len(nms_props)

            contain_props = apply_containment_suppression(
                nms_props,
                containment_ratio_thr=args.containment_ratio_thr,
                containment_area_ratio=args.containment_area_ratio,
                containment_score_margin=args.containment_score_margin,
            )
            stats["contain_props_total"] += len(contain_props)

            capped_props = apply_family_caps(contain_props, family_caps)
            stats["capped_props_total"] += len(capped_props)

            scored_props = apply_family_score_multipliers(capped_props, family_score_mult)
            scored_props = apply_token_score_multipliers(scored_props, token_score_mult)
            subset_props = apply_global_subset_suppression(
                scored_props,
                subset_containment_thr=args.subset_containment_thr,
                subset_area_ratio_thr=args.subset_area_ratio_thr,
            )
            stats["subset_props_total"] += len(subset_props)
            dedup_props = apply_cross_token_dedup(
                subset_props,
                iou_thr=args.cross_token_iou_thr,
                area_ratio_thr=args.cross_token_area_ratio_thr,
                center_dist_thr=args.cross_token_center_dist_thr,
            )
            stats["cross_dedup_props_total"] += len(dedup_props)
            final_props = apply_global_quota_and_topk(
                dedup_props,
                family_quota,
                args.global_topk,
                slot_iou_thr=args.slot_iou_thr,
                subset_containment_thr=args.subset_containment_thr,
                subset_area_ratio_thr=args.subset_area_ratio_thr,
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

    print(f"[5/6] Saving outputs ...")
    with open(output_pkl, "wb") as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

    avg_raw = (
        float(stats["raw_props_total"]) / max(stats["total_frames"], 1)
    )
    avg_person_gate = (
        float(stats["person_gate_props_total"]) / max(stats["total_frames"], 1)
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
            "person_tracks_pkl": str(person_tracks_path),
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
        "box_filters": {
            "min_area": args.min_area,
            "phone_max_area": args.phone_max_area,
            "person_expand_ratio": args.person_expand_ratio,
            "person_min_intersection": args.person_min_intersection,
        },
        "containment_suppression": {
            "ratio_thr": args.containment_ratio_thr,
            "area_ratio": args.containment_area_ratio,
            "score_margin": args.containment_score_margin,
        },
        "cross_token_dedup": {
            "iou_thr": args.cross_token_iou_thr,
            "area_ratio_thr": args.cross_token_area_ratio_thr,
            "center_dist_thr": args.cross_token_center_dist_thr,
        },
        "slot_dedup": {
            "slot_iou_thr": args.slot_iou_thr,
        },
        "subset_rule": {
            "subset_containment_thr": args.subset_containment_thr,
            "subset_area_ratio_thr": args.subset_area_ratio_thr,
        },
        "family_score_multipliers": family_score_mult,
        "token_score_multipliers": token_score_mult,
        "family_to_id": {**{"unknown-family": 0}, **FAMILY_TO_ID},
        "id_to_family": {**{0: "unknown-family"}, **ID_TO_FAMILY},
        "token_to_id": {**{"unknown": 0}, **TOKEN_TO_ID},
        "id_to_token": {**{0: "unknown"}, **ID_TO_TOKEN},
        "token_to_family": TOKEN_TO_FAMILY,
        "stats": {
            **stats,
            "avg_raw_props_per_frame": avg_raw,
            "avg_person_gate_props_per_frame": avg_person_gate,
            "avg_final_props_per_frame": avg_final,
            "avg_valid_rows_per_frame": avg_valid,
            "final_family_counter": dict(family_counter),
            "final_token_counter": dict(token_counter),
            "raw_pack_counter": dict(pack_counter),
        },
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[6/6] Done.")
    print(f"  Saved pkl : {output_pkl}")
    print(f"  Saved meta: {output_meta}")
    if args.vis_dir and args.vis_samples > 0:
        print(f"  Saved vis : {args.vis_dir}")
    print("  Summary:")
    print(f"    Total clips               : {stats['total_clips']}")
    print(f"    Total frames              : {stats['total_frames']}")
    print(f"    Frames with no raw props  : {stats['frames_no_raw']}")
    print(f"    Frames with no person hit : {stats['frames_no_person_overlap']}")
    print(f"    Frames with no final props: {stats['frames_no_final']}")
    print(f"    Avg raw props / frame     : {avg_raw:.3f}")
    print(f"    Avg person-gated / frame  : {avg_person_gate:.3f}")
    print(f"    Avg final props / frame   : {avg_final:.3f}")
    print(f"    Avg valid rows / frame    : {avg_valid:.3f} (M={args.global_topk})")


if __name__ == "__main__":
    main()
