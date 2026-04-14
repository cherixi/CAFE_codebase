#!/usr/bin/env python3
"""Offline object proposal extraction for CAFE using YOLOv8 LocalMix."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._yolov8_extract_utils import (
    FAMILY_ORDER as YOLO_FAMILY_ORDER,
    FAMILY_TO_ID as YOLO_FAMILY_TO_ID,
    ID_TO_FAMILY as YOLO_ID_TO_FAMILY,
    ID_TO_TOKEN as YOLO_ID_TO_TOKEN,
    PromptPack,
    TOKEN_ORDER as YOLO_TOKEN_ORDER,
    TOKEN_TO_FAMILY as YOLO_TOKEN_TO_FAMILY,
    TOKEN_TO_ID as YOLO_TOKEN_TO_ID,
    build_prompt_packs_from_args,
    load_yolo_detector_from_args,
    run_yolo_on_paths,
    run_yolo_on_pil_images,
    split_prompt_packs_for_localmix as split_yolo_prompt_packs_for_localmix,
)

gdino_load_image = None
gdino_load_model = None
gdino_predict = None
OFFICIAL_GDINO_AVAILABLE = False

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    HF_GDINO_AVAILABLE = True
except ImportError:
    AutoModelForZeroShotObjectDetection = None
    AutoProcessor = None
    HF_GDINO_AVAILABLE = False


FAMILY_ORDER = list(YOLO_FAMILY_ORDER)
FAMILY_TO_ID = dict(YOLO_FAMILY_TO_ID)
ID_TO_FAMILY = dict(YOLO_ID_TO_FAMILY)
TOKEN_ORDER = list(YOLO_TOKEN_ORDER)
TOKEN_TO_ID = dict(YOLO_TOKEN_TO_ID)
ID_TO_TOKEN = dict(YOLO_ID_TO_TOKEN)
TOKEN_TO_FAMILY = dict(YOLO_TOKEN_TO_FAMILY)
ALIAS_TO_TOKEN: Dict[str, str] = {}
ALIAS_KEYS_SORTED: List[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract offline object proposals from CAFE using YOLOv8 LocalMix."
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
        help="Output pickle path. Default: <data_root>/object_tracks_yolov8_localmix.pkl",
    )
    parser.add_argument(
        "--output_meta",
        type=str,
        default="",
        help="Output meta json path. Default: <data_root>/object_tracks_yolov8_localmix_meta.json",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of shard workers (typically number of GPUs).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Current worker rank in [0, world_size-1].",
    )
    parser.add_argument(
        "--merge_shards",
        action="store_true",
        help="Merge shard outputs into final output_pkl/output_meta and exit.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="yolo",
        choices=["yolo"],
        help="Detection backend for this script. Fixed to YOLO.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Unused placeholder for compatibility.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Unused placeholder for compatibility.",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="",
        help="Unused placeholder for compatibility.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="",
        help="Unused placeholder for compatibility.",
    )
    parser.add_argument(
        "--hf_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Unused placeholder for compatibility.",
    )
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--hf_pack_batch",
        dest="hf_pack_batch",
        action="store_true",
        help="Batch HF prompt packs with identical thresholds in one forward pass (recommended).",
    )
    batch_group.add_argument(
        "--hf_no_pack_batch",
        dest="hf_pack_batch",
        action="store_false",
        help="Disable HF pack batching and run one pack per forward pass.",
    )
    parser.set_defaults(hf_pack_batch=True)
    parser.add_argument(
        "--hf_compile",
        action="store_true",
        help="Unused placeholder for compatibility.",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8x.pt",
        help="YOLO model weights path or model alias.",
    )
    parser.add_argument(
        "--yolo_imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.01,
        help="YOLO confidence threshold before pack-level thresholding.",
    )
    parser.add_argument(
        "--yolo_iou",
        type=float,
        default=0.70,
        help="YOLO internal NMS IoU threshold.",
    )
    parser.add_argument(
        "--yolo_max_det",
        type=int,
        default=300,
        help="YOLO max detections per image.",
    )
    parser.add_argument(
        "--yolo_batch_size",
        type=int,
        default=8,
        help="YOLO inference batch size.",
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
        "--frame_batch_size",
        type=int,
        default=4,
        help="Frames per inference batch.",
    )
    oom_group = parser.add_mutually_exclusive_group()
    oom_group.add_argument(
        "--auto_reduce_batch_on_oom",
        dest="auto_reduce_batch_on_oom",
        action="store_true",
        help="On CUDA OOM during frame batching, automatically halve batch size and retry.",
    )
    oom_group.add_argument(
        "--no_auto_reduce_batch_on_oom",
        dest="auto_reduce_batch_on_oom",
        action="store_false",
        help="Disable auto OOM recovery for frame batching.",
    )
    parser.set_defaults(auto_reduce_batch_on_oom=True)
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
        "--local_small_branch_mode",
        type=str,
        default="full",
        choices=["off", "rescue", "full"],
        help=(
            "Local small-object branch mode: "
            "off=disable local branch, "
            "rescue=run local A/B/C only when global A/B/C is empty, "
            "full=global D + local A/B/C."
        ),
    )
    parser.add_argument(
        "--local_person_expand_ratio",
        type=float,
        default=1.35,
        help="Person box horizontal expand ratio (left/right) used to build local crop ROIs.",
    )
    parser.add_argument(
        "--local_person_expand_ratio_y",
        type=float,
        default=1.00,
        help="Person box vertical expand ratio (up/down) used to build local crop ROIs.",
    )
    parser.add_argument(
        "--local_merge_iou_thr",
        type=float,
        default=0.42,
        help="Merge two expanded person boxes into one local ROI when IoU >= this threshold.",
    )
    parser.add_argument(
        "--local_merge_center_thr",
        type=float,
        default=0.13,
        help="Merge two expanded person boxes into one local ROI when center distance <= this threshold.",
    )
    parser.add_argument(
        "--local_merge_center_aux_iou_thr",
        type=float,
        default=0.05,
        help=(
            "Auxiliary IoU threshold for center-distance merge. "
            "Center-based merge triggers only when center_dist<=thr and IoU>=this value."
        ),
    )
    parser.add_argument(
        "--local_roi_dedup_iou",
        type=float,
        default=0.80,
        help="Deduplicate merged local ROIs by IoU threshold.",
    )
    parser.add_argument(
        "--local_max_person_crops",
        type=int,
        default=8,
        help="Maximum local ROIs per frame for A/B/C local detection.",
    )
    local_fallback_group = parser.add_mutually_exclusive_group()
    local_fallback_group.add_argument(
        "--local_fallback_global_when_no_person",
        dest="local_fallback_global_when_no_person",
        action="store_true",
        help="When no person ROI exists, run A/B/C once on full image as fallback.",
    )
    local_fallback_group.add_argument(
        "--no_local_fallback_global_when_no_person",
        dest="local_fallback_global_when_no_person",
        action="store_false",
        help="Disable A/B/C full-image fallback when no person ROI exists.",
    )
    parser.set_defaults(local_fallback_global_when_no_person=True)
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


def ensure_official_backend_imported() -> None:
    global gdino_load_image, gdino_load_model, gdino_predict, OFFICIAL_GDINO_AVAILABLE
    if OFFICIAL_GDINO_AVAILABLE:
        return
    try:
        from groundingdino.util.inference import (
            load_image as _gdino_load_image,
            load_model as _gdino_load_model,
            predict as _gdino_predict,
        )
    except ImportError as exc:
        raise ImportError(
            "Official GroundingDINO backend unavailable. "
            "Install groundingdino package and use matching environment."
        ) from exc

    gdino_load_image = _gdino_load_image
    gdino_load_model = _gdino_load_model
    gdino_predict = _gdino_predict
    OFFICIAL_GDINO_AVAILABLE = True


def resolve_hf_torch_dtype(device: str, hf_dtype: str) -> Optional[torch.dtype]:
    dev = str(device).strip().lower()
    req = str(hf_dtype).strip().lower()
    if req == "auto":
        if dev.startswith("cuda"):
            return torch.float16
        return torch.float32
    if req == "float16":
        return torch.float16
    if req == "bfloat16":
        return torch.bfloat16
    if req == "float32":
        return torch.float32
    raise ValueError(f"Unsupported --hf_dtype: {hf_dtype}")


def build_hf_autocast_context(device: str, amp_dtype: Optional[torch.dtype]):
    dev = str(device).strip().lower()
    if not dev.startswith("cuda"):
        return contextlib.nullcontext()
    if amp_dtype not in (torch.float16, torch.bfloat16):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def load_detector(
    backend: str,
    device: str,
    config_path: str = "",
    checkpoint_path: str = "",
    hf_model_id: str = "IDEA-Research/grounding-dino-base",
    hf_cache_dir: str = "",
    hf_dtype: str = "auto",
    hf_pack_batch: bool = True,
    hf_compile: bool = False,
) -> Dict[str, Any]:
    del backend, device, config_path, checkpoint_path, hf_model_id, hf_cache_dir
    del hf_dtype, hf_pack_batch, hf_compile
    raise RuntimeError("Use load_detector_from_args() for YOLO backend.")


def load_detector_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return load_yolo_detector_from_args(args)


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
    expand_ratio_y: Optional[float] = None,
) -> List[Tuple[float, float, float, float]]:
    clip_tracks = person_tracks.get((vid, cid))
    if clip_tracks is None:
        return []
    tracks_arr = _clip_frame_tracks_to_array(clip_tracks, fid=fid)
    if tracks_arr.shape[0] == 0:
        return []

    ratio_x = max(float(expand_ratio), 0.0)
    ratio_y = (
        ratio_x
        if expand_ratio_y is None
        else max(float(expand_ratio_y), 0.0)
    )
    boxes: List[Tuple[float, float, float, float]] = []
    for row in tracks_arr:
        # row format in gt_tracks.pkl: [id, x1, y1, x2, y2], normalized xyxy.
        parsed = _sanitize_xyxy(float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        if parsed is None:
            continue
        x1, y1, x2, y2 = parsed
        if ratio_x <= 1.0 and ratio_y <= 1.0:
            boxes.append((x1, y1, x2, y2))
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1) * ratio_x
        h = (y2 - y1) * ratio_y
        ex1 = max(0.0, cx - 0.5 * w)
        ey1 = max(0.0, cy - 0.5 * h)
        ex2 = min(1.0, cx + 0.5 * w)
        ey2 = min(1.0, cy + 0.5 * h)
        expanded = _sanitize_xyxy(ex1, ey1, ex2, ey2)
        if expanded is not None:
            boxes.append(expanded)
    return boxes


def split_prompt_packs_for_localmix(
    prompt_packs: Sequence[PromptPack],
) -> Tuple[List[PromptPack], List[PromptPack]]:
    return split_yolo_prompt_packs_for_localmix(prompt_packs)


def _box_iou_xyxy_tuple(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    inter = _intersection_xyxy(a, b)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = max(area_a + area_b - inter, 1e-12)
    return inter / union


def _box_center_dist_xyxy_tuple(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax = 0.5 * (a[0] + a[2])
    ay = 0.5 * (a[1] + a[3])
    bx = 0.5 * (b[0] + b[2])
    by = 0.5 * (b[1] + b[3])
    dx = ax - bx
    dy = ay - by
    return float((dx * dx + dy * dy) ** 0.5)


def build_localmix_rois_from_person_boxes(
    person_boxes_xyxy: Sequence[Tuple[float, float, float, float]],
    merge_iou_thr: float,
    merge_center_thr: float,
    merge_center_aux_iou_thr: float,
    dedup_iou_thr: float,
    max_crops: int,
) -> List[Tuple[float, float, float, float]]:
    boxes = [b for b in person_boxes_xyxy if _sanitize_xyxy(*b) is not None]
    if not boxes:
        return []

    n = len(boxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a_idx: int, b_idx: int) -> None:
        ra = find(a_idx)
        rb = find(b_idx)
        if ra != rb:
            parent[rb] = ra

    iou_thr = max(float(merge_iou_thr), 0.0)
    ctr_thr = max(float(merge_center_thr), 0.0)
    ctr_aux_iou_thr = max(float(merge_center_aux_iou_thr), 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            iou = _box_iou_xyxy_tuple(boxes[i], boxes[j])
            ctr = _box_center_dist_xyxy_tuple(boxes[i], boxes[j])
            merge_by_iou = iou >= iou_thr
            merge_by_center = (ctr <= ctr_thr) and (iou >= ctr_aux_iou_thr)
            if merge_by_iou or merge_by_center:
                union(i, j)

    groups: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for i, b in enumerate(boxes):
        groups[find(i)].append(b)

    merged: List[Tuple[Tuple[float, float, float, float], int, float]] = []
    for members in groups.values():
        x1 = min(m[0] for m in members)
        y1 = min(m[1] for m in members)
        x2 = max(m[2] for m in members)
        y2 = max(m[3] for m in members)
        roi = _sanitize_xyxy(x1, y1, x2, y2)
        if roi is None:
            continue
        area = max(0.0, roi[2] - roi[0]) * max(0.0, roi[3] - roi[1])
        merged.append((roi, len(members), area))

    # Keep larger connected components first.
    merged.sort(key=lambda x: (x[1], x[2]), reverse=True)

    dedup_thr = max(float(dedup_iou_thr), 0.0)
    deduped: List[Tuple[float, float, float, float]] = []
    for roi, _, _ in merged:
        duplicated = False
        for kept in deduped:
            if _box_iou_xyxy_tuple(roi, kept) >= dedup_thr:
                duplicated = True
                break
        if not duplicated:
            deduped.append(roi)

    limit = int(max(max_crops, 0))
    if limit > 0:
        deduped = deduped[:limit]
    else:
        deduped = []
    return deduped


def crop_pil_with_norm_roi(
    pil_image: Image.Image,
    roi_xyxy: Tuple[float, float, float, float],
) -> Tuple[Optional[Image.Image], Optional[Tuple[float, float, float, float]]]:
    w, h = pil_image.size
    if w <= 1 or h <= 1:
        return None, None

    x1, y1, x2, y2 = roi_xyxy
    px1 = int(np.floor(max(0.0, min(1.0, x1)) * w))
    py1 = int(np.floor(max(0.0, min(1.0, y1)) * h))
    px2 = int(np.ceil(max(0.0, min(1.0, x2)) * w))
    py2 = int(np.ceil(max(0.0, min(1.0, y2)) * h))

    px1 = max(0, min(w - 1, px1))
    py1 = max(0, min(h - 1, py1))
    px2 = max(px1 + 1, min(w, px2))
    py2 = max(py1 + 1, min(h, py2))
    if px2 <= px1 or py2 <= py1:
        return None, None

    crop = pil_image.crop((px1, py1, px2, py2))
    eff = (
        float(px1) / float(w),
        float(py1) / float(h),
        float(px2) / float(w),
        float(py2) / float(h),
    )
    return crop, eff


def run_packs_for_pil_images_hf(
    model,
    pil_images: Sequence[Image.Image],
    prompt_packs: Sequence[PromptPack],
    min_area: float,
    allow_unknown_phrase: bool,
    device: str,
    phone_max_area: float = 0.06,
    source_tag: str = "global",
    max_pairs_per_forward: int = 4,
) -> List[List[dict]]:
    del allow_unknown_phrase, device
    return run_yolo_on_pil_images(
        model_bundle=model,
        pil_images=pil_images,
        prompt_packs=prompt_packs,
        min_area=min_area,
        phone_max_area=phone_max_area,
        source_tag=source_tag,
        max_pairs_per_forward=max_pairs_per_forward,
    )


def map_local_props_to_full_image(
    local_props: Sequence[dict],
    roi_xyxy: Tuple[float, float, float, float],
    roi_index: int,
) -> List[dict]:
    rx1, ry1, rx2, ry2 = roi_xyxy
    rw = max(rx2 - rx1, 1e-12)
    rh = max(ry2 - ry1, 1e-12)
    out: List[dict] = []
    for p in local_props:
        gx1 = rx1 + float(p["x1"]) * rw
        gy1 = ry1 + float(p["y1"]) * rh
        gx2 = rx1 + float(p["x2"]) * rw
        gy2 = ry1 + float(p["y2"]) * rh
        box = _sanitize_xyxy(gx1, gy1, gx2, gy2)
        if box is None:
            continue
        q = dict(p)
        q["x1"], q["y1"], q["x2"], q["y2"] = box
        q["source"] = "local"
        q["local_roi_idx"] = int(roi_index)
        out.append(q)
    return out


def run_localmix_for_frame(
    model,
    image_path: Path,
    prompt_packs: Sequence[PromptPack],
    local_small_branch_mode: str,
    person_tracks: Dict,
    vid: int,
    cid: int,
    fid: int,
    min_area: float,
    allow_unknown_phrase: bool,
    device: str,
    phone_max_area: float,
    local_person_expand_ratio: float,
    local_person_expand_ratio_y: float,
    local_merge_iou_thr: float,
    local_merge_center_thr: float,
    local_merge_center_aux_iou_thr: float,
    local_roi_dedup_iou: float,
    local_max_person_crops: int,
    local_fallback_global_when_no_person: bool,
) -> Tuple[np.ndarray, List[dict], List[Tuple[float, float, float, float]], Dict[str, Any]]:
    mode = str(local_small_branch_mode).strip().lower()
    if mode not in {"off", "rescue", "full"}:
        raise ValueError(f"Unsupported --local_small_branch_mode: {local_small_branch_mode}")

    local_small_packs, global_service_packs = split_prompt_packs_for_localmix(prompt_packs)
    if mode == "off":
        return_image, raw_props = run_packs_for_frame(
            model=model,
            image_path=image_path,
            prompt_packs=prompt_packs,
            min_area=min_area,
            allow_unknown_phrase=allow_unknown_phrase,
            device=device,
            phone_max_area=phone_max_area,
        )
        for p in raw_props:
            p.setdefault("source", "global")
        return return_image, raw_props, [], {
            "local_branch_used": False,
            "local_rois": 0,
            "local_crops": 0,
            "local_props": 0,
            "fallback_global_no_person": False,
        }

    # Global branch
    if mode == "full":
        global_packs = global_service_packs
    else:
        global_packs = list(prompt_packs)

    if global_packs:
        image_source, raw_props = run_packs_for_frame(
            model=model,
            image_path=image_path,
            prompt_packs=global_packs,
            min_area=min_area,
            allow_unknown_phrase=allow_unknown_phrase,
            device=device,
            phone_max_area=phone_max_area,
        )
    else:
        pil = Image.open(image_path).convert("RGB")
        image_source = np.asarray(pil)
        raw_props = []
    for p in raw_props:
        p.setdefault("source", "global")

    need_local = True
    if mode == "rescue":
        need_local = not any(p.get("pack_name") in {"A", "B", "C"} for p in raw_props)
    if not need_local or not local_small_packs:
        return image_source, raw_props, [], {
            "local_branch_used": False,
            "local_rois": 0,
            "local_crops": 0,
            "local_props": 0,
            "fallback_global_no_person": False,
        }

    pil_full = Image.fromarray(image_source)
    local_person_boxes = get_expanded_person_boxes_xyxy(
        person_tracks=person_tracks,
        vid=vid,
        cid=cid,
        fid=fid,
        expand_ratio=local_person_expand_ratio,
        expand_ratio_y=local_person_expand_ratio_y,
    )
    local_rois = build_localmix_rois_from_person_boxes(
        person_boxes_xyxy=local_person_boxes,
        merge_iou_thr=local_merge_iou_thr,
        merge_center_thr=local_merge_center_thr,
        merge_center_aux_iou_thr=local_merge_center_aux_iou_thr,
        dedup_iou_thr=local_roi_dedup_iou,
        max_crops=local_max_person_crops,
    )

    local_props_total = 0
    fallback_no_person = False
    if local_rois:
        if isinstance(model, dict) and str(model.get("backend", "")).lower() in {"hf", "yolo"}:
            crop_images: List[Image.Image] = []
            eff_rois: List[Tuple[float, float, float, float]] = []
            for roi in local_rois:
                crop_img, eff_roi = crop_pil_with_norm_roi(pil_full, roi)
                if crop_img is None or eff_roi is None:
                    continue
                crop_images.append(crop_img)
                eff_rois.append(eff_roi)

            if crop_images:
                local_batches = run_packs_for_pil_images_hf(
                    model=model,
                    pil_images=crop_images,
                    prompt_packs=local_small_packs,
                    min_area=min_area,
                    allow_unknown_phrase=allow_unknown_phrase,
                    device=device,
                    phone_max_area=phone_max_area,
                    source_tag="local",
                )
                mapped_local_props: List[dict] = []
                for roi_idx, (eff_roi, roi_props) in enumerate(zip(eff_rois, local_batches)):
                    mapped_local_props.extend(
                        map_local_props_to_full_image(roi_props, eff_roi, roi_idx)
                    )
                raw_props.extend(mapped_local_props)
                local_props_total = len(mapped_local_props)
            local_rois = eff_rois
        else:
            # Official backend currently keeps full-image A/B/C for compatibility.
            _, fallback_props = run_packs_for_frame(
                model=model,
                image_path=image_path,
                prompt_packs=local_small_packs,
                min_area=min_area,
                allow_unknown_phrase=allow_unknown_phrase,
                device=device,
                phone_max_area=phone_max_area,
            )
            for p in fallback_props:
                p["source"] = "global"
            raw_props.extend(fallback_props)
            local_props_total = len(fallback_props)
    elif local_fallback_global_when_no_person:
        fallback_no_person = True
        if isinstance(model, dict) and str(model.get("backend", "")).lower() in {"hf", "yolo"}:
            fallback_batches = run_packs_for_pil_images_hf(
                model=model,
                pil_images=[pil_full],
                prompt_packs=local_small_packs,
                min_area=min_area,
                allow_unknown_phrase=allow_unknown_phrase,
                device=device,
                phone_max_area=phone_max_area,
                source_tag="global",
            )
            fallback_props = fallback_batches[0] if fallback_batches else []
        else:
            _, fallback_props = run_packs_for_frame(
                model=model,
                image_path=image_path,
                prompt_packs=local_small_packs,
                min_area=min_area,
                allow_unknown_phrase=allow_unknown_phrase,
                device=device,
                phone_max_area=phone_max_area,
            )
        for p in fallback_props:
            p.setdefault("source", "global")
        raw_props.extend(fallback_props)
        local_props_total = len(fallback_props)

    return image_source, raw_props, local_rois, {
        "local_branch_used": True,
        "local_rois": len(local_rois),
        "local_crops": len(local_rois),
        "local_props": local_props_total,
        "fallback_global_no_person": bool(fallback_no_person),
    }


def build_prompt_packs(args: argparse.Namespace) -> List[PromptPack]:
    return build_prompt_packs_from_args(args)


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
        "keyboard": study_mul,
        "mouse": study_mul,
        "cup": dining_mul,
        "bottle": dining_mul,
        "bowl": dining_mul,
        "wine glass": dining_mul,
        "fork": dining_mul,
        "knife": dining_mul,
        "spoon": dining_mul,
        "banana": dining_mul,
        "apple": dining_mul,
        "sandwich": dining_mul,
        "pizza": dining_mul,
        "donut": dining_mul,
        "cake": dining_mul,
        "table": service_mul,
        "tv": service_mul,
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
    del allow_unknown_phrase, device
    batch_results = run_yolo_on_paths(
        model_bundle=model,
        image_paths=[image_path],
        prompt_packs=prompt_packs,
        min_area=min_area,
        phone_max_area=phone_max_area,
        source_tag="global",
        max_pairs_per_forward=1,
    )
    if not batch_results:
        return None, []
    return batch_results[0]


def is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def run_packs_for_frames(
    model,
    image_paths: Sequence[Path],
    prompt_packs: Sequence[PromptPack],
    min_area: float,
    allow_unknown_phrase: bool,
    device: str,
    phone_max_area: float = 0.06,
) -> List[Tuple[np.ndarray, List[dict]]]:
    del allow_unknown_phrase, device
    return run_yolo_on_paths(
        model_bundle=model,
        image_paths=image_paths,
        prompt_packs=prompt_packs,
        min_area=min_area,
        phone_max_area=phone_max_area,
        source_tag="global",
        max_pairs_per_forward=max(1, len(image_paths)),
    )


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


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 2,
    dash_len: int = 6,
    gap_len: int = 4,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    step = max(dash_len + gap_len, 1)
    n = int(dist // step) + 1
    for i in range(n):
        start = i * step
        end = min(start + dash_len, dist)
        if end <= start:
            continue
        sx = int(x1 + dx * (start / dist))
        sy = int(y1 + dy * (start / dist))
        ex = int(x1 + dx * (end / dist))
        ey = int(y1 + dy * (end / dist))
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    width: int = 2,
    dash_len: int = 6,
    gap_len: int = 4,
) -> None:
    x1, y1, x2, y2 = rect
    _draw_dashed_line(draw, (x1, y1), (x2, y1), color, width, dash_len, gap_len)
    _draw_dashed_line(draw, (x2, y1), (x2, y2), color, width, dash_len, gap_len)
    _draw_dashed_line(draw, (x2, y2), (x1, y2), color, width, dash_len, gap_len)
    _draw_dashed_line(draw, (x1, y2), (x1, y1), color, width, dash_len, gap_len)


def draw_visualization(
    image_source: np.ndarray,
    selected: Sequence[dict],
    save_path: Path,
    local_roi_boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
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

    if local_roi_boxes:
        roi_color = (0, 220, 220)
        for ridx, roi in enumerate(local_roi_boxes):
            x1 = int(roi[0] * w)
            y1 = int(roi[1] * h)
            x2 = int(roi[2] * w)
            y2 = int(roi[3] * h)
            _draw_dashed_rectangle(draw, (x1, y1, x2, y2), roi_color, width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), f"ROI#{ridx}", fill=roi_color)

    for proposal in selected:
        x1 = int(proposal["x1"] * w)
        y1 = int(proposal["y1"] * h)
        x2 = int(proposal["x2"] * w)
        y2 = int(proposal["y2"] * h)
        family = proposal["family"]
        color = colors.get(family, (255, 255, 255))
        src = str(proposal.get("source", "global"))
        label = f"{proposal['token']}:{proposal['score']:.2f}:{src}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)

    image.save(save_path)


def resolve_base_outputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    data_root = Path(args.data_root)
    if args.output_pkl:
        base_output_pkl = Path(args.output_pkl)
    else:
        base_output_pkl = data_root / "object_tracks_yolov8_localmix.pkl"

    if args.output_meta:
        base_output_meta = Path(args.output_meta)
    else:
        base_output_meta = data_root / "object_tracks_yolov8_localmix_meta.json"
    return base_output_pkl, base_output_meta


def shard_path(path: Path, rank: int, world_size: int) -> Path:
    return path.with_name(f"{path.stem}.rank{rank:02d}of{world_size:02d}{path.suffix}")


def ensure_outputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    base_output_pkl, base_output_meta = resolve_base_outputs(args)
    if args.merge_shards:
        output_pkl, output_meta = base_output_pkl, base_output_meta
    elif args.world_size > 1:
        output_pkl = shard_path(base_output_pkl, args.rank, args.world_size)
        output_meta = shard_path(base_output_meta, args.rank, args.world_size)
    else:
        output_pkl, output_meta = base_output_pkl, base_output_meta

    for path in (output_pkl, output_meta):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {path}. Use --overwrite to replace it."
            )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    return output_pkl, output_meta


def merge_sharded_outputs(
    args: argparse.Namespace,
    final_output_pkl: Path,
    final_output_meta: Path,
) -> None:
    if args.world_size <= 1:
        raise ValueError("--merge_shards requires --world_size > 1.")
    base_output_pkl, base_output_meta = resolve_base_outputs(args)
    shard_pkls = [
        shard_path(base_output_pkl, rank, args.world_size) for rank in range(args.world_size)
    ]
    shard_metas = [
        shard_path(base_output_meta, rank, args.world_size) for rank in range(args.world_size)
    ]
    missing_pkls = [str(p) for p in shard_pkls if not p.exists()]
    if missing_pkls:
        raise FileNotFoundError(
            "Missing shard pkl files:\n" + "\n".join(missing_pkls)
        )

    merged_tracks: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for p in shard_pkls:
        with open(p, "rb") as f:
            shard_tracks = pickle.load(f)
        if not isinstance(shard_tracks, dict):
            raise TypeError(f"Shard track file should be dict: {p}")
        overlap = set(merged_tracks.keys()).intersection(set(shard_tracks.keys()))
        if overlap:
            sample = sorted(list(overlap))[:5]
            raise ValueError(
                f"Duplicate clip keys found while merging {p}: sample={sample}"
            )
        merged_tracks.update(shard_tracks)

    final_output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_pkl, "wb") as f:
        pickle.dump(merged_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

    shard_meta_payloads = []
    for p in shard_metas:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                shard_meta_payloads.append(json.load(f))

    if shard_meta_payloads:
        merged_meta = shard_meta_payloads[0]
    else:
        merged_meta = {"format": {}, "paths": {}, "stats": {}}

    sum_keys = [
        "total_clips",
        "total_frames",
        "frames_no_raw",
        "frames_no_person_overlap",
        "frames_no_final",
        "local_branch_frames",
        "local_roi_total",
        "local_crop_infer_total",
        "local_props_total",
        "local_fallback_global_total",
        "raw_props_total",
        "person_boxes_total",
        "person_gate_props_total",
        "person_filtered_out_total",
        "nms_props_total",
        "contain_props_total",
        "capped_props_total",
        "subset_props_total",
        "cross_dedup_props_total",
        "final_props_total",
        "final_valid_total",
        "errors",
    ]
    merged_stats: Dict[str, float] = {k: 0.0 for k in sum_keys}
    merged_family_counter: Counter = Counter()
    merged_token_counter: Counter = Counter()
    merged_pack_counter: Counter = Counter()
    for m in shard_meta_payloads:
        st = m.get("stats", {})
        for k in sum_keys:
            v = st.get(k, 0)
            if isinstance(v, (int, float)):
                merged_stats[k] += float(v)
        merged_family_counter.update(st.get("final_family_counter", {}))
        merged_token_counter.update(st.get("final_token_counter", {}))
        merged_pack_counter.update(st.get("raw_pack_counter", {}))

    total_frames = max(int(merged_stats["total_frames"]), 1)
    merged_stats["avg_raw_props_per_frame"] = (
        float(merged_stats["raw_props_total"]) / total_frames
    )
    merged_stats["avg_person_gate_props_per_frame"] = (
        float(merged_stats["person_gate_props_total"]) / total_frames
    )
    merged_stats["avg_local_rois_per_frame"] = (
        float(merged_stats["local_roi_total"]) / total_frames
    )
    merged_stats["avg_local_props_per_frame"] = (
        float(merged_stats["local_props_total"]) / total_frames
    )
    merged_stats["avg_final_props_per_frame"] = (
        float(merged_stats["final_props_total"]) / total_frames
    )
    merged_stats["avg_valid_rows_per_frame"] = (
        float(merged_stats["final_valid_total"]) / total_frames
    )
    merged_stats["final_family_counter"] = dict(merged_family_counter)
    merged_stats["final_token_counter"] = dict(merged_token_counter)
    merged_stats["raw_pack_counter"] = dict(merged_pack_counter)

    merged_meta.setdefault("paths", {})
    merged_meta["paths"]["output_pkl"] = str(final_output_pkl)
    merged_meta["paths"]["output_meta"] = str(final_output_meta)
    merged_meta["paths"]["base_output_pkl"] = str(base_output_pkl)
    merged_meta["paths"]["base_output_meta"] = str(base_output_meta)
    merged_meta["paths"]["merged_from_shards"] = [str(p) for p in shard_pkls]
    merged_meta["sharding"] = {
        "world_size": int(args.world_size),
        "mode": "merged",
    }
    merged_meta["stats"] = merged_stats

    final_output_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_meta, "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    print("[merge] Done.")
    print(f"  Merged shards : {len(shard_pkls)}")
    print(f"  Saved pkl     : {final_output_pkl}")
    print(f"  Saved meta    : {final_output_meta}")


def main() -> None:
    args = parse_args()
    if args.global_topk <= 0:
        raise ValueError("--global_topk must be > 0 for fixed-length export.")
    if args.frame_batch_size <= 0:
        raise ValueError("--frame_batch_size must be > 0.")
    if args.world_size <= 0:
        raise ValueError("--world_size must be > 0.")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world_size.")
    local_mode = str(args.local_small_branch_mode).strip().lower()
    if local_mode not in {"off", "rescue", "full"}:
        raise ValueError("--local_small_branch_mode must be one of: off, rescue, full.")
    if args.local_max_person_crops < 0:
        raise ValueError("--local_max_person_crops must be >= 0.")
    if args.local_person_expand_ratio <= 0.0:
        raise ValueError("--local_person_expand_ratio must be > 0.")
    if args.local_person_expand_ratio_y <= 0.0:
        raise ValueError("--local_person_expand_ratio_y must be > 0.")
    if str(args.device).lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but no visible GPU in this process. "
                f"rank/world={args.rank}/{args.world_size}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
                f"torch={torch.__version__}, torch_cuda={torch.version.cuda}"
            )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_pkl, output_meta = ensure_outputs(args)
    if args.merge_shards:
        merge_sharded_outputs(
            args=args,
            final_output_pkl=output_pkl,
            final_output_meta=output_meta,
        )
        return

    prompt_packs = build_prompt_packs(args)
    local_small_packs, global_service_packs = split_prompt_packs_for_localmix(prompt_packs)
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

    print(f"[2/6] Loading YOLO backend={args.backend} on device={args.device} ...")
    model = load_detector_from_args(args)
    model_backend = (
        str(model.get("backend", "yolo")).strip().lower()
        if isinstance(model, dict)
        else "yolo"
    )
    if local_mode != "off" and model_backend not in {"hf", "yolo"}:
        print(
            "[warn] Local crop branch expects an in-memory backend (yolo/hf). "
            "Current backend will fallback to full-image A/B/C inference."
        )

    print(f"[3/6] Scanning clips under: {data_root}")
    clips, frame_keys = collect_clips(data_root)
    if not clips:
        raise RuntimeError(f"No valid clips found under: {data_root}")
    total_clips_all = len(clips)
    total_frames_all = len(frame_keys)
    if args.world_size > 1:
        local_clips = [
            clip for idx, clip in enumerate(clips) if idx % args.world_size == args.rank
        ]
        local_frame_keys: List[Tuple[int, int, int]] = []
        for clip in local_clips:
            vid = clip["vid"]
            cid = clip["cid"]
            local_frame_keys.extend([(vid, cid, fid) for fid, _ in clip["frames"]])
        clips = local_clips
        frame_keys = local_frame_keys
    print(f"  Found clips (all) : {total_clips_all}")
    print(f"  Found frames (all): {total_frames_all}")
    if args.world_size > 1:
        print(
            f"  Shard rank/world  : {args.rank}/{args.world_size} "
            f"(clips={len(clips)}, frames={len(frame_keys)})"
        )
    else:
        print(f"  Found clips       : {len(clips)}")
        print(f"  Found frames      : {len(frame_keys)}")

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
        "local_branch_frames": 0,
        "local_roi_total": 0,
        "local_crop_infer_total": 0,
        "local_props_total": 0,
        "local_fallback_global_total": 0,
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

    clip_desc = (
        f"Clips[r{args.rank}/{args.world_size}]"
        if args.world_size > 1
        else "Clips"
    )
    clip_bar = tqdm(clips, desc=clip_desc, ncols=100)
    for clip in clip_bar:
        vid = clip["vid"]
        cid = clip["cid"]
        max_fid = clip["max_fid"]
        clip_tracks = [
            np.zeros((args.global_topk, 10), dtype=np.float32) for _ in range(max_fid + 1)
        ]

        use_frame_batch = (
            local_mode == "off"
            and model_backend in {"hf", "yolo"}
            and args.frame_batch_size > 1
        )
        target_batch_size = max(1, int(args.frame_batch_size)) if use_frame_batch else 1
        current_batch_size = target_batch_size

        clip_frames = clip["frames"]
        frame_idx = 0
        while frame_idx < len(clip_frames):
            if local_mode == "off":
                take = min(current_batch_size, len(clip_frames) - frame_idx)
            else:
                take = 1
            frame_batch = clip_frames[frame_idx : frame_idx + take]
            fids = [item[0] for item in frame_batch]
            paths = [item[1] for item in frame_batch]

            batch_results: List[Tuple[Optional[np.ndarray], List[dict]]]
            batch_local_rois: List[List[Tuple[float, float, float, float]]]
            batch_local_info: List[Dict[str, Any]]
            try:
                if local_mode == "off":
                    if use_frame_batch:
                        batch_results = run_packs_for_frames(
                            model=model,
                            image_paths=paths,
                            prompt_packs=prompt_packs,
                            min_area=args.min_area,
                            allow_unknown_phrase=args.allow_unknown_phrase,
                            device=args.device,
                            phone_max_area=args.phone_max_area,
                        )
                    else:
                        batch_results = []
                        for pth in paths:
                            image_source, raw_props = run_packs_for_frame(
                                model=model,
                                image_path=pth,
                                prompt_packs=prompt_packs,
                                min_area=args.min_area,
                                allow_unknown_phrase=args.allow_unknown_phrase,
                                device=args.device,
                                phone_max_area=args.phone_max_area,
                            )
                            batch_results.append((image_source, raw_props))
                    batch_local_rois = [[] for _ in frame_batch]
                    batch_local_info = [
                        {
                            "local_branch_used": False,
                            "local_rois": 0,
                            "local_crops": 0,
                            "local_props": 0,
                            "fallback_global_no_person": False,
                        }
                        for _ in frame_batch
                    ]
                else:
                    batch_results = []
                    batch_local_rois = []
                    batch_local_info = []
                    for cur_fid, pth in frame_batch:
                        image_source, raw_props, local_rois, local_info = run_localmix_for_frame(
                            model=model,
                            image_path=pth,
                            prompt_packs=prompt_packs,
                            local_small_branch_mode=local_mode,
                            person_tracks=person_tracks,
                            vid=vid,
                            cid=cid,
                            fid=cur_fid,
                            min_area=args.min_area,
                            allow_unknown_phrase=args.allow_unknown_phrase,
                            device=args.device,
                            phone_max_area=args.phone_max_area,
                            local_person_expand_ratio=args.local_person_expand_ratio,
                            local_person_expand_ratio_y=args.local_person_expand_ratio_y,
                            local_merge_iou_thr=args.local_merge_iou_thr,
                            local_merge_center_thr=args.local_merge_center_thr,
                            local_merge_center_aux_iou_thr=args.local_merge_center_aux_iou_thr,
                            local_roi_dedup_iou=args.local_roi_dedup_iou,
                            local_max_person_crops=args.local_max_person_crops,
                            local_fallback_global_when_no_person=args.local_fallback_global_when_no_person,
                        )
                        batch_results.append((image_source, raw_props))
                        batch_local_rois.append(local_rois)
                        batch_local_info.append(local_info)
            except RuntimeError as exc:
                if (
                    use_frame_batch
                    and args.auto_reduce_batch_on_oom
                    and take > 1
                    and is_cuda_oom_error(exc)
                ):
                    current_batch_size = max(1, take // 2)
                    if str(args.device).lower().startswith("cuda"):
                        torch.cuda.empty_cache()
                    clip_bar.set_postfix_str(f"oom_retry_bs={current_batch_size}")
                    continue
                stats["errors"] += len(frame_batch)
                batch_results = [(None, []) for _ in frame_batch]
                batch_local_rois = [[] for _ in frame_batch]
                batch_local_info = [
                    {
                        "local_branch_used": False,
                        "local_rois": 0,
                        "local_crops": 0,
                        "local_props": 0,
                        "fallback_global_no_person": False,
                    }
                    for _ in frame_batch
                ]
            except Exception:
                stats["errors"] += len(frame_batch)
                batch_results = [(None, []) for _ in frame_batch]
                batch_local_rois = [[] for _ in frame_batch]
                batch_local_info = [
                    {
                        "local_branch_used": False,
                        "local_rois": 0,
                        "local_crops": 0,
                        "local_props": 0,
                        "fallback_global_no_person": False,
                    }
                    for _ in frame_batch
                ]

            if use_frame_batch and current_batch_size < target_batch_size:
                current_batch_size = min(target_batch_size, current_batch_size * 2)

            for fid, (image_source, raw_props), local_rois, local_info in zip(
                fids, batch_results, batch_local_rois, batch_local_info
            ):
                stats["total_frames"] += 1
                if local_info.get("local_branch_used", False):
                    stats["local_branch_frames"] += 1
                stats["local_roi_total"] += int(local_info.get("local_rois", 0))
                stats["local_crop_infer_total"] += int(local_info.get("local_crops", 0))
                stats["local_props_total"] += int(local_info.get("local_props", 0))
                if local_info.get("fallback_global_no_person", False):
                    stats["local_fallback_global_total"] += 1

                if not raw_props:
                    stats["frames_no_raw"] += 1
                    stats["frames_no_final"] += 1
                    continue

                stats["raw_props_total"] += len(raw_props)
                for p in raw_props:
                    pack_counter[str(p.get("pack_name", ""))] += 1

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
                    draw_visualization(
                        image_source,
                        final_props,
                        vis_path,
                        local_roi_boxes=local_rois,
                    )

            frame_idx += take

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
    avg_local_rois = (
        float(stats["local_roi_total"]) / max(stats["total_frames"], 1)
    )
    avg_local_props = (
        float(stats["local_props_total"]) / max(stats["total_frames"], 1)
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
            "base_output_pkl": str(resolve_base_outputs(args)[0]),
            "base_output_meta": str(resolve_base_outputs(args)[1]),
            "vis_dir": str(args.vis_dir) if args.vis_dir else "",
            "backend": args.backend,
            "device": args.device,
            "yolo_model": args.yolo_model,
            "yolo_imgsz": int(args.yolo_imgsz),
            "yolo_conf": float(args.yolo_conf),
            "yolo_iou": float(args.yolo_iou),
            "yolo_max_det": int(args.yolo_max_det),
            "yolo_batch_size": int(args.yolo_batch_size),
            "frame_batch_size": int(args.frame_batch_size),
            "auto_reduce_batch_on_oom": bool(args.auto_reduce_batch_on_oom),
            "person_tracks_pkl": str(person_tracks_path),
            "local_small_branch_mode": local_mode,
        },
        "sharding": {
            "world_size": int(args.world_size),
            "rank": int(args.rank),
            "mode": "shard" if args.world_size > 1 else "single",
            "clips_all": int(total_clips_all),
            "frames_all": int(total_frames_all),
            "clips_this_rank": int(len(clips)),
            "frames_this_rank": int(len(frame_keys)),
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
        "localmix": {
            "mode": local_mode,
            "global_branch_packs": (
                [p.name for p in prompt_packs]
                if local_mode in {"off", "rescue"}
                else [p.name for p in (global_service_packs or prompt_packs)]
            ),
            "local_branch_packs": [p.name for p in local_small_packs],
            "local_person_expand_ratio_x": args.local_person_expand_ratio,
            "local_person_expand_ratio_y": args.local_person_expand_ratio_y,
            "local_merge_iou_thr": args.local_merge_iou_thr,
            "local_merge_center_thr": args.local_merge_center_thr,
            "local_merge_center_aux_iou_thr": args.local_merge_center_aux_iou_thr,
            "local_roi_dedup_iou": args.local_roi_dedup_iou,
            "local_max_person_crops": args.local_max_person_crops,
            "local_fallback_global_when_no_person": bool(
                args.local_fallback_global_when_no_person
            ),
        },
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
            "avg_local_rois_per_frame": avg_local_rois,
            "avg_local_props_per_frame": avg_local_props,
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
    if args.world_size > 1:
        print(
            f"  Shard info: rank={args.rank}/{args.world_size} "
            "(run all ranks, then use --merge_shards)"
        )
    if args.vis_dir and args.vis_samples > 0:
        print(f"  Saved vis : {args.vis_dir}")
    print("  Summary:")
    print(f"    Total clips               : {stats['total_clips']}")
    print(f"    Total frames              : {stats['total_frames']}")
    print(f"    Frames with no raw props  : {stats['frames_no_raw']}")
    print(f"    Frames with no person hit : {stats['frames_no_person_overlap']}")
    print(f"    Frames with no final props: {stats['frames_no_final']}")
    print(f"    Local branch frames       : {stats['local_branch_frames']}")
    print(f"    Local ROIs total          : {stats['local_roi_total']}")
    print(f"    Local props total         : {stats['local_props_total']}")
    print(f"    Local fallback(no-person) : {stats['local_fallback_global_total']}")
    print(f"    Avg raw props / frame     : {avg_raw:.3f}")
    print(f"    Avg person-gated / frame  : {avg_person_gate:.3f}")
    print(f"    Avg local ROIs / frame    : {avg_local_rois:.3f}")
    print(f"    Avg local props / frame   : {avg_local_props:.3f}")
    print(f"    Avg final props / frame   : {avg_final:.3f}")
    print(f"    Avg valid rows / frame    : {avg_valid:.3f} (M={args.global_topk})")


if __name__ == "__main__":
    main()
