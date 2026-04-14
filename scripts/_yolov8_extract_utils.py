from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False


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
    "keyboard",
    "mouse",
    "cup",
    "bottle",
    "bowl",
    "wine glass",
    "fork",
    "knife",
    "spoon",
    "banana",
    "apple",
    "sandwich",
    "pizza",
    "donut",
    "cake",
    "table",
    "tv",
]
TOKEN_TO_ID = {name: idx + 1 for idx, name in enumerate(TOKEN_ORDER)}
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}

TOKEN_TO_FAMILY = {
    "phone": "phone-family",
    "laptop": "study-family",
    "book": "study-family",
    "keyboard": "study-family",
    "mouse": "study-family",
    "cup": "dining-family",
    "bottle": "dining-family",
    "bowl": "dining-family",
    "wine glass": "dining-family",
    "fork": "dining-family",
    "knife": "dining-family",
    "spoon": "dining-family",
    "banana": "dining-family",
    "apple": "dining-family",
    "sandwich": "dining-family",
    "pizza": "dining-family",
    "donut": "dining-family",
    "cake": "dining-family",
    "table": "service-family",
    "tv": "service-family",
}

PACK_TO_TOKENS = {
    "A": {"phone"},
    "B": {"laptop", "book", "keyboard", "mouse"},
    "C": {
        "cup",
        "bottle",
        "bowl",
        "wine glass",
        "fork",
        "knife",
        "spoon",
        "banana",
        "apple",
        "sandwich",
        "pizza",
        "donut",
        "cake",
    },
    "D": {"table", "tv"},
}
TOKEN_TO_PACK = {
    token: pack_name
    for pack_name, tokens in PACK_TO_TOKENS.items()
    for token in tokens
}

YOLO_CLASS_TO_TOKEN = {
    "cell phone": "phone",
    "laptop": "laptop",
    "book": "book",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "cup": "cup",
    "bottle": "bottle",
    "bowl": "bowl",
    "wine glass": "wine glass",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
    "dining table": "table",
    "tv": "tv",
}


@dataclass(frozen=True)
class PromptPack:
    name: str
    pack_id: int
    caption: str
    box_threshold: float
    text_threshold: float


def build_prompt_packs_from_args(args) -> List[PromptPack]:
    return [
        PromptPack(
            name="A",
            pack_id=1,
            caption="yolo-pack-a-phone",
            box_threshold=float(getattr(args, "box_th_a", 0.25)),
            text_threshold=float(getattr(args, "text_th_a", 0.20)),
        ),
        PromptPack(
            name="B",
            pack_id=2,
            caption="yolo-pack-b-study",
            box_threshold=float(getattr(args, "box_th_b", 0.25)),
            text_threshold=float(getattr(args, "text_th_b", 0.20)),
        ),
        PromptPack(
            name="C",
            pack_id=3,
            caption="yolo-pack-c-dining",
            box_threshold=float(getattr(args, "box_th_c", 0.27)),
            text_threshold=float(getattr(args, "text_th_c", 0.22)),
        ),
        PromptPack(
            name="D",
            pack_id=4,
            caption="yolo-pack-d-service",
            box_threshold=float(getattr(args, "box_th_d", 0.35)),
            text_threshold=float(getattr(args, "text_th_d", 0.25)),
        ),
    ]


def split_prompt_packs_for_localmix(
    prompt_packs: Sequence[PromptPack],
) -> Tuple[List[PromptPack], List[PromptPack]]:
    local_small = [p for p in prompt_packs if p.name in {"A", "B", "C"}]
    global_service = [p for p in prompt_packs if p.name == "D"]
    return local_small, global_service


def _normalize_yolo_device(device: str):
    dev = str(device).strip().lower()
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if ":" in dev:
            return dev.split(":", 1)[1]
        return 0
    return "cpu"


def load_yolo_detector_from_args(args):
    if not YOLO_AVAILABLE:
        raise ImportError(
            "ultralytics is not installed. Please install it in your environment: "
            "`pip install ultralytics`"
        )
    yolo_model = str(getattr(args, "yolo_model", "yolov8x.pt"))
    model = YOLO(yolo_model)
    return {
        "backend": "yolo",
        "model": model,
        "device": _normalize_yolo_device(getattr(args, "device", "cuda")),
        "imgsz": int(getattr(args, "yolo_imgsz", 1280)),
        "conf": float(getattr(args, "yolo_conf", 0.01)),
        "iou": float(getattr(args, "yolo_iou", 0.7)),
        "max_det": int(getattr(args, "yolo_max_det", 300)),
        "batch_size": int(getattr(args, "yolo_batch_size", 8)),
    }


def is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _build_pack_thresholds(prompt_packs: Sequence[PromptPack], yolo_conf: float):
    by_pack = {
        p.name: max(float(yolo_conf), float(p.box_threshold))
        for p in prompt_packs
    }
    return by_pack


def _postprocess_yolo_result_for_frame(
    result,
    img_w: int,
    img_h: int,
    pack_thresh: Dict[str, float],
    min_area: float,
    phone_max_area: float,
    source_tag: str,
    class_to_token: Optional[Dict[str, str]] = None,
    token_to_pack: Optional[Dict[str, str]] = None,
    token_to_family: Optional[Dict[str, str]] = None,
    family_to_id: Optional[Dict[str, int]] = None,
    token_to_id: Optional[Dict[str, int]] = None,
    pack_id_map: Optional[Dict[str, int]] = None,
) -> List[dict]:
    out: List[dict] = []
    if result is None or getattr(result, "boxes", None) is None:
        return out
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.detach().cpu().float().numpy()
    conf = boxes.conf.detach().cpu().float().numpy()
    cls = boxes.cls.detach().cpu().int().numpy()

    names_map = getattr(result, "names", None)
    if names_map is None:
        names_map = {}

    class_to_token = class_to_token or YOLO_CLASS_TO_TOKEN
    token_to_pack = token_to_pack or TOKEN_TO_PACK
    token_to_family = token_to_family or TOKEN_TO_FAMILY
    family_to_id = family_to_id or FAMILY_TO_ID
    token_to_id = token_to_id or TOKEN_TO_ID
    pack_id_map = pack_id_map or {"A": 1, "B": 2, "C": 3, "D": 4}

    norm = np.asarray([img_w, img_h, img_w, img_h], dtype=np.float32)

    for i in range(xyxy.shape[0]):
        class_idx = int(cls[i])
        class_name = str(names_map.get(class_idx, "")).strip().lower()
        token = class_to_token.get(class_name)
        if token is None:
            continue
        pack_name = token_to_pack.get(token)
        if pack_name is None:
            continue
        th = float(pack_thresh.get(pack_name, 0.0))
        score = float(conf[i])
        if score < th:
            continue

        family = token_to_family[token]
        family_id = family_to_id[family]
        token_id = token_to_id[token]
        pack_id = int(pack_id_map[pack_name])

        b = (xyxy[i] / norm).astype(np.float32)
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < float(min_area):
            continue
        if (
            family == "phone-family"
            and float(phone_max_area) > 0.0
            and area > float(phone_max_area)
        ):
            continue

        out.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": score,
                "family": family,
                "family_id": family_id,
                "pack_id": float(pack_id),
                "pack_name": pack_name,
                "token": token,
                "token_id": float(token_id),
                "raw_phrase": class_name,
                "source": source_tag,
            }
        )
    return out


def run_yolo_on_pil_images(
    model_bundle,
    pil_images: Sequence[Image.Image],
    prompt_packs: Sequence[PromptPack],
    min_area: float,
    phone_max_area: float,
    source_tag: str = "global",
    max_pairs_per_forward: Optional[int] = None,
    class_to_token: Optional[Dict[str, str]] = None,
    token_to_pack: Optional[Dict[str, str]] = None,
    token_to_family: Optional[Dict[str, str]] = None,
    family_to_id: Optional[Dict[str, int]] = None,
    token_to_id: Optional[Dict[str, int]] = None,
    pack_id_map: Optional[Dict[str, int]] = None,
) -> List[List[dict]]:
    if not pil_images:
        return []
    model = model_bundle["model"]
    device = model_bundle["device"]
    imgsz = int(model_bundle["imgsz"])
    conf = float(model_bundle["conf"])
    iou = float(model_bundle["iou"])
    max_det = int(model_bundle["max_det"])
    bs_default = int(model_bundle["batch_size"])
    batch_size = max_pairs_per_forward if max_pairs_per_forward is not None else bs_default
    batch_size = max(1, int(batch_size))

    pack_thresh = _build_pack_thresholds(prompt_packs, yolo_conf=conf)

    frame_props: List[List[dict]] = [[] for _ in pil_images]
    cursor = 0
    cur_bs = min(batch_size, len(pil_images))
    while cursor < len(pil_images):
        chunk = pil_images[cursor : cursor + cur_bs]
        np_chunk = [np.asarray(im.convert("RGB")) for im in chunk]
        try:
            results = model.predict(
                source=np_chunk,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
                stream=False,
                batch=len(np_chunk),
            )
        except RuntimeError as exc:
            if (
                str(device).lower() != "cpu"
                and is_cuda_oom_error(exc)
                and cur_bs > 1
            ):
                cur_bs = max(1, cur_bs // 2)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise

        for off, res in enumerate(results):
            frame_idx = cursor + off
            w, h = pil_images[frame_idx].size
            frame_props[frame_idx] = _postprocess_yolo_result_for_frame(
                result=res,
                img_w=w,
                img_h=h,
                pack_thresh=pack_thresh,
                min_area=min_area,
                phone_max_area=phone_max_area,
                source_tag=source_tag,
                class_to_token=class_to_token,
                token_to_pack=token_to_pack,
                token_to_family=token_to_family,
                family_to_id=family_to_id,
                token_to_id=token_to_id,
                pack_id_map=pack_id_map,
            )
        cursor += len(chunk)
        if cur_bs < batch_size:
            cur_bs = min(batch_size, cur_bs * 2)
    return frame_props


def run_yolo_on_paths(
    model_bundle,
    image_paths: Sequence,
    prompt_packs: Sequence[PromptPack],
    min_area: float,
    phone_max_area: float,
    source_tag: str = "global",
    max_pairs_per_forward: Optional[int] = None,
    class_to_token: Optional[Dict[str, str]] = None,
    token_to_pack: Optional[Dict[str, str]] = None,
    token_to_family: Optional[Dict[str, str]] = None,
    family_to_id: Optional[Dict[str, int]] = None,
    token_to_id: Optional[Dict[str, int]] = None,
    pack_id_map: Optional[Dict[str, int]] = None,
):
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    image_sources = [np.asarray(im) for im in pil_images]
    frame_props = run_yolo_on_pil_images(
        model_bundle=model_bundle,
        pil_images=pil_images,
        prompt_packs=prompt_packs,
        min_area=min_area,
        phone_max_area=phone_max_area,
        source_tag=source_tag,
        max_pairs_per_forward=max_pairs_per_forward,
        class_to_token=class_to_token,
        token_to_pack=token_to_pack,
        token_to_family=token_to_family,
        family_to_id=family_to_id,
        token_to_id=token_to_id,
        pack_id_map=pack_id_map,
    )
    return list(zip(image_sources, frame_props))
