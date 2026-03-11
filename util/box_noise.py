import hashlib
from typing import Dict, List

import torch


def _stable_seed(base_seed: int, vid, sid, fids) -> int:
    key = f"{int(base_seed)}|{int(vid)}|{int(sid)}|{','.join(map(str, fids))}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def should_apply_box_noise(args, phase: str) -> bool:
    policy = getattr(args, "box_noise_policy", "none")
    if policy == "none":
        return False
    if policy == "infer_only":
        return phase == "infer"
    if policy == "train_and_infer":
        return phase in ("train", "infer")
    return False


def _sample_noise(shape, seed: int, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def apply_box_noise(batch_boxes: torch.Tensor, infos: List[Dict], args, phase: str) -> torch.Tensor:
    """
    Apply deterministic noise to normalized boxes (cx, cy, w, h).
    """
    if not should_apply_box_noise(args, phase):
        return batch_boxes

    if batch_boxes.ndim != 4 or batch_boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes shape [B, T, N, 4], got {tuple(batch_boxes.shape)}")

    center_std = float(getattr(args, "box_noise_center_std", 0.10))
    scale_std = float(getattr(args, "box_noise_scale_std", 0.08))
    aspect_std = float(getattr(args, "box_noise_aspect_std", 0.08))
    min_size = float(getattr(args, "box_noise_min_size", 1e-4))
    max_size = float(getattr(args, "box_noise_max_size", 1.0))
    base_seed = int(getattr(args, "box_noise_seed", getattr(args, "random_seed", 1)))

    boxes = batch_boxes.clone()
    bsz, t, n, _ = boxes.shape
    for b in range(bsz):
        info = infos[b]
        seed = _stable_seed(base_seed, info["vid"], info["sid"], info["fid"])

        eps_center = _sample_noise((t, n, 2), seed + 11, boxes.device)
        eps_scale = _sample_noise((t, n), seed + 23, boxes.device)
        eps_aspect = _sample_noise((t, n), seed + 37, boxes.device)

        cx = boxes[b, :, :, 0]
        cy = boxes[b, :, :, 1]
        w = boxes[b, :, :, 2]
        h = boxes[b, :, :, 3]

        valid = (w > 1e-6) & (h > 1e-6)
        if valid.sum().item() == 0:
            continue

        cx_new = cx + eps_center[:, :, 0] * center_std * w
        cy_new = cy + eps_center[:, :, 1] * center_std * h

        size_scale = torch.exp(eps_scale * scale_std)
        ratio_scale = torch.exp(eps_aspect * aspect_std)
        w_new = w * size_scale * ratio_scale
        h_new = h * size_scale / ratio_scale

        w_new = w_new.clamp(min=min_size, max=max_size)
        h_new = h_new.clamp(min=min_size, max=max_size)

        cx_new = torch.min(torch.max(cx_new, w_new / 2.0), 1.0 - w_new / 2.0)
        cy_new = torch.min(torch.max(cy_new, h_new / 2.0), 1.0 - h_new / 2.0)

        boxes[b, :, :, 0] = torch.where(valid, cx_new, cx)
        boxes[b, :, :, 1] = torch.where(valid, cy_new, cy)
        boxes[b, :, :, 2] = torch.where(valid, w_new, w)
        boxes[b, :, :, 3] = torch.where(valid, h_new, h)

    return boxes
