import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import profile, ProfilerActivity

# Ensure project root is on PYTHONPATH when running as `python scripts/profile_ablation.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.models import GADTR


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def parse_value(raw):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return str2bool(lowered)
    try:
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def cast_like(value, like_value):
    if isinstance(like_value, bool):
        return str2bool(value) if isinstance(value, str) else bool(value)
    if isinstance(like_value, int) and not isinstance(like_value, bool):
        return int(value)
    if isinstance(like_value, float):
        return float(value)
    return value


def parse_ablate_flags(ablate_flags, base_cfg):
    if not ablate_flags:
        return [{"name": "base", "overrides": {}}]

    dims = []
    for spec in ablate_flags:
        if "=" not in spec:
            raise ValueError(f"Invalid --ablate spec: {spec}. Expect key=v1,v2")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if key not in base_cfg:
            raise KeyError(f"Unknown ablation key: {key}")
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No values found in --ablate spec: {spec}")
        parsed = [cast_like(parse_value(v), base_cfg[key]) for v in values]
        dims.append((key, parsed))

    variants = []
    keys = [k for k, _ in dims]
    all_values = [vals for _, vals in dims]
    for combo in itertools.product(*all_values):
        overrides = dict(zip(keys, combo))
        name_parts = [f"{k}={overrides[k]}" for k in keys]
        variants.append({"name": " | ".join(name_parts), "overrides": overrides})
    return variants


def load_variants_from_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("variants json must be a list")
    variants = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict) or "overrides" not in item:
            raise ValueError(f"variants[{i}] must be an object with key 'overrides'")
        name = item.get("name", f"variant_{i}")
        overrides = item["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"variants[{i}].overrides must be an object")
        variants.append({"name": name, "overrides": overrides})
    return variants


def to_namespace(cfg_dict):
    return SimpleNamespace(**cfg_dict)


def make_dummy_batch(cfg, device):
    b = 1
    t = cfg.num_frame
    n = cfg.num_boxes
    h = cfg.image_height
    w = cfg.image_width

    images = torch.randn(b, t, 3, h, w, device=device)

    centers = torch.rand(b, t, n, 2, device=device)
    sizes = 0.05 + 0.25 * torch.rand(b, t, n, 2, device=device)
    boxes = torch.cat([centers, sizes], dim=-1).clamp(0.0, 1.0)

    dummy_mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    valid_actors = max(1, min(cfg.valid_actors, n))
    if valid_actors < n:
        dummy_mask[:, valid_actors:] = True

    mae_feats = None
    if cfg.use_mae and cfg.mae_fusion != "none":
        mae_feats = torch.randn(b, cfg.mae_dim, device=device)

    return images, boxes, dummy_mask, mae_feats


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, inputs, use_cuda):
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
            _ = model(*inputs)
        if use_cuda:
            torch.cuda.synchronize()

    flops = 0.0
    for evt in prof.key_averages():
        if evt.flops is not None and not math.isnan(evt.flops):
            flops += float(evt.flops)
    return flops


def measure_latency_and_memory(model, inputs, device, warmup, iters):
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(*inputs)
        if use_cuda:
            torch.cuda.synchronize()

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    with torch.inference_mode():
        for _ in range(iters):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(*inputs)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(*inputs)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    if use_cuda:
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_mem_bytes = 0

    return sum(times_ms) / len(times_ms), peak_mem_bytes


def default_config(args):
    use_mae = (not args.no_mae) and args.mae_fusion != "none"
    mae_dim = 1408 if args.mae_version == "v2" else 768

    return {
        "dataset": "cafe",
        "num_class": args.num_class,
        "num_frame": args.num_frame,
        "num_boxes": args.num_boxes,
        "hidden_dim": args.hidden_dim,
        "backbone": args.backbone,
        "dilation": args.dilation,
        "frozen_batch_norm": args.frozen_batch_norm,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_blocks": args.unfreeze_blocks,
        "drop_rate": args.drop_rate,
        "crop_size": args.crop_size,
        "position_embedding": "sine",
        "gar_nheads": args.gar_nheads,
        "gar_enc_layers": args.gar_enc_layers,
        "gar_ffn_dim": args.gar_ffn_dim,
        "num_group_tokens": args.num_group_tokens,
        "distance_threshold": args.distance_threshold,
        "hoi_mode": args.hoi_mode,
        "hoi_nheads": args.hoi_nheads,
        "hoi_topk": args.hoi_topk,
        "hoi_hard_thresh": args.hoi_hard_thresh,
        "temporal_layers": args.temporal_layers,
        "tcn_kernel_size": args.tcn_kernel_size,
        "tcn_dropout": args.tcn_dropout,
        "temporal_agg_mode": args.temporal_agg_mode,
        "mae_fusion": args.mae_fusion,
        "mae_fusion_stage": args.mae_fusion_stage,
        "use_mae": use_mae,
        "mae_dim": mae_dim if use_mae else 0,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "valid_actors": args.valid_actors,
    }


def apply_overrides(base_cfg, overrides):
    cfg = dict(base_cfg)
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"Unknown override key: {k}")
        cfg[k] = cast_like(v, cfg[k])
    return cfg


def save_results(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_profile.json"
    csv_path = out_dir / "ablation_profile.csv"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Profile ablations: Params/FLOPs/Latency/Memory on single GPU, bs=1")

    parser.add_argument("--backbone", default="resnet18", type=str)
    parser.add_argument("--num_frame", default=5, type=int)
    parser.add_argument("--num_boxes", default=14, type=int)
    parser.add_argument("--num_group_tokens", default=12, type=int)
    parser.add_argument("--num_class", default=6, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--crop_size", default=5, type=int)
    parser.add_argument("--drop_rate", default=0.1, type=float)
    parser.add_argument("--gar_nheads", default=4, type=int)
    parser.add_argument("--gar_enc_layers", default=6, type=int)
    parser.add_argument("--gar_ffn_dim", default=512, type=int)
    parser.add_argument("--distance_threshold", default=0.2, type=float)
    parser.add_argument("--hoi_nheads", default=4, type=int)
    parser.add_argument("--hoi_topk", default=0, type=int)
    parser.add_argument("--hoi_mode", default="penalty", choices=["none", "bias", "hard_mask", "penalty"])
    parser.add_argument("--hoi_hard_thresh", default=None, type=float)
    parser.add_argument("--temporal_layers", default=3, type=int)
    parser.add_argument("--tcn_kernel_size", default=3, type=int)
    parser.add_argument("--tcn_dropout", default=0.1, type=float)
    parser.add_argument("--temporal_agg_mode", default="learned_pool", choices=["learned_pool", "frame_mean_main"])
    parser.add_argument("--mae_fusion", default="adaptive_two_branch",
                        choices=["none", "static_add", "static_concat", "static_pool", "adaptive_shared", "adaptive_two_branch"])
    parser.add_argument("--mae_fusion_stage", default="post_group", choices=["post_group", "pre_group"])
    parser.add_argument("--mae_version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--no_mae", action="store_true")
    parser.add_argument("--frozen_batch_norm", action="store_true")
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_blocks", default=0, type=int)

    parser.add_argument("--image_height", default=630, type=int)
    parser.add_argument("--image_width", default=1120, type=int)
    parser.add_argument("--valid_actors", default=14, type=int)

    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iters", default=50, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--ablate", action="append", default=[],
                        help="Ablation dimension, e.g. --ablate temporal_layers=1,3,5 . Can repeat.")
    parser.add_argument("--variants_json", default="", type=str,
                        help="JSON list of variants: [{\"name\":\"x\",\"overrides\":{\"key\":value}}]")
    parser.add_argument("--out_dir", default="./result/ablation_profile", type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available. Latency and memory will be CPU-only/0-memory.")

    base_cfg = default_config(args)

    if args.variants_json:
        variants = load_variants_from_json(args.variants_json)
    else:
        variants = parse_ablate_flags(args.ablate, base_cfg)

    rows = []
    for idx, variant in enumerate(variants):
        cfg = apply_overrides(base_cfg, variant["overrides"])
        cfg_ns = to_namespace(cfg)

        print(f"[{idx + 1}/{len(variants)}] Profiling: {variant['name']}")
        model = GADTR(cfg_ns).to(device)
        model.eval()

        inputs = make_dummy_batch(cfg_ns, device)

        total_params, trainable_params = count_params(model)
        flops = measure_flops(model, inputs, use_cuda=(device.type == "cuda"))
        latency_ms, peak_mem_bytes = measure_latency_and_memory(
            model, inputs, device=device, warmup=args.warmup, iters=args.iters
        )

        row = {
            "name": variant["name"],
            "params_total": int(total_params),
            "params_trainable": int(trainable_params),
            "flops": float(flops),
            "flops_g": float(flops) / 1e9,
            "latency_ms": float(latency_ms),
            "peak_mem_mb": float(peak_mem_bytes) / (1024 ** 2),
            "config": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

        print(
            f"  trainable={row['params_trainable']:,} | "
            f"FLOPs={row['flops_g']:.3f}G | "
            f"latency={row['latency_ms']:.3f} ms | "
            f"peak_mem={row['peak_mem_mb']:.2f} MB"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    json_path, csv_path = save_results(rows, out_dir)
    print(f"\nSaved results:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")


if __name__ == "__main__":
    main()
