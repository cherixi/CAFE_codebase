import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

# Ensure project root is on PYTHONPATH when running as `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloader.dataloader import read_dataset
from models.models import GADTR

ACTIVITY_NAMES = [
    "Queueing",
    "Ordering",
    "Eating/Drinking",
    "Working/Studying",
    "Fighting",
    "TakingSelfie",
    "No/Outlier",
]


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack([image for image in batch[0]])
    return tuple(batch)


def to_namespace(d):
    return SimpleNamespace(**d)


def apply_defaults(cfg):
    defaults = {
        "dataset": "cafe",
        "val_mode": False,
        "split": "place",
        "data_path": "../Dataset/",
        "image_width": 1120,
        "image_height": 630,
        "random_sampling": False,
        "num_frame": 5,
        "num_class": 6,
        "backbone": "resnet18",
        "dilation": False,
        "frozen_batch_norm": False,
        "freeze_backbone": False,
        "unfreeze_blocks": 0,
        "hidden_dim": 256,
        "num_boxes": 14,
        "crop_size": 5,
        "gar_nheads": 4,
        "gar_enc_layers": 6,
        "gar_ffn_dim": 512,
        "position_embedding": "sine",
        "num_group_tokens": 12,
        "distance_threshold": 0.2,
        "hoi_nheads": 4,
        "hoi_topk": 0,
        "hoi_mode": "penalty",
        "hoi_hard_thresh": None,
        "temporal_layers": 3,
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.1,
        "temporal_agg_mode": "learned_pool",
        "drop_rate": 0.1,
        "no_mae": False,
        "mae_fusion": "adaptive_two_branch",
        "mae_fusion_stage": "post_group",
        "mae_version": "v2",
        "videomae_feats_path": "./videomae_features_giant",
    }
    out = dict(defaults)
    out.update(cfg)
    if out.get("no_mae", False):
        out["mae_fusion"] = "none"
    out["use_mae"] = (not out.get("no_mae", False)) and out.get("mae_fusion", "none") != "none"
    out["mae_dim"] = 1408 if out.get("mae_version", "v2") == "v2" else 768
    if not out["use_mae"]:
        out["mae_dim"] = 0
    return out


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected


class SpatialBiasCollector:
    def __init__(self, max_points=300000, seed=1):
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(seed)
        self.dist_values = np.empty((0,), dtype=np.float32)
        self.bias_mean_values = np.empty((0,), dtype=np.float32)
        self.bias_per_head = None
        self.total_seen = 0
        self.use_kwargs_hook = True
        self.hook_handle = None

    def _reservoir_merge(self, old, new):
        if old.size == 0:
            if new.size <= self.max_points:
                return new
            idx = self.rng.choice(new.size, self.max_points, replace=False)
            return new[idx]
        cat = np.concatenate([old, new], axis=0)
        if cat.size <= self.max_points:
            return cat
        idx = self.rng.choice(cat.size, self.max_points, replace=False)
        return cat[idx]

    def _merge_heads(self, new_heads):
        if self.bias_per_head is None:
            self.bias_per_head = [np.empty((0,), dtype=np.float32) for _ in range(len(new_heads))]
        for i, arr in enumerate(new_heads):
            self.bias_per_head[i] = self._reservoir_merge(self.bias_per_head[i], arr)

    def _collect(self, module, x, boxes, attn_mask=None):
        with torch.no_grad():
            geom_feat, dist = module._pairwise_geometry(boxes)  # [bt,n,n,5], [bt,n,n]
            geom_bias = module.geom_mlp(geom_feat) if module.use_geom_bias and module.geom_mlp is not None else None
            if geom_bias is None:
                return

            bt, n, _, h = geom_bias.shape
            device = geom_bias.device
            valid = torch.ones((bt, n, n), dtype=torch.bool, device=device)
            diag = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
            valid = valid & (~diag)
            if attn_mask is not None:
                valid = valid & (~attn_mask.bool())

            if valid.sum().item() == 0:
                return

            dist_vals = dist[valid].detach().float().cpu().numpy()
            bias_mean_vals = geom_bias.mean(dim=-1)[valid].detach().float().cpu().numpy()
            per_head_vals = [geom_bias[..., hi][valid].detach().float().cpu().numpy() for hi in range(h)]

            self.dist_values = self._reservoir_merge(self.dist_values, dist_vals)
            self.bias_mean_values = self._reservoir_merge(self.bias_mean_values, bias_mean_vals)
            self._merge_heads(per_head_vals)
            self.total_seen += int(dist_vals.size)

    def hook_with_kwargs(self, module, args, kwargs, output):
        x = args[0]
        boxes = args[1]
        attn_mask = kwargs.get("attn_mask", None)
        self._collect(module, x, boxes, attn_mask)

    def hook_no_kwargs(self, module, args, output):
        x = args[0]
        boxes = args[1]
        self._collect(module, x, boxes, attn_mask=None)

    def attach(self, module):
        try:
            self.hook_handle = module.register_forward_hook(self.hook_with_kwargs, with_kwargs=True)
            self.use_kwargs_hook = True
        except TypeError:
            self.hook_handle = module.register_forward_hook(self.hook_no_kwargs)
            self.use_kwargs_hook = False

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class GateCollector:
    def __init__(self, max_points=300000, seed=1, num_classes=7):
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(seed)
        self.num_classes = int(num_classes)

        self.actor_gate_all = np.empty((0,), dtype=np.float32)
        self.group_gate_all = np.empty((0,), dtype=np.float32)
        self.actor_gate_by_class = [np.empty((0,), dtype=np.float32) for _ in range(self.num_classes)]
        self.group_gate_by_class = [np.empty((0,), dtype=np.float32) for _ in range(self.num_classes)]
        self.clip_actor_group_pairs = np.empty((0, 2), dtype=np.float32)

        self.latest_actor_gate = None
        self.latest_group_gate = None
        self.hook_handle = None
        self.supported = True

    def _reservoir_merge(self, old, new):
        if new.size == 0:
            return old
        if old.size == 0:
            if new.size <= self.max_points:
                return new
            idx = self.rng.choice(new.size, self.max_points, replace=False)
            return new[idx]
        cat = np.concatenate([old, new], axis=0)
        if cat.size <= self.max_points:
            return cat
        idx = self.rng.choice(cat.size, self.max_points, replace=False)
        return cat[idx]

    def _stash_from_adapter(self, module, actor_tokens, group_tokens, global_feat):
        if module.fusion not in ["adaptive_two_branch", "adaptive_shared"]:
            self.supported = False
            self.latest_actor_gate = None
            self.latest_group_gate = None
            return
        with torch.no_grad():
            global_emb = module.proj(global_feat)
            _, n, _ = actor_tokens.shape
            _, k, _ = group_tokens.shape
            global_actor = global_emb.unsqueeze(1).expand(-1, n, -1)
            global_group = global_emb.unsqueeze(1).expand(-1, k, -1)

            if module.fusion == "adaptive_two_branch":
                alpha = module.actor_gate(torch.cat([actor_tokens, global_actor], dim=-1))
                beta = module.group_gate(torch.cat([group_tokens, global_group], dim=-1))
            else:
                alpha = module.shared_gate(torch.cat([actor_tokens, global_actor], dim=-1))
                beta = module.shared_gate(torch.cat([group_tokens, global_group], dim=-1))

            self.latest_actor_gate = alpha.detach().squeeze(-1).float().cpu().numpy()  # [bt, n]
            self.latest_group_gate = beta.detach().squeeze(-1).float().cpu().numpy()   # [bt, k]

    def hook_no_kwargs(self, module, args, output):
        actor_tokens, group_tokens, global_feat = args
        self._stash_from_adapter(module, actor_tokens, group_tokens, global_feat)

    def attach(self, module):
        self.hook_handle = module.register_forward_hook(self.hook_no_kwargs)

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def consume(self, outputs, dummy_mask, num_frame):
        if self.latest_actor_gate is None or self.latest_group_gate is None:
            return

        pred_actions = outputs["pred_actions"].argmax(dim=-1).detach().cpu().numpy()      # [b, n]
        pred_activities = outputs["pred_activities"].argmax(dim=-1).detach().cpu().numpy() # [b, k]
        dummy_np = dummy_mask.detach().cpu().numpy().astype(bool)                          # [b, n]

        bs, n = pred_actions.shape
        k = pred_activities.shape[1]
        t = int(num_frame)

        actor_gate = self.latest_actor_gate.reshape(bs, t, n)
        group_gate = self.latest_group_gate.reshape(bs, t, k)

        self.actor_gate_all = self._reservoir_merge(self.actor_gate_all, actor_gate.reshape(-1))
        self.group_gate_all = self._reservoir_merge(self.group_gate_all, group_gate.reshape(-1))

        pairs = []
        for b in range(bs):
            valid_actor = ~dummy_np[b]
            if valid_actor.any():
                pairs.append([
                    float(actor_gate[b, :, valid_actor].mean()),
                    float(group_gate[b].mean()),
                ])

            for i in range(n):
                if dummy_np[b, i]:
                    continue
                cls = int(pred_actions[b, i])
                if cls < 0 or cls >= self.num_classes:
                    continue
                vals = actor_gate[b, :, i].reshape(-1)
                self.actor_gate_by_class[cls] = self._reservoir_merge(self.actor_gate_by_class[cls], vals)

            for j in range(k):
                cls = int(pred_activities[b, j])
                if cls < 0 or cls >= self.num_classes:
                    continue
                vals = group_gate[b, :, j].reshape(-1)
                self.group_gate_by_class[cls] = self._reservoir_merge(self.group_gate_by_class[cls], vals)

        if pairs:
            pair_arr = np.array(pairs, dtype=np.float32)
            if self.clip_actor_group_pairs.size == 0:
                self.clip_actor_group_pairs = pair_arr
            else:
                cat = np.concatenate([self.clip_actor_group_pairs, pair_arr], axis=0)
                if len(cat) > self.max_points:
                    idx = self.rng.choice(len(cat), self.max_points, replace=False)
                    cat = cat[idx]
                self.clip_actor_group_pairs = cat


def plot_hist(values, out_path, title, xlabel):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=80, color="#2E7D32", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_head_boxplot(per_head, out_path):
    plt.figure(figsize=(9, 5))
    labels = [f"h{i}" for i in range(len(per_head))]
    plt.boxplot(per_head, labels=labels, showfliers=False)
    plt.title("Geom Bias Per Head Distribution")
    plt.xlabel("Head")
    plt.ylabel("Bias value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bias_vs_dist(dist_values, bias_values, out_path):
    if dist_values.size == 0:
        return
    max_d = float(np.percentile(dist_values, 99.5))
    max_d = max(max_d, 1e-3)
    bins = np.linspace(0.0, max_d, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, stds = [], []
    for i in range(len(bins) - 1):
        m = (dist_values >= bins[i]) & (dist_values < bins[i + 1])
        if m.sum() == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(np.mean(bias_values[m])))
            stds.append(float(np.std(bias_values[m])))

    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)

    plt.figure(figsize=(8, 5))
    valid = ~np.isnan(means)
    plt.plot(centers[valid], means[valid], color="#1565C0", linewidth=2, label="mean bias")
    plt.fill_between(
        centers[valid],
        means[valid] - stds[valid],
        means[valid] + stds[valid],
        color="#90CAF9",
        alpha=0.35,
        label="+/-1 std",
    )
    plt.title("Mean Geom Bias vs Distance")
    plt.xlabel("Distance")
    plt.ylabel("Bias (mean over heads)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gate_hist(actor_values, group_values, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(actor_values, bins=80, alpha=0.65, label="actor gate", color="#1976D2")
    plt.hist(group_values, bins=80, alpha=0.65, label="group gate", color="#D32F2F")
    plt.title("Gate Distribution")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gate_by_class(gate_by_class, out_path, title, class_names):
    vals = []
    labels = []
    for i, arr in enumerate(gate_by_class):
        if arr.size > 0:
            vals.append(arr)
            labels.append(class_names[i] if i < len(class_names) else f"class_{i}")
    if not vals:
        return
    plt.figure(figsize=(11, 5))
    plt.boxplot(vals, labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("Gate value")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gate_correlation(pairs, out_path):
    if pairs.size == 0:
        return
    x = pairs[:, 0]
    y = pairs[:, 1]
    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, s=14, alpha=0.45, color="#5D4037")
    if len(x) >= 2:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float("nan")
    plt.title(f"Actor vs Group Gate (clip mean), corr={corr:.3f}")
    plt.xlabel("Actor gate mean")
    plt.ylabel("Group gate mean")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize learned spatial bias from a trained checkpoint.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to .pth checkpoint")
    parser.add_argument("--args_json", required=True, type=str, help="Path to args.json used for training")
    parser.add_argument("--data_path", default="", type=str, help="Override data path from args.json")
    parser.add_argument("--split", default="", type=str, help="Override split from args.json")
    parser.add_argument("--val_mode", action="store_true", help="Override val_mode=True")
    parser.add_argument(
        "--max_batches",
        default=-1,
        type=int,
        help="Max number of batches to run. <=0 means full test epoch (default).",
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--max_points", default=300000, type=int, help="Max sampled pair points to keep")
    parser.add_argument("--out_dir", default="./result/spatial_bias_viz", type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable, running on CPU.")

    train_args = json.loads(Path(args.args_json).read_text(encoding="utf-8"))
    cfg = apply_defaults(train_args)
    if args.data_path:
        cfg["data_path"] = args.data_path
    if args.split:
        cfg["split"] = args.split
    if args.val_mode:
        cfg["val_mode"] = True
    cfg_ns = to_namespace(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, test_set = read_dataset(cfg_ns)
    test_loader = data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GADTR(cfg_ns).to(device)
    model.eval()
    missing, unexpected = load_checkpoint(model, args.model_path, device)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    spatial_collector = None
    if model.frame_graph is not None and getattr(model.frame_graph, "use_geom_bias", False):
        spatial_collector = SpatialBiasCollector(max_points=args.max_points, seed=args.seed)
        spatial_collector.attach(model.frame_graph)
        print("Spatial bias collector: enabled")
    else:
        print("Spatial bias collector: skipped (frame_graph geom bias disabled)")

    gate_collector = None
    if cfg_ns.use_mae and model.videomae_adapter is not None:
        fusion = getattr(model.videomae_adapter, "fusion", "none")
        if fusion in ["adaptive_two_branch", "adaptive_shared"]:
            gate_collector = GateCollector(max_points=args.max_points, seed=args.seed, num_classes=cfg_ns.num_class + 1)
            gate_collector.attach(model.videomae_adapter)
            print(f"Gate collector: enabled (fusion={fusion})")
        else:
            print(f"Gate collector: skipped (fusion={fusion}, no adaptive gate)")
    else:
        print("Gate collector: skipped (MAE disabled or adapter missing)")

    if args.max_batches <= 0:
        run_limit = len(test_loader)
        print(f"Collecting stats from full test epoch ({run_limit} batches)...")
    else:
        run_limit = min(args.max_batches, len(test_loader))
        print(f"Collecting stats from up to {run_limit} batches...")

    with torch.no_grad():
        for bi, (images, targets, infos) in enumerate(test_loader):
            if bi >= run_limit:
                break
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = torch.stack([t["boxes"] for t in targets])
            dummy_mask = torch.stack([t["actions"] == cfg_ns.num_class + 1 for t in targets], dim=0).squeeze(1)

            mae_feats = None
            if cfg_ns.use_mae and "mae_feats" in targets[0]:
                mae_feats = torch.stack([t["mae_feats"] for t in targets])

            outputs = model(images, boxes, dummy_mask, mae_feats)
            if gate_collector is not None:
                gate_collector.consume(outputs, dummy_mask, cfg_ns.num_frame)

            if (bi + 1) % 10 == 0:
                print(f"  processed {bi + 1} batches")

    if spatial_collector is not None:
        spatial_collector.detach()
    if gate_collector is not None:
        gate_collector.detach()

    summary = {
        "model_path": str(args.model_path),
        "args_json": str(args.args_json),
        "num_batches_used": run_limit,
        "spatial_bias_enabled": spatial_collector is not None,
        "gate_enabled": gate_collector is not None,
    }

    if spatial_collector is not None and spatial_collector.bias_per_head is not None and len(spatial_collector.bias_per_head) > 0:
        bias_all = np.concatenate(spatial_collector.bias_per_head, axis=0)
        dist_all = spatial_collector.dist_values
        bias_mean_all = spatial_collector.bias_mean_values

        sigma = float(torch.exp(model.frame_graph.log_sigma.detach().cpu()).item())
        log_sigma = float(model.frame_graph.log_sigma.detach().cpu().item())

        plot_hist(
            bias_all,
            out_dir / "geom_bias_hist.png",
            "Geom Bias Distribution (all heads)",
            "bias value",
        )
        plot_head_boxplot(spatial_collector.bias_per_head, out_dir / "geom_bias_per_head_boxplot.png")
        plot_bias_vs_dist(dist_all, bias_mean_all, out_dir / "geom_bias_vs_distance.png")

        np.save(out_dir / "geom_bias_all.npy", bias_all)
        np.save(out_dir / "geom_bias_dist.npy", dist_all)
        np.save(out_dir / "geom_bias_mean_over_heads.npy", bias_mean_all)
        np.savez(
            out_dir / "geom_bias_per_head.npz",
            **{f"h{i}": arr for i, arr in enumerate(spatial_collector.bias_per_head)}
        )

        summary["spatial_bias"] = {
            "total_pairs_seen_before_sampling": spatial_collector.total_seen,
            "sampled_pairs_kept": int(dist_all.size),
            "hook_with_kwargs": spatial_collector.use_kwargs_hook,
            "sigma": sigma,
            "log_sigma": log_sigma,
            "bias_all_mean": float(np.mean(bias_all)),
            "bias_all_std": float(np.std(bias_all)),
            "dist_mean": float(np.mean(dist_all)),
            "dist_std": float(np.std(dist_all)),
            "per_head_mean": [float(np.mean(x)) for x in spatial_collector.bias_per_head],
            "per_head_std": [float(np.std(x)) for x in spatial_collector.bias_per_head],
        }

        print("Saved spatial bias visualization:")
        print(f"  {out_dir / 'geom_bias_hist.png'}")
        print(f"  {out_dir / 'geom_bias_per_head_boxplot.png'}")
        print(f"  {out_dir / 'geom_bias_vs_distance.png'}")
        print(f"  sigma={sigma:.6f} (log_sigma={log_sigma:.6f})")
    else:
        print("Spatial bias outputs: skipped/no samples.")

    if gate_collector is not None and gate_collector.actor_gate_all.size > 0 and gate_collector.group_gate_all.size > 0:
        plot_gate_hist(gate_collector.actor_gate_all, gate_collector.group_gate_all, out_dir / "gate_hist.png")
        class_names = ACTIVITY_NAMES[: cfg_ns.num_class] + [ACTIVITY_NAMES[-1]]
        plot_gate_by_class(
            gate_collector.actor_gate_by_class,
            out_dir / "actor_gate_by_pred_class_boxplot.png",
            "Actor Gate by Predicted Action Class",
            class_names,
        )
        plot_gate_by_class(
            gate_collector.group_gate_by_class,
            out_dir / "group_gate_by_pred_class_boxplot.png",
            "Group Gate by Predicted Activity Class",
            class_names,
        )
        plot_gate_correlation(gate_collector.clip_actor_group_pairs, out_dir / "actor_group_gate_correlation.png")

        np.save(out_dir / "actor_gate_all.npy", gate_collector.actor_gate_all)
        np.save(out_dir / "group_gate_all.npy", gate_collector.group_gate_all)
        np.save(out_dir / "actor_group_gate_clip_pairs.npy", gate_collector.clip_actor_group_pairs)
        np.savez(
            out_dir / "actor_gate_by_pred_class.npz",
            **{f"class_{i}": arr for i, arr in enumerate(gate_collector.actor_gate_by_class)}
        )
        np.savez(
            out_dir / "group_gate_by_pred_class.npz",
            **{f"class_{i}": arr for i, arr in enumerate(gate_collector.group_gate_by_class)}
        )

        summary["gates"] = {
            "actor_gate_mean": float(np.mean(gate_collector.actor_gate_all)),
            "actor_gate_std": float(np.std(gate_collector.actor_gate_all)),
            "group_gate_mean": float(np.mean(gate_collector.group_gate_all)),
            "group_gate_std": float(np.std(gate_collector.group_gate_all)),
            "actor_gate_class_counts": [int(x.size) for x in gate_collector.actor_gate_by_class],
            "group_gate_class_counts": [int(x.size) for x in gate_collector.group_gate_by_class],
            "clip_pair_count": int(gate_collector.clip_actor_group_pairs.shape[0]),
        }
        if gate_collector.clip_actor_group_pairs.shape[0] >= 2:
            corr = float(np.corrcoef(gate_collector.clip_actor_group_pairs[:, 0], gate_collector.clip_actor_group_pairs[:, 1])[0, 1])
            summary["gates"]["actor_group_clip_corr"] = corr

        print("Saved gate visualization:")
        print(f"  {out_dir / 'gate_hist.png'}")
        print(f"  {out_dir / 'actor_gate_by_pred_class_boxplot.png'}")
        print(f"  {out_dir / 'group_gate_by_pred_class_boxplot.png'}")
        print(f"  {out_dir / 'actor_group_gate_correlation.png'}")
    else:
        print("Gate outputs: skipped/no samples.")

    (out_dir / "spatial_and_gate_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved summary: {out_dir / 'spatial_and_gate_stats.json'}")


if __name__ == "__main__":
    main()
