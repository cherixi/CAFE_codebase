import io
import json
import os
from typing import Dict, List

import matplotlib

# Use non-interactive backend for servers/CLI
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_args(path: str, args: Dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        clean_args = {k: _to_serializable(v) for k, v in args.items()}
        json.dump(clean_args, f, indent=2)


def _to_serializable(val):
    # file-like objects: store path/name
    if isinstance(val, io.IOBase):
        return getattr(val, "name", str(val))
    # try default json conversion
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)


def update_history(history: Dict, split: str, epoch: int, metrics: Dict):
    entry = {"epoch": epoch}
    entry.update({k: float(v) for k, v in metrics.items()})
    history.setdefault(split, []).append(entry)
    return history


def update_best(best: Dict, epoch: int, metrics: Dict, keys=("group_mAP_0.5", "group_mAP_1.0", "loss")):
    for k in keys:
        if k not in metrics:
            continue
        val = float(metrics[k])
        if k not in best or (k == "loss" and val < best[k]["value"]) or (k != "loss" and val > best[k]["value"]):
            best[k] = {"epoch": epoch, "value": val}
    return best


def save_summary(path: str, args: Dict, history: Dict, best: Dict):
    ensure_dir(os.path.dirname(path))
    summary = {"args": args, "history": history, "best": best}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def plot_curves(path: str, history: Dict):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(10, 6))

    # Loss curves
    for split in ["train", "val"]:
        if split in history:
            epochs = [h["epoch"] for h in history[split]]
            losses = [h["loss"] for h in history[split] if "loss" in h]
            if len(epochs) == len(losses) and len(epochs) > 0:
                plt.plot(epochs, losses, label=f"{split}_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    loss_path = path.replace(".png", "_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # mAP curves (if available)
    plt.figure(figsize=(10, 6))
    for split in ["val"]:
        if split in history:
            epochs = [h["epoch"] for h in history[split]]
            map1 = [h["group_mAP_1.0"] for h in history[split] if "group_mAP_1.0" in h]
            map05 = [h["group_mAP_0.5"] for h in history[split] if "group_mAP_0.5" in h]
            if len(epochs) == len(map1) and len(epochs) > 0:
                plt.plot(epochs, map1, label=f"{split}_mAP@1.0")
            if len(epochs) == len(map05) and len(epochs) > 0:
                plt.plot(epochs, map05, label=f"{split}_mAP@0.5")

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    map_path = path.replace(".png", "_map.png")
    plt.savefig(map_path, bbox_inches="tight")
    plt.close()
