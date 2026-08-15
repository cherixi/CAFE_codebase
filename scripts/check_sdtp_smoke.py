#!/usr/bin/env python
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hoi_graph import StaticDynamicTemporalPool


def _assert_finite(name, value):
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def main():
    torch.manual_seed(7)
    pool = StaticDynamicTemporalPool(
        d_model=32,
        hidden_dim=16,
        dropout=0.0,
        dynamic_scale_init=0.1,
        dynamic_scale_max=0.5,
    )

    tokens = torch.randn(6, 8, 32, requires_grad=True)
    padding = torch.zeros(6, 8, dtype=torch.bool)
    padding[0, -2:] = True
    clip, diag = pool(tokens, key_padding_mask=padding)

    assert clip.shape == (6, 32)
    _assert_finite("clip", clip)
    for name, value in diag.items():
        _assert_finite(name, value)
    assert 0.0 < diag["dynamic_scale"].item() < 0.5

    loss = clip.square().mean()
    loss.backward()
    if tokens.grad is None:
        raise AssertionError("input gradient is missing")
    _assert_finite("input gradient", tokens.grad)

    stationary = torch.randn(3, 1, 32).repeat(1, 8, 1)
    stationary_clip, stationary_diag = pool(stationary)
    _assert_finite("stationary clip", stationary_clip)
    assert stationary_diag["dynamic_ratio"].item() < 1e-7

    single_clip, single_diag = pool(torch.randn(3, 1, 32))
    assert single_clip.shape == (3, 32)
    _assert_finite("single-frame clip", single_clip)
    assert single_diag["dynamic_ratio"].item() < 1e-7

    print(
        "SDTP smoke passed: "
        f"shape={tuple(clip.shape)}, "
        f"scale={diag['dynamic_scale'].item():.4f}, "
        f"gate={diag['gate_mean'].item():.4f}, "
        f"dynamic_ratio={diag['dynamic_ratio'].item():.4f}"
    )


if __name__ == "__main__":
    main()
