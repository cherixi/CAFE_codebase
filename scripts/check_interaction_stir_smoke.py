#!/usr/bin/env python3
"""
Smoke test for IA-STIR / FrameInteractionGraph.

This is a structural CPU check. It does not require CAFE data or a detector pkl.
It verifies that anchor-only edge modulation produces finite actor tokens and
bounded diagnostics, and that the no-anchor path safely falls back to zero
interaction evidence.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hoi_graph import FrameInteractionGraph


def make_actor_mask(actor_valid: torch.Tensor) -> torch.Tensor:
    valid_pairs = actor_valid.unsqueeze(2) & actor_valid.unsqueeze(1)
    mask = ~valid_pairs
    diag_idx = torch.arange(actor_valid.shape[1], device=actor_valid.device)
    mask[:, diag_idx, diag_idx] = False
    return mask


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains NaN/Inf")


def run_once(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    b_tokens, n, m, d = args.batch_tokens, args.num_actors, args.num_objects, args.hidden_dim
    assert d % args.nheads == 0

    graph = FrameInteractionGraph(
        d_model=d,
        nhead=args.nheads,
        dropout=0.0,
        topk=0,
        use_geom_bias=True,
        use_logit_penalty=True,
        use_anchors=True,
        anchor_scale_max=args.anchor_scale_max,
        anchor_scale_init=args.anchor_scale_init,
        anchor_bias_clip=args.anchor_bias_clip,
        anchor_attn_tau=args.anchor_attn_tau,
        anchor_source=args.anchor_source,
    ).eval()

    actor_tokens = torch.randn(b_tokens, n, d)
    actor_boxes = torch.rand(b_tokens, n, 4)
    actor_boxes[..., 2:] = actor_boxes[..., 2:].mul(0.25).add(0.05)
    actor_boxes[..., :2] = actor_boxes[..., :2].mul(0.8).add(0.1)

    actor_valid = torch.ones(b_tokens, n, dtype=torch.bool)
    actor_valid[-1, -1] = False
    attn_mask = make_actor_mask(actor_valid)

    object_tokens = torch.randn(b_tokens, m, d)
    obj_xy1 = torch.rand(b_tokens, m, 2).mul(0.75)
    obj_wh = torch.rand(b_tokens, m, 2).mul(0.2).add(0.05)
    object_boxes = torch.cat((obj_xy1, (obj_xy1 + obj_wh).clamp(max=0.98)), dim=-1)
    object_scores = torch.rand(b_tokens, m)
    object_valid = torch.ones(b_tokens, m, dtype=torch.bool)
    object_family = torch.zeros(b_tokens, m, dtype=torch.long)
    object_family[:, 0] = 5  # table
    object_family[:, 1] = 4  # service
    object_family[:, 2:] = 1

    out, attn, diag = graph(
        actor_tokens,
        actor_boxes,
        attn_mask=attn_mask,
        actor_valid_mask=actor_valid,
        object_tokens=object_tokens,
        object_boxes_xyxy=object_boxes,
        object_scores=object_scores,
        object_family_id=object_family,
        object_valid_mask=object_valid,
    )

    if out.shape != actor_tokens.shape:
        raise AssertionError(f"unexpected output shape: {out.shape}")
    if attn.shape != (b_tokens, args.nheads, n, n):
        raise AssertionError(f"unexpected attention shape: {attn.shape}")
    assert_finite("out", out)
    assert_finite("attn", attn)
    for key, value in diag.items():
        assert_finite(key, value)

    scale = float(diag["interaction_anchor_scale_mean"].item())
    expected_scale = args.anchor_scale_max / (1.0 + math.exp(-args.anchor_scale_init))
    if abs(scale - expected_scale) > 1e-5:
        raise AssertionError(f"scale mismatch: {scale} vs {expected_scale}")

    max_allowed = args.anchor_scale_max * args.anchor_bias_clip + 1e-6
    if abs(float(diag["interaction_anchor_bias_max"].item())) > max_allowed:
        raise AssertionError("anchor_bias_max exceeds configured bound")
    if abs(float(diag["interaction_anchor_bias_min"].item())) > max_allowed:
        raise AssertionError("anchor_bias_min exceeds configured bound")

    geom_feat, _ = graph._pairwise_geometry(actor_boxes)
    anchor_bias, _, _, _ = graph._anchor_edge_bias(
        x=actor_tokens,
        boxes=actor_boxes,
        geom_feat=geom_feat,
        actor_valid_mask=actor_valid,
        object_tokens=object_tokens,
        object_boxes_xyxy=object_boxes,
        object_scores=object_scores,
        object_family_id=object_family,
        object_valid_mask=object_valid,
    )
    diag_idx = torch.arange(n)
    if anchor_bias[:, :, diag_idx, diag_idx].abs().max().item() != 0.0:
        raise AssertionError("anchor edge bias diagonal should be exactly zero")

    loss = out.square().mean()
    loss.backward()
    if graph.anchor_scale_logit.grad is None:
        raise AssertionError("anchor_scale_logit did not receive gradients")
    assert_finite("anchor_scale_logit.grad", graph.anchor_scale_logit.grad)

    no_anchor_family = torch.zeros_like(object_family)
    _, _, no_anchor_diag = graph(
        actor_tokens,
        actor_boxes,
        attn_mask=attn_mask,
        actor_valid_mask=actor_valid,
        object_tokens=object_tokens,
        object_boxes_xyxy=object_boxes,
        object_scores=object_scores,
        object_family_id=no_anchor_family,
        object_valid_mask=object_valid,
    )
    if float(no_anchor_diag["interaction_anchor_bias_abs_mean"].item()) != 0.0:
        raise AssertionError("no-anchor path should have zero anchor bias")

    print("IA-STIR smoke passed")
    print(f"  out: {tuple(out.shape)}")
    print(f"  attn: {tuple(attn.shape)}")
    print(f"  anchor_scale_mean: {scale:.6f}")
    print(f"  anchor_bias_abs_mean: {float(diag['interaction_anchor_bias_abs_mean'].item()):.6f}")
    print(f"  anchor_shared_table_mean: {float(diag['interaction_anchor_shared_table_mean'].item()):.6f}")
    print(f"  anchor_valid_per_actor: {float(diag['interaction_anchor_valid_per_actor'].item()):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test IA-STIR FrameInteractionGraph.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_tokens", default=3, type=int)
    parser.add_argument("--num_actors", default=5, type=int)
    parser.add_argument("--num_objects", default=6, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--nheads", default=4, type=int)
    parser.add_argument("--anchor_scale_max", default=0.5, type=float)
    parser.add_argument("--anchor_scale_init", default=-6.0, type=float)
    parser.add_argument("--anchor_bias_clip", default=2.0, type=float)
    parser.add_argument("--anchor_attn_tau", default=3.0, type=float)
    parser.add_argument("--anchor_source", default="gdino", choices=["gdino", "yolo"])
    args = parser.parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
