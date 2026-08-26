import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.dinov2_multilevel_adapter import MultiDepthActorAdapter


def main():
    torch.manual_seed(7)
    bt, num_actors = 2, 3
    channels, hidden_dim = 32, 16
    crop_size, num_layers = 3, 3

    module = MultiDepthActorAdapter(
        in_channels=channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        crop_size=crop_size,
        bottleneck_dim=8,
        dropout=0.0,
        scale_init=0.05,
        scale_max=0.5,
    )
    module.train()

    feature_maps = [
        torch.randn(bt, channels, 8, 8, requires_grad=True)
        for _ in range(num_layers)
    ]
    boxes_list = [
        torch.tensor(
            [[0.5, 0.5, 4.5, 6.5], [2.0, 1.0, 7.0, 7.0], [1.0, 2.0, 5.5, 7.5]],
            dtype=torch.float32,
        )
        for _ in range(bt)
    ]
    base_tokens = torch.randn(bt, num_actors, hidden_dim, requires_grad=True)
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])

    fused, diag = module(
        intermediate_features=feature_maps,
        boxes_list=boxes_list,
        base_actor_tokens=base_tokens,
        valid_actor_mask=valid_mask,
    )

    assert fused.shape == base_tokens.shape
    assert torch.isfinite(fused).all()
    assert diag['layer_weight_mean'].shape == (1, num_layers)
    assert torch.allclose(
        diag['layer_weight_mean'].sum(dim=-1),
        torch.ones(1),
        atol=1e-5,
    )
    assert 0.0 < diag['scale'].item() < 0.5
    assert torch.isfinite(diag['gate_entropy']).all()
    assert torch.isfinite(diag['delta_ratio']).all()

    loss = fused.square().mean()
    loss.backward()
    assert base_tokens.grad is not None and torch.isfinite(base_tokens.grad).all()
    assert all(feature.grad is not None for feature in feature_maps)
    assert module.layer_adapters[0].proj.weight.grad is not None

    print('DINOv2 multi-level actor adapter smoke check passed')
    print('fused_shape:', tuple(fused.shape))
    print('scale:', round(diag['scale'].item(), 6))
    print('weights:', [round(v, 6) for v in diag['layer_weight_mean'][0].tolist()])
    print('delta_ratio:', round(diag['delta_ratio'].item(), 6))
    print('gate_entropy:', round(diag['gate_entropy'].item(), 6))


if __name__ == '__main__':
    main()
