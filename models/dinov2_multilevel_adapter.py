import math

import torch
import torch.nn as nn
from torchvision.ops import RoIAlign


class _ActorRoIAdapter(nn.Module):
    def __init__(self, in_channels, bottleneck_dim, hidden_dim, crop_size, dropout):
        super().__init__()
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, bottleneck_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(bottleneck_dim * crop_size * crop_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.channel_adapter[0].weight, nonlinearity="linear")
        # Start close to the historical final-layer-only path.
        nn.init.normal_(self.proj.weight, std=1e-3)
        nn.init.zeros_(self.proj.bias)

    def forward(self, roi_features):
        x = self.channel_adapter(roi_features)
        x = x.flatten(1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(x)


class MultiDepthActorAdapter(nn.Module):
    """Fuse frozen DINOv2 intermediate features at actor-RoI granularity."""

    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_layers,
        crop_size,
        bottleneck_dim=64,
        dropout=0.1,
        scale_init=0.05,
        scale_max=0.5,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if not 0.0 <= scale_init < scale_max:
            raise ValueError("scale_init must satisfy 0 <= scale_init < scale_max")

        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.scale_max = float(scale_max)
        self.roi_align = RoIAlign(
            output_size=(crop_size, crop_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        self.layer_adapters = nn.ModuleList(
            [
                _ActorRoIAdapter(
                    in_channels=in_channels,
                    bottleneck_dim=bottleneck_dim,
                    hidden_dim=hidden_dim,
                    crop_size=crop_size,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        gate_hidden = max(hidden_dim // 2, 32)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.layer_bias = nn.Parameter(torch.zeros(self.num_layers))

        if scale_init == 0.0:
            scale_logit = -12.0
        else:
            ratio = min(max(scale_init / scale_max, 1e-6), 1.0 - 1e-6)
            scale_logit = math.log(ratio / (1.0 - ratio))
        self.scale_logit = nn.Parameter(torch.tensor(scale_logit, dtype=torch.float32))
        self.fusion_drop = nn.Dropout(dropout)

    def effective_scale(self):
        return self.scale_max * torch.sigmoid(self.scale_logit)

    def forward(self, intermediate_features, boxes_list, base_actor_tokens, valid_actor_mask=None):
        """
        Args:
            intermediate_features: sequence of [BT, C, Hf, Wf] feature maps.
            boxes_list: BT tensors of actor boxes in feature-map xyxy coordinates.
            base_actor_tokens: [BT, N, H] final-layer actor tokens.
            valid_actor_mask: optional [BT, N] mask used only for diagnostics.
        """
        if len(intermediate_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} intermediate maps, got {len(intermediate_features)}"
            )

        bt, n, hidden_dim = base_actor_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}"
            )

        layer_tokens = []
        for feature_map, adapter in zip(intermediate_features, self.layer_adapters):
            roi_features = self.roi_align(feature_map, boxes_list)
            token = adapter(roi_features).reshape(bt, n, hidden_dim)
            layer_tokens.append(token)
        layer_tokens = torch.stack(layer_tokens, dim=2)  # [BT, N, L, H]

        base_expanded = base_actor_tokens.unsqueeze(2).expand_as(layer_tokens)
        gate_input = torch.cat([base_expanded, layer_tokens], dim=-1)
        gate_logits = self.gate(gate_input).squeeze(-1)
        gate_logits = gate_logits + self.layer_bias.view(1, 1, -1)
        layer_weights = torch.softmax(gate_logits, dim=-1)

        adapter_delta = torch.sum(layer_weights.unsqueeze(-1) * layer_tokens, dim=2)
        scale = self.effective_scale()
        fused_tokens = base_actor_tokens + scale * self.fusion_drop(adapter_delta)

        if valid_actor_mask is None:
            valid_actor_mask = torch.ones(
                bt, n, dtype=torch.bool, device=base_actor_tokens.device
            )
        else:
            valid_actor_mask = valid_actor_mask.bool()
        valid_weights = valid_actor_mask.to(layer_weights.dtype)
        valid_count = valid_weights.sum().clamp_min(1.0)
        layer_weight_mean = (
            layer_weights * valid_weights.unsqueeze(-1)
        ).sum(dim=(0, 1)) / valid_count

        entropy = -(layer_weights.clamp_min(1e-8).log() * layer_weights).sum(dim=-1)
        entropy_mean = (entropy * valid_weights).sum() / valid_count
        base_norm = base_actor_tokens.norm(dim=-1)
        delta_norm = (scale * adapter_delta).norm(dim=-1)
        delta_ratio = (
            (delta_norm / base_norm.clamp_min(1e-6)) * valid_weights
        ).sum() / valid_count

        diagnostics = {
            "scale": scale.reshape(1),
            "layer_weight_mean": layer_weight_mean.reshape(1, self.num_layers),
            "gate_entropy": entropy_mean.reshape(1),
            "delta_ratio": delta_ratio.reshape(1),
        }
        return fused_tokens, diagnostics
