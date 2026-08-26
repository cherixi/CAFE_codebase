# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        if args.frozen_batch_norm:
            backbone = getattr(torchvision.models, args.backbone)(
                replace_stride_with_dilation=[False, False, args.dilation],
                pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, args.backbone)(
                replace_stride_with_dilation=[False, False, args.dilation],
                pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})

    def forward(self, x):
        x = self.body(x)["0"]

        return x


class DinoV2Backbone(nn.Module):
    """
    DINOv2 Backbone wrapper that outputs 4D feature maps compatible with existing pipeline.
    Supports partial unfreezing of the last N transformer blocks.
    """
    def __init__(self, args):
        super().__init__()
        # 确定模型名称
        model_name = args.backbone if 'dinov2' in args.backbone else 'dinov2_vits14'
        print(f"[DinoV2Backbone] Loading model: {model_name}")
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # 根据模型型号自动设置 num_channels
        if 'vits' in model_name:
            self.num_channels = 384
        elif 'vitb' in model_name:
            self.num_channels = 768
        elif 'vitl' in model_name:
            self.num_channels = 1024
        elif 'vitg' in model_name:
            self.num_channels = 1536
        else:
            raise ValueError(f"Unknown DINOv2 model: {model_name}")
        
        print(f"[DinoV2Backbone] num_channels = {self.num_channels}")
        
        self.patch_size = 14

        self.use_multilevel_adapter = bool(
            getattr(args, 'use_dinov2_multilevel_adapter', False)
        )
        raw_adapter_layers = getattr(args, 'dinov2_adapter_layers', '3,6,9')
        if isinstance(raw_adapter_layers, str):
            adapter_layers = [
                int(item.strip())
                for item in raw_adapter_layers.split(',')
                if item.strip()
            ]
        else:
            adapter_layers = [int(item) for item in raw_adapter_layers]
        self.intermediate_layer_indices = tuple(sorted(set(adapter_layers)))
        
        # 冻结/解冻策略
        freeze_backbone = getattr(args, 'freeze_backbone', False)
        unfreeze_blocks = getattr(args, 'unfreeze_blocks', 0)  # 默认解冻 0 层
        
        if freeze_backbone:
            # 完全冻结所有参数
            print("[DinoV2Backbone] Freezing ALL backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif unfreeze_blocks > 0:
            # 部分解冻：先冻结所有，再解冻最后 N 层 blocks
            print(f"[DinoV2Backbone] Partial unfreezing: freezing all, then unfreezing last {unfreeze_blocks} blocks")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # 解冻最后 unfreeze_blocks 个 transformer blocks
            total_blocks = len(self.backbone.blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i >= total_blocks - unfreeze_blocks:
                    for param in block.parameters():
                        param.requires_grad = True
                    print(f"  - Unfreezing block {i}")
            
            # 解冻最终的 Norm 层
            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                print("  - Unfreezing final norm layer")
        else:
            # 全部解冻（全参数微调）
            print("[DinoV2Backbone] All backbone parameters are TRAINABLE (full fine-tuning)")

        total_blocks = len(self.backbone.blocks)
        if self.use_multilevel_adapter:
            if not self.intermediate_layer_indices:
                raise ValueError("dinov2_adapter_layers must contain at least one layer")
            invalid_layers = [
                idx for idx in self.intermediate_layer_indices
                if idx < 0 or idx >= total_blocks - 1
            ]
            if invalid_layers:
                raise ValueError(
                    "DINOv2 adapter layers must be zero-based intermediate block indices "
                    f"in [0, {total_blocks - 2}], got {invalid_layers}"
                )
            print(
                "[DinoV2Backbone] Multi-level outputs: blocks "
                + ", ".join(str(idx) for idx in self.intermediate_layer_indices)
                + f"; final block {total_blocks - 1} remains the base feature"
            )

    def forward(self, x):
        """
        Args:
            x: [B*T, 3, H, W]
        Returns:
            feature_map: [B*T, C, H/14, W/14]
        """
        b, c, h, w = x.shape
        p = self.patch_size
        
        # 尺寸对齐：确保 H, W 是 patch_size 的倍数
        if h % p != 0 or w % p != 0:
            new_h = (h // p) * p
            new_w = (w // p) * p
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            h, w = new_h, new_w
        
        if self.use_multilevel_adapter:
            layer_indices = list(self.intermediate_layer_indices)
            final_layer_idx = len(self.backbone.blocks) - 1
            patch_tokens_by_layer = self.backbone.get_intermediate_layers(
                x,
                n=layer_indices + [final_layer_idx],
                reshape=False,
                return_class_token=False,
                norm=True,
            )
            patch_tokens = patch_tokens_by_layer[-1]
        else:
            output = self.backbone.forward_features(x)
            patch_tokens = output['x_norm_patchtokens']  # [B*T, N_patches, D]
        
        # Reshape 为 2D 特征图: [B*T, H_grid*W_grid, D] -> [B*T, D, H_grid, W_grid]
        h_grid = h // p
        w_grid = w // p
        feature_map = patch_tokens.reshape(b, h_grid, w_grid, self.num_channels).permute(0, 3, 1, 2)
        
        if not self.use_multilevel_adapter:
            return feature_map

        intermediate_maps = tuple(
            tokens.reshape(b, h_grid, w_grid, self.num_channels).permute(0, 3, 1, 2).contiguous()
            for tokens in patch_tokens_by_layer[:-1]
        )
        return feature_map.contiguous(), intermediate_maps


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        bs, t, _, h, w = x.shape
        x = x.reshape(bs * t, 3, h, w)

        backbone_output = self[0](x)
        intermediate_features = None
        if isinstance(backbone_output, tuple):
            features, intermediate_features = backbone_output
        else:
            features = backbone_output
        _, c, oh, ow = features.shape

        pos = self[1](features).to(x.dtype)

        if intermediate_features is not None:
            return features, pos, intermediate_features
        return features, pos


def  build_backbone(args):
    pos_embed = build_position_encoding(args)
    
    # 根据 backbone 名称选择不同的 Backbone
    if 'dinov2' in args.backbone:
        backbone = DinoV2Backbone(args)
    else:
        backbone = Backbone(args)
    
    model = Joiner(backbone, pos_embed)
    model.num_channels = backbone.num_channels
    model.intermediate_layer_indices = getattr(backbone, 'intermediate_layer_indices', tuple())
    return model
