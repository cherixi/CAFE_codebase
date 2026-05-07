import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign

from .backbone import build_backbone
from .group_transformer import build_group_transformer
from .feed_forward import MLP
from .hoi_graph import FrameHOIGraph, TemporalEncoder
from .videomae_adapter import VideoMAEAdapter


class GADTR(nn.Module):
    def __init__(self, args):
        super(GADTR, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_class
        self.num_frame = args.num_frame
        self.num_boxes = args.num_boxes

        self.hidden_dim = args.hidden_dim
        self.backbone = build_backbone(args)

        # RoI Align
        self.crop_size = args.crop_size
        self.roi_align = RoIAlign(output_size=(self.crop_size, self.crop_size), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        self.fc_emb = nn.Linear(self.crop_size*self.crop_size*self.backbone.num_channels, self.hidden_dim)
        self.drop_emb = nn.Dropout(p=args.drop_rate)

        # Actor embedding
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.box_pos_emb = MLP(4, self.hidden_dim, self.hidden_dim, 3)

        # Individual action classification head
        self.class_emb = nn.Linear(self.hidden_dim, self.num_class + 1)

        # Group Transformer (shared group queries across frames)
        self.group_transformer = build_group_transformer(args)
        self.num_group_tokens = args.num_group_tokens
        self.group_query_emb = nn.Embedding(self.num_group_tokens, self.hidden_dim)
        
        # Group activity classfication head
        self.group_emb = nn.Linear(self.hidden_dim, self.num_class + 1)

        # HOI mapping + temporal modeling
        self.hoi_mode = getattr(args, 'hoi_mode', 'penalty')
        hoi_nheads = getattr(args, 'hoi_nheads', 4)
        hoi_topk = getattr(args, 'hoi_topk', 0)
        hoi_penalty_type = getattr(args, 'hoi_penalty_type', 'quadratic')
        hoi_hard_thresh = getattr(args, 'hoi_hard_thresh', None)
        if hoi_hard_thresh is None:
            hoi_hard_thresh = getattr(args, 'distance_threshold', None)

        if self.hoi_mode != 'none':
            use_geom_bias = self.hoi_mode in ['bias', 'penalty']
            use_logit_penalty = self.hoi_mode == 'penalty'
            hard_mask_thresh = hoi_hard_thresh if self.hoi_mode == 'hard_mask' else None
            self.frame_graph = FrameHOIGraph(
                self.hidden_dim,
                nhead=hoi_nheads,
                dropout=args.drop_rate,
                topk=hoi_topk,
                use_geom_bias=use_geom_bias,
                use_logit_penalty=use_logit_penalty,
                hard_mask_thresh=hard_mask_thresh,
                penalty_type=hoi_penalty_type,
            )
        else:
            self.frame_graph = None
        self.temporal_encoder = TemporalEncoder(self.hidden_dim, nhead=args.gar_nheads, 
                                                num_layers=args.temporal_layers,
                                                tcn_kernel_size=args.tcn_kernel_size,
                                                tcn_dropout=args.tcn_dropout,
                                                dropout=args.drop_rate)
        self.time_pos_emb = nn.Embedding(self.num_frame, self.hidden_dim)
        self.actor_time_pool = nn.Linear(self.hidden_dim, 1)
        self.group_time_pool = nn.Linear(self.hidden_dim, 1)
        self.temporal_agg_mode = getattr(args, 'temporal_agg_mode', 'learned_pool')
        
        # Distance mask threshold (kept for backward compatibility)
        self.distance_threshold = getattr(args, 'distance_threshold', None)

        # Membership prediction heads
        self.actor_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.group_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mae_fusion = getattr(args, 'mae_fusion', 'adaptive_two_branch')
        self.mae_fusion_stage = getattr(args, 'mae_fusion_stage', 'post_group')
        self.use_mae = getattr(args, 'use_mae', False) and self.mae_fusion != 'none'
        if self.use_mae:
            mae_dim = getattr(args, 'mae_dim', 768)
            print(f"Initializing VideoMAE Adapter with dim={mae_dim}, fusion={self.mae_fusion}, stage={self.mae_fusion_stage}...")
            self.videomae_adapter = VideoMAEAdapter(
                global_dim=mae_dim,
                hidden_dim=self.hidden_dim,
                fusion=self.mae_fusion,
            )
        else:
            self.videomae_adapter = None

        self.relu = F.relu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'group_transformer' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def calculate_pairwise_distnace(self, boxes):
        bs = boxes.shape[0]

        rx = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))
        ry = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))

        dist = rx - 2.0 * boxes.matmul(boxes.transpose(1, 2)) + ry.transpose(1, 2)

        return torch.sqrt(dist)

    def forward(self, x, boxes, dummy_mask, mae_feats=None):
        """
        :param x: [B, T, 3, H, W]
        :param boxes: [B, T, N, 4]
        :param dummy_mask: [B, N]
        :return:
        """
        bs, t, _, h, w = x.shape
        n = boxes.shape[2]

        # keep normalized boxes for geometry; flatten copy for ROI Align
        boxes_norm = boxes.reshape(bs, t, n, 4)
        boxes_flat = boxes_norm.reshape(-1, 4)                                          # [b x t x n, 4]
        boxes_idx = [i * torch.ones(n, dtype=torch.int) for i in range(bs * t)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes.device)
        boxes_idx_flat = torch.reshape(boxes_idx, (bs * t * n, ))                       # [b x t x n]

        features, pos = self.backbone(x)
        _, c, oh, ow = features.shape                                                   # [b x t, d, oh, ow]

        src = self.input_proj(features)
        src = torch.reshape(src, (bs, t, -1, oh, ow))                                   # [b, t, c, oh, ow]

        # ignore dummy boxes (padded boxes to match the number of actors)
        dummy_mask = dummy_mask.unsqueeze(1).repeat(1, t, 1).reshape(-1, n).bool()
        valid_pairs = (~dummy_mask).unsqueeze(2) & (~dummy_mask).unsqueeze(1)
        actor_mask = ~valid_pairs  # True where either query/key is dummy

        # Unmask diagonal to prevent NaNs in attention (especially for dummy actors)
        diag_idx = torch.arange(n, device=actor_mask.device)
        actor_mask[:, diag_idx, diag_idx] = False

        group_dummy_mask = dummy_mask.clone()
        # Ensure at least one key is not masked for cross-attention
        all_masked = group_dummy_mask.all(dim=-1)
        if all_masked.any():
            group_dummy_mask[all_masked, 0] = False

        boxes_flat_pixel = boxes_flat.clone()
        boxes_flat_pixel[:, 0] = (boxes_flat[:, 0] - boxes_flat[:, 2] / 2) * ow
        boxes_flat_pixel[:, 1] = (boxes_flat[:, 1] - boxes_flat[:, 3] / 2) * oh
        boxes_flat_pixel[:, 2] = (boxes_flat[:, 0] + boxes_flat[:, 2] / 2) * ow
        boxes_flat_pixel[:, 3] = (boxes_flat[:, 1] + boxes_flat[:, 3] / 2) * oh

        boxes_flat_pixel.requires_grad = False
        boxes_idx_flat.requires_grad = False

        # extract actor features
        # torchvision RoIAlign expects List[Tensor[N, 4]], so we split by batch
        boxes_list = [boxes_flat_pixel[boxes_idx_flat == i] for i in range(bs * t)]
        actor_features = self.roi_align(features, boxes_list)
        actor_features = torch.reshape(actor_features, (bs * t * n, -1))
        actor_features = self.fc_emb(actor_features)
        actor_features = F.relu(actor_features)
        actor_features = self.drop_emb(actor_features)
        actor_features = actor_features.reshape(bs, t, n, self.hidden_dim)

        # add positional information to box features
        box_pos_emb = self.box_pos_emb(boxes)
        box_pos_emb = torch.reshape(box_pos_emb, (bs, t, n, -1))                        # [b, t, n, c]
        actor_features = actor_features + box_pos_emb

        # frame-level HOI mapping on actor tokens
        if self.frame_graph is not None:
            actor_graph_in = actor_features.reshape(bs * t, n, self.hidden_dim)
            boxes_for_graph = boxes_norm.reshape(bs * t, n, 4)
            actor_graph_out, _ = self.frame_graph(actor_graph_in, boxes_for_graph, attn_mask=actor_mask)
            actor_features = actor_graph_out.reshape(bs, t, n, self.hidden_dim)

        group_query_embed = self.group_query_emb.weight
        if mae_feats is not None and self.videomae_adapter is not None and self.mae_fusion_stage == 'pre_group':
            if mae_feats.dim() == 3:
                mae_feats = mae_feats.squeeze(1)

            mae_feats_expanded = mae_feats.unsqueeze(1).repeat(1, t, 1).reshape(bs * t, -1)
            actor_features_flat = actor_features.reshape(bs * t, n, -1)
            group_query_flat = self.group_query_emb.weight.unsqueeze(0).expand(bs * t, -1, -1)

            actor_features_flat, group_query_flat = self.videomae_adapter(
                actor_features_flat, group_query_flat, mae_feats_expanded
            )

            actor_features = actor_features_flat.reshape(bs, t, n, -1)
            group_query_embed = group_query_flat.reshape(bs, t, self.num_group_tokens, -1)

        # group transformer
        hs, actor_att, feature_att = self.group_transformer(src, actor_mask, group_dummy_mask,
                                                            group_query_embed, pos, actor_features)
        # [1, bs * t, n + k, f'], [1, bs * t, k, n], [1, bs * t, n + k, oh x ow]   M: # group tokens, K: # boxes

        actor_hs = hs[0, :, :n]
        group_hs = hs[0, :, n:]

        actor_hs = actor_hs.reshape(bs, t, n, -1)
        actor_hs = actor_features + actor_hs
        group_hs = group_hs.reshape(bs, t, self.num_group_tokens, -1)

        if mae_feats is not None and self.videomae_adapter is not None and self.mae_fusion_stage == 'post_group':
            # if not getattr(self, 'has_printed_videomae_status', False):
            #     print("VideoMAE features detected in forward pass. Applying enhancement...")
            #     self.has_printed_videomae_status = True

            if mae_feats.dim() == 3:
                mae_feats = mae_feats.squeeze(1)

            mae_feats_expanded = mae_feats.unsqueeze(1).repeat(1, t, 1).reshape(bs * t, -1)
            actor_hs_flat = actor_hs.reshape(bs * t, n, -1)
            group_hs_flat = group_hs.reshape(bs * t, self.num_group_tokens, -1)
            
            actor_hs_flat, group_hs_flat = self.videomae_adapter(actor_hs_flat, group_hs_flat, mae_feats_expanded)
            
            actor_hs = actor_hs_flat.reshape(bs, t, n, -1)
            group_hs = group_hs_flat.reshape(bs, t, self.num_group_tokens, -1)

        if self.temporal_agg_mode == 'frame_mean_main':
            # Main-branch style ablation:
            # 1) actor/group clip tokens are simple frame means
            # 2) clip logits are means of per-frame logits
            actor_clip = actor_hs.mean(dim=1)
            group_clip = group_hs.mean(dim=1)
            outputs_class = self.class_emb(actor_hs).mean(dim=1)
            outputs_group_class = self.group_emb(group_hs).mean(dim=1)
        else:
            # temporal modeling for actors
            temporal_actor_in = actor_hs.permute(0, 2, 1, 3).reshape(bs * n, t, self.hidden_dim)  # [b*n, t, c]
            time_pos = self.time_pos_emb.weight[:t].unsqueeze(0)                                         # [1, t, c]
            temporal_actor_out, _ = self.temporal_encoder(temporal_actor_in, pos=time_pos)

            actor_time_logits = self.actor_time_pool(temporal_actor_out).squeeze(-1)                     # [b*n, t]
            actor_time_weight = torch.softmax(actor_time_logits, dim=1).unsqueeze(-1)                    # [b*n, t, 1]
            actor_clip = (temporal_actor_out * actor_time_weight).sum(dim=1).reshape(bs, n, self.hidden_dim)

            # temporal modeling for group tokens
            temporal_group_in = group_hs.permute(0, 2, 1, 3).reshape(bs * self.num_group_tokens, t, self.hidden_dim)
            temporal_group_out, _ = self.temporal_encoder(temporal_group_in, pos=time_pos)
            group_time_logits = self.group_time_pool(temporal_group_out).squeeze(-1)                      # [b*k, t]
            group_time_weight = torch.softmax(group_time_logits, dim=1).unsqueeze(-1)                     # [b*k, t, 1]
            group_clip = (temporal_group_out * group_time_weight).sum(dim=1).reshape(bs, self.num_group_tokens, self.hidden_dim)

            # prediction heads (clip-level)
            outputs_class = self.class_emb(actor_clip)               # [b, n, num_class+1]
            outputs_group_class = self.group_emb(group_clip)         # [b, k, num_class+1]

        # normalize
        inst_repr = F.normalize(actor_clip, p=2, dim=2)
        group_repr = F.normalize(group_clip, p=2, dim=2)

        outputs_actor_emb = self.actor_match_emb(inst_repr)
        outputs_group_emb = self.group_match_emb(group_repr)

        membership = torch.bmm(outputs_group_emb, outputs_actor_emb.transpose(1, 2))
        membership = F.softmax(membership, dim=1)

        out = {
            "pred_actions": outputs_class,
            "pred_activities": outputs_group_class,
            "membership": membership.reshape(bs, self.num_group_tokens, self.num_boxes),
            "actor_embeddings": inst_repr,
        }

        return out
