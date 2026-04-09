import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.num_object_boxes = int(getattr(args, 'num_object_boxes', 10))

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
        self.use_pairwise_refiner = bool(getattr(args, 'use_pairwise_refiner', True))
        self.pairwise_refine_scale = float(getattr(args, 'pairwise_refine_scale', 0.5))
        self.pairwise_use_object_relation = bool(getattr(args, 'pairwise_use_object_relation', True))
        self.pairwise_use_geom_relation = bool(getattr(args, 'pairwise_use_geom_relation', True))
        self.pairwise_geom_dim = 6
        self.pairwise_obj_dim = 2
        if self.use_pairwise_refiner:
            self.pairwise_affinity_mlp = MLP(
                2 * self.hidden_dim + self.pairwise_geom_dim + self.pairwise_obj_dim,
                self.hidden_dim,
                1,
                3,
            )

        # OLIC (Object-Conditioned Local Interaction Conditioner)
        self.use_olic = bool(getattr(args, 'use_olic', False))
        self.disable_group_olic = bool(getattr(args, 'disable_group_olic', True))
        self.olic_topk_obj = int(getattr(args, 'olic_topk_obj', 6))
        self.olic_dropout_p = float(getattr(args, 'olic_dropout', args.drop_rate))
        self.olic_use_ffn = bool(getattr(args, 'olic_use_ffn', False))
        self.olic_score_use = getattr(args, 'olic_score_use', 'prune_relevance')
        self.olic_res_scale_init = float(getattr(args, 'olic_res_scale_init', 0.0))
        self.olic_gate_init_bias = float(getattr(args, 'olic_gate_init_bias', -4.0))
        self.olic_attn_tau = float(getattr(args, 'olic_attn_tau', 2.0))
        if self.olic_attn_tau <= 0.0:
            self.olic_attn_tau = 1.0
        self.olic_geom_scale_max = float(getattr(args, 'olic_geom_scale_max', 2.0))
        if self.olic_geom_scale_max <= 0.0:
            self.olic_geom_scale_max = 2.0
        self.olic_geom_scale_init = float(getattr(args, 'olic_geom_scale_init', 1.0))

        if self.use_olic:
            # Object tokenization uses the same RoIAlign config as actor path.
            self.obj_fc_emb = nn.Linear(
                self.crop_size * self.crop_size * self.backbone.num_channels,
                self.hidden_dim,
            )
            self.obj_drop_emb = nn.Dropout(p=args.drop_rate)
            self.obj_box_pos_emb = MLP(4, self.hidden_dim, self.hidden_dim, 3)

            # Legacy relevance head kept for checkpoint compatibility.
            # Pruning is disabled in the current stable OLIC path.
            self.olic_relevance_mlp = MLP(2 * self.hidden_dim + 7, self.hidden_dim, 1, 3)

            # Actor-object routing
            self.olic_q_actor = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_k_obj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_v_obj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_geom_bias_ao = MLP(6, self.hidden_dim, 1, 2)
            # Learnable geometry contribution scale in [0, olic_geom_scale_max].
            init_ratio = self.olic_geom_scale_init / self.olic_geom_scale_max
            init_ratio = max(min(init_ratio, 1.0 - 1e-4), 1e-4)
            init_logit = math.log(init_ratio / (1.0 - init_ratio))
            self.olic_geom_scale_logit = nn.Parameter(
                torch.tensor(init_logit, dtype=torch.float32)
            )

            # Group-aware aggregation
            self.olic_group_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_actor_key = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Gates
            self.olic_gate_actor = MLP(2 * self.hidden_dim, self.hidden_dim, 1, 2)
            self.olic_gate_group = MLP(2 * self.hidden_dim, self.hidden_dim, 1, 2)
            self.olic_drop = nn.Dropout(p=self.olic_dropout_p)
            # Learnable residual scaling for stable cold start.
            self.olic_actor_res_scale = nn.Parameter(
                torch.tensor(self.olic_res_scale_init, dtype=torch.float32)
            )
            self.olic_group_res_scale = nn.Parameter(
                torch.tensor(self.olic_res_scale_init, dtype=torch.float32)
            )

            # Optional FFN block with zero-initialized residual scale.
            if self.olic_use_ffn:
                ffn_dim = 4 * self.hidden_dim
                self.olic_actor_ffn_norm = nn.LayerNorm(self.hidden_dim)
                self.olic_group_ffn_norm = nn.LayerNorm(self.hidden_dim)
                self.olic_actor_ffn = nn.Sequential(
                    nn.Linear(self.hidden_dim, ffn_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.olic_dropout_p),
                    nn.Linear(ffn_dim, self.hidden_dim),
                    nn.Dropout(self.olic_dropout_p),
                )
                self.olic_group_ffn = nn.Sequential(
                    nn.Linear(self.hidden_dim, ffn_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.olic_dropout_p),
                    nn.Linear(ffn_dim, self.hidden_dim),
                    nn.Dropout(self.olic_dropout_p),
                )
                ffn_init = float(getattr(args, 'olic_ffn_scale_init', 0.0))
                self.olic_actor_ffn_scale = nn.Parameter(torch.tensor(ffn_init, dtype=torch.float32))
                self.olic_group_ffn_scale = nn.Parameter(torch.tensor(ffn_init, dtype=torch.float32))

        self.mae_fusion = getattr(args, 'mae_fusion', 'adaptive_two_branch')
        self.use_mae = getattr(args, 'use_mae', False) and self.mae_fusion != 'none'
        if self.use_mae:
            mae_dim = getattr(args, 'mae_dim', 768)
            print(f"Initializing VideoMAE Adapter with dim={mae_dim}, fusion={self.mae_fusion}...")
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

        # Apply gate bias init after generic module init to avoid being overwritten.
        if self.use_olic:
            if self.olic_gate_actor.layers[-1].bias is not None:
                nn.init.constant_(self.olic_gate_actor.layers[-1].bias, self.olic_gate_init_bias)
            if self.olic_gate_group.layers[-1].bias is not None:
                nn.init.constant_(self.olic_gate_group.layers[-1].bias, self.olic_gate_init_bias)

    def calculate_pairwise_distnace(self, boxes):
        bs = boxes.shape[0]

        rx = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))
        ry = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))

        dist = rx - 2.0 * boxes.matmul(boxes.transpose(1, 2)) + ry.transpose(1, 2)

        return torch.sqrt(dist)

    @staticmethod
    def _cxcywh_to_xyxy(boxes):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack((x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h), dim=-1)

    @staticmethod
    def _xyxy_to_cxcywh(boxes):
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        return torch.stack((cx, cy, w, h), dim=-1)

    @staticmethod
    def _pair_geom_from_cxcywh(actor_cxcywh, obj_cxcywh, actor_xyxy, obj_xyxy):
        # actor: [BT, N, 4], obj: [BT, M, 4]
        ax, ay, aw, ah = actor_cxcywh.unbind(-1)
        ox, oy, ow, oh = obj_cxcywh.unbind(-1)

        dx = ax.unsqueeze(-1) - ox.unsqueeze(-2)
        dy = ay.unsqueeze(-1) - oy.unsqueeze(-2)
        lwr = torch.log((aw.unsqueeze(-1) / ow.unsqueeze(-2)).clamp(min=1e-6))
        lhr = torch.log((ah.unsqueeze(-1) / oh.unsqueeze(-2)).clamp(min=1e-6))
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)

        ax1, ay1, ax2, ay2 = actor_xyxy.unbind(-1)
        ox1, oy1, ox2, oy2 = obj_xyxy.unbind(-1)
        inter_x1 = torch.maximum(ax1.unsqueeze(-1), ox1.unsqueeze(-2))
        inter_y1 = torch.maximum(ay1.unsqueeze(-1), oy1.unsqueeze(-2))
        inter_x2 = torch.minimum(ax2.unsqueeze(-1), ox2.unsqueeze(-2))
        inter_y2 = torch.minimum(ay2.unsqueeze(-1), oy2.unsqueeze(-2))
        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter = inter_w * inter_h
        area_a = ((ax2 - ax1).clamp(min=0.0) * (ay2 - ay1).clamp(min=0.0)).unsqueeze(-1)
        area_o = ((ox2 - ox1).clamp(min=0.0) * (oy2 - oy1).clamp(min=0.0)).unsqueeze(-2)
        union = (area_a + area_o - inter).clamp(min=1e-6)
        iou = inter / union

        return torch.stack((dx, dy, lwr, lhr, dist, iou), dim=-1)

    @staticmethod
    def _masked_softmax(logits, mask, dim=-1):
        # mask: True means valid.
        logits = logits.masked_fill(~mask, -1e4)
        probs = torch.softmax(logits, dim=dim)
        probs = probs * mask.float()
        denom = probs.sum(dim=dim, keepdim=True).clamp(min=1e-6)
        return probs / denom

    @staticmethod
    def _clip_pair_geom(boxes_norm):
        # boxes_norm: [B, T, N, 4] normalized cxcywh
        cx = boxes_norm[..., 0]
        cy = boxes_norm[..., 1]
        w = boxes_norm[..., 2].clamp(min=1e-6)
        h = boxes_norm[..., 3].clamp(min=1e-6)

        dx = cx.unsqueeze(-1) - cx.unsqueeze(-2)
        dy = cy.unsqueeze(-1) - cy.unsqueeze(-2)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        mean_dist = dist.mean(dim=1)
        min_dist = dist.min(dim=1).values

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        inter_x1 = torch.maximum(x1.unsqueeze(-1), x1.unsqueeze(-2))
        inter_y1 = torch.maximum(y1.unsqueeze(-1), y1.unsqueeze(-2))
        inter_x2 = torch.minimum(x2.unsqueeze(-1), x2.unsqueeze(-2))
        inter_y2 = torch.minimum(y2.unsqueeze(-1), y2.unsqueeze(-2))
        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter = inter_w * inter_h
        area = ((x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0))
        union = (area.unsqueeze(-1) + area.unsqueeze(-2) - inter).clamp(min=1e-6)
        mean_iou = (inter / union).mean(dim=1)

        aspect = torch.log((w / h).clamp(min=1e-6))
        aspect_diff = (aspect.unsqueeze(-1) - aspect.unsqueeze(-2)).mean(dim=1)
        mean_dx = dx.mean(dim=1)
        mean_dy = dy.mean(dim=1)

        return torch.stack((mean_dist, min_dist, mean_iou, aspect_diff, mean_dx, mean_dy), dim=-1)

    @staticmethod
    def _pairwise_object_relation(actor_obj_clip):
        normed = F.normalize(actor_obj_clip, p=2, dim=-1)
        cos = torch.einsum('bih,bjh->bij', normed, normed)
        diff = actor_obj_clip.unsqueeze(2) - actor_obj_clip.unsqueeze(1)
        diff_norm = torch.norm(diff, dim=-1)
        return torch.stack((cos, diff_norm), dim=-1)

    def _run_pairwise_refiner(
        self,
        actor_clip,
        outputs_actor_emb,
        outputs_group_emb,
        boxes_norm,
        dummy_mask,
        actor_obj_clip=None,
    ):
        bs, n, _ = actor_clip.shape
        actor_valid = ~dummy_mask.bool()
        base_logits = torch.bmm(outputs_group_emb, outputs_actor_emb.transpose(1, 2))

        if not self.use_pairwise_refiner:
            zero_pair = actor_clip.new_zeros(bs, n, n)
            pair_valid = actor_valid.unsqueeze(1) & actor_valid.unsqueeze(2)
            membership = F.softmax(base_logits, dim=1)
            entropy = -(membership * membership.clamp(min=1e-9).log()).sum(dim=1)
            entropy = (entropy * actor_valid.float()).sum() / actor_valid.float().sum().clamp(min=1.0)
            return {
                "membership_logits_base": base_logits,
                "membership_logits_refined": base_logits,
                "membership": membership,
                "pairwise_affinity_logits": zero_pair,
                "pairwise_affinity_probs": zero_pair,
                "pairwise_valid_mask": pair_valid,
                "pairwise_refine_delta_mean": actor_clip.new_tensor(0.0),
                "membership_entropy": entropy,
            }

        pair_actor_i = actor_clip.unsqueeze(2).expand(bs, n, n, self.hidden_dim)
        pair_actor_j = actor_clip.unsqueeze(1).expand(bs, n, n, self.hidden_dim)

        if self.pairwise_use_geom_relation:
            pair_geom = self._clip_pair_geom(boxes_norm)
        else:
            pair_geom = actor_clip.new_zeros(bs, n, n, self.pairwise_geom_dim)

        if self.pairwise_use_object_relation and actor_obj_clip is not None:
            pair_obj = self._pairwise_object_relation(actor_obj_clip)
        else:
            pair_obj = actor_clip.new_zeros(bs, n, n, self.pairwise_obj_dim)

        pair_feat = torch.cat((pair_actor_i, pair_actor_j, pair_geom, pair_obj), dim=-1)
        pair_logits = self.pairwise_affinity_mlp(pair_feat).squeeze(-1)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))

        pair_valid = actor_valid.unsqueeze(1) & actor_valid.unsqueeze(2)
        eye = torch.eye(n, dtype=torch.bool, device=actor_clip.device).unsqueeze(0)
        pair_valid = pair_valid & (~eye)
        pair_logits = pair_logits.masked_fill(~pair_valid, 0.0)
        pair_probs = torch.sigmoid(pair_logits) * pair_valid.float()

        base_probs = F.softmax(base_logits, dim=1) * actor_valid.unsqueeze(1).float()
        support = torch.einsum('bgi,bij->bgj', base_probs, pair_probs)
        support = support * actor_valid.unsqueeze(1).float()
        refined_logits = base_logits + self.pairwise_refine_scale * support
        membership = F.softmax(refined_logits, dim=1)

        refine_mask = actor_valid.unsqueeze(1).expand_as(refined_logits).float()
        refine_delta = (refined_logits - base_logits).abs() * refine_mask
        refine_delta_mean = refine_delta.sum() / refine_mask.sum().clamp(min=1.0)
        membership_safe = membership.clamp(min=1e-9)
        membership_entropy = -(membership * membership_safe.log()).sum(dim=1)
        membership_entropy = (membership_entropy * actor_valid.float()).sum() / actor_valid.float().sum().clamp(min=1.0)

        return {
            "membership_logits_base": base_logits,
            "membership_logits_refined": refined_logits,
            "membership": membership,
            "pairwise_affinity_logits": pair_logits,
            "pairwise_affinity_probs": pair_probs,
            "pairwise_valid_mask": pair_valid,
            "pairwise_refine_delta_mean": refine_delta_mean,
            "membership_entropy": membership_entropy,
        }

    def _run_olic(
        self,
        features,
        actor_boxes_norm,
        actor_tokens_for_olic,
        group_tokens_for_olic,
        actor_valid_mask_bt_n,
        object_boxes_xyxy,
        object_valid_mask,
        object_scores,
        oh,
        ow,
    ):
        # shapes:
        # features: [BT, C, oh, ow]
        # actor_boxes_norm: [B, T, N, 4] cxcywh normalized
        # actor_tokens/group_tokens: [B, T, *, H]
        # object_boxes_xyxy: [B, T, M, 4] normalized xyxy
        # object_valid_mask: [B, T, M]
        # object_scores: [B, T, M]
        bs, t, n, _ = actor_boxes_norm.shape
        bt = bs * t
        m = object_boxes_xyxy.shape[2]

        actor_tokens_bt = actor_tokens_for_olic.reshape(bt, n, self.hidden_dim)
        group_tokens_bt = group_tokens_for_olic.reshape(bt, self.num_group_tokens, self.hidden_dim)
        actor_boxes_bt = actor_boxes_norm.reshape(bt, n, 4)
        actor_xyxy_bt = self._cxcywh_to_xyxy(actor_boxes_bt).clamp(0.0, 1.0)

        obj_xyxy = object_boxes_xyxy.reshape(bt, m, 4).clamp(0.0, 1.0)
        ox1, oy1, ox2, oy2 = obj_xyxy.unbind(-1)
        ox1, ox2 = torch.minimum(ox1, ox2), torch.maximum(ox1, ox2)
        oy1, oy2 = torch.minimum(oy1, oy2), torch.maximum(oy1, oy2)
        obj_xyxy = torch.stack((ox1, oy1, ox2, oy2), dim=-1)
        obj_cxcywh = self._xyxy_to_cxcywh(obj_xyxy)

        obj_valid = object_valid_mask.reshape(bt, m) > 0.5

        # Object RoIAlign on the same feature map scale/path as actor RoIAlign.
        obj_pixel = obj_xyxy.clone()
        obj_pixel[..., 0] = obj_xyxy[..., 0] * ow
        obj_pixel[..., 1] = obj_xyxy[..., 1] * oh
        obj_pixel[..., 2] = obj_xyxy[..., 2] * ow
        obj_pixel[..., 3] = obj_xyxy[..., 3] * oh
        obj_pixel[..., 2] = torch.maximum(obj_pixel[..., 2], obj_pixel[..., 0] + 1e-3)
        obj_pixel[..., 3] = torch.maximum(obj_pixel[..., 3], obj_pixel[..., 1] + 1e-3)

        boxes_list = [obj_pixel[i] for i in range(bt)]
        obj_features = self.roi_align(features, boxes_list)
        obj_features = obj_features.reshape(bt * m, -1)
        obj_features = self.obj_fc_emb(obj_features)
        obj_features = F.relu(obj_features)
        obj_features = self.obj_drop_emb(obj_features)
        obj_features = obj_features.reshape(bt, m, self.hidden_dim)
        obj_features = obj_features + self.obj_box_pos_emb(obj_cxcywh.reshape(bt, m, 4))

        actor_valid = actor_valid_mask_bt_n.bool()

        geom_ao = self._pair_geom_from_cxcywh(
            actor_boxes_bt,
            obj_cxcywh,
            actor_xyxy_bt,
            obj_xyxy,
        )  # [BT, N, M, 6]
        rel_valid_mask = actor_valid.unsqueeze(-1) & obj_valid.unsqueeze(1)

        q = self.olic_q_actor(actor_tokens_bt)                          # [BT, N, H]
        k = self.olic_k_obj(obj_features)                               # [BT, M, H]
        v = self.olic_v_obj(obj_features)                               # [BT, M, H]
        qk_logits = torch.einsum('bnh,bmh->bnm', q, k) / math.sqrt(self.hidden_dim)
        geom_bias = self.olic_geom_bias_ao(geom_ao).squeeze(-1)
        geom_scale = self.olic_geom_scale_max * torch.sigmoid(self.olic_geom_scale_logit)
        geom_bias_scaled = geom_scale * geom_bias
        attn_logits = (qk_logits + geom_bias_scaled) / self.olic_attn_tau

        # No hard pruning: route over all valid objects with soft attention.
        attn = self._masked_softmax(attn_logits, rel_valid_mask, dim=-1)
        c_actor = torch.einsum('bnm,bmh->bnh', attn, v)
        c_actor = c_actor * actor_valid.unsqueeze(-1).float()

        # Diagnostics for geometry-vs-content balance and attention collapse checks.
        valid_pairs_f = rel_valid_mask.float()
        valid_pairs_count = valid_pairs_f.sum().clamp(min=1.0)
        qk_mean = (qk_logits * valid_pairs_f).sum() / valid_pairs_count
        qk_var = ((qk_logits - qk_mean) ** 2 * valid_pairs_f).sum() / valid_pairs_count
        qk_std = torch.sqrt(qk_var + 1e-12)

        geom_mean = (geom_bias_scaled * valid_pairs_f).sum() / valid_pairs_count
        geom_var = ((geom_bias_scaled - geom_mean) ** 2 * valid_pairs_f).sum() / valid_pairs_count
        geom_std = torch.sqrt(geom_var + 1e-12)
        geom_qk_ratio = geom_std / (qk_std + 1e-6)

        actor_valid_f = actor_valid.float()
        valid_actor_count = actor_valid_f.sum().clamp(min=1.0)
        attn_safe = attn.clamp(min=1e-9)
        attn_entropy = -(attn * torch.log(attn_safe)).sum(dim=-1)  # [BT, N]
        attn_entropy_mean = (attn_entropy * actor_valid_f).sum() / valid_actor_count
        attn_top1 = attn.max(dim=-1).values  # [BT, N]
        attn_top1_mean = (attn_top1 * actor_valid_f).sum() / valid_actor_count
        valid_obj_per_actor = (valid_pairs_f.sum(dim=-1) * actor_valid_f).sum() / valid_actor_count

        # Group-aware aggregation from actor summaries.
        qg = self.olic_group_query(group_tokens_bt)                     # [BT, K, H]
        ka = self.olic_actor_key(actor_tokens_bt)                       # [BT, N, H]
        group_logits = torch.einsum('bkh,bnh->bkn', qg, ka) / math.sqrt(self.hidden_dim)
        group_attn = self._masked_softmax(
            group_logits,
            actor_valid.unsqueeze(1).expand(bt, self.num_group_tokens, n),
            dim=-1,
        )
        c_group = torch.einsum('bkn,bnh->bkh', group_attn, c_actor)

        alpha = torch.sigmoid(self.olic_gate_actor(torch.cat((actor_tokens_bt, c_actor), dim=-1)))  # [BT, N, 1]
        beta = torch.sigmoid(self.olic_gate_group(torch.cat((group_tokens_bt, c_group), dim=-1)))   # [BT, K, 1]
        alpha = alpha * actor_valid.unsqueeze(-1).float()

        c_actor = c_actor.reshape(bs, t, n, self.hidden_dim)
        c_group = c_group.reshape(bs, t, self.num_group_tokens, self.hidden_dim)
        alpha = alpha.reshape(bs, t, n, 1)
        beta = beta.reshape(bs, t, self.num_group_tokens, 1)
        diag = {
            "olic_qk_std": qk_std,
            "olic_geom_std": geom_std,
            "olic_geom_qk_ratio": geom_qk_ratio,
            "olic_attn_entropy": attn_entropy_mean,
            "olic_attn_top1_mean": attn_top1_mean,
            "olic_valid_obj_per_actor": valid_obj_per_actor,
            "olic_geom_scale": geom_scale,
        }
        return c_actor, c_group, alpha, beta, diag

    def forward(
        self,
        x,
        boxes,
        dummy_mask,
        mae_feats=None,
        object_boxes_xyxy=None,
        object_valid_mask=None,
        object_scores=None,
        olic_warmup_scale=1.0,
    ):
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

        # group transformer
        hs, actor_att, feature_att = self.group_transformer(src, actor_mask, group_dummy_mask,
                                                            self.group_query_emb.weight, pos, actor_features)
        # [1, bs * t, n + k, f'], [1, bs * t, k, n], [1, bs * t, n + k, oh x ow]   M: # group tokens, K: # boxes

        actor_hs = hs[0, :, :n]
        group_hs = hs[0, :, n:]

        actor_hs = actor_hs.reshape(bs, t, n, -1)
        actor_hs = actor_features + actor_hs
        group_hs = group_hs.reshape(bs, t, self.num_group_tokens, -1)

        # Keep decoder tokens for OLIC routing (independent of scene branch).
        actor_hs_decoder = actor_hs
        group_hs_decoder = group_hs

        if mae_feats is not None and self.videomae_adapter is not None:
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

        # OLIC parallel fusion after scene branch, before temporal modeling.
        if (
            self.use_olic
            and object_boxes_xyxy is not None
            and object_valid_mask is not None
            and object_scores is not None
        ):
            olic_scale = float(max(0.0, min(1.0, olic_warmup_scale)))
            actor_valid_bt = (~dummy_mask).reshape(bs * t, n)
            c_actor, c_group, alpha_o, beta_o, olic_diag = self._run_olic(
                features=features,
                actor_boxes_norm=boxes_norm,
                actor_tokens_for_olic=actor_hs_decoder,
                group_tokens_for_olic=group_hs_decoder,
                actor_valid_mask_bt_n=actor_valid_bt,
                object_boxes_xyxy=object_boxes_xyxy,
                object_valid_mask=object_valid_mask,
                object_scores=object_scores,
                oh=oh,
                ow=ow,
            )

            actor_hs = actor_hs + olic_scale * self.olic_actor_res_scale * self.olic_drop(alpha_o * c_actor)
            if not self.disable_group_olic:
                group_hs = group_hs + olic_scale * self.olic_group_res_scale * self.olic_drop(beta_o * c_group)
            else:
                beta_o = torch.zeros_like(beta_o)

            if self.olic_use_ffn:
                actor_hs = actor_hs + self.olic_actor_ffn_scale * self.olic_actor_ffn(
                    self.olic_actor_ffn_norm(actor_hs)
                )
                if not self.disable_group_olic:
                    group_hs = group_hs + self.olic_group_ffn_scale * self.olic_group_ffn(
                        self.olic_group_ffn_norm(group_hs)
                    )
            olic_alpha_mean = alpha_o.mean()
            olic_beta_mean = beta_o.mean()
        else:
            olic_alpha_mean = actor_hs.new_tensor(0.0)
            olic_beta_mean = actor_hs.new_tensor(0.0)
            olic_scale = float(max(0.0, min(1.0, olic_warmup_scale)))
            olic_diag = {
                "olic_qk_std": actor_hs.new_tensor(0.0),
                "olic_geom_std": actor_hs.new_tensor(0.0),
                "olic_geom_qk_ratio": actor_hs.new_tensor(0.0),
                "olic_attn_entropy": actor_hs.new_tensor(0.0),
                "olic_attn_top1_mean": actor_hs.new_tensor(0.0),
                "olic_valid_obj_per_actor": actor_hs.new_tensor(0.0),
                "olic_geom_scale": actor_hs.new_tensor(0.0),
            }

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
        actor_obj_clip = None
        if self.use_olic and object_boxes_xyxy is not None and object_valid_mask is not None and object_scores is not None:
            actor_obj_clip = c_actor.mean(dim=1)
        pairwise_out = self._run_pairwise_refiner(
            actor_clip=inst_repr,
            outputs_actor_emb=outputs_actor_emb,
            outputs_group_emb=outputs_group_emb,
            boxes_norm=boxes_norm,
            dummy_mask=dummy_mask.reshape(bs, t, n)[:, 0, :],
            actor_obj_clip=actor_obj_clip,
        )
        membership = pairwise_out["membership"]

        out = {
            "pred_actions": outputs_class,
            "pred_activities": outputs_group_class,
            "membership": membership.reshape(bs, self.num_group_tokens, self.num_boxes),
            "actor_embeddings": inst_repr,
            "membership_logits_base": pairwise_out["membership_logits_base"],
            "membership_logits_refined": pairwise_out["membership_logits_refined"],
            "pairwise_affinity_logits": pairwise_out["pairwise_affinity_logits"],
            "pairwise_affinity_probs": pairwise_out["pairwise_affinity_probs"],
            "pairwise_valid_mask": pairwise_out["pairwise_valid_mask"],
            "pairwise_refine_delta_mean": pairwise_out["pairwise_refine_delta_mean"].detach(),
            "membership_entropy": pairwise_out["membership_entropy"].detach(),
            "group_olic_disabled": inst_repr.new_tensor(1.0 if self.disable_group_olic else 0.0),
            "olic_alpha_mean": olic_alpha_mean.detach(),
            "olic_beta_mean": olic_beta_mean.detach(),
            "olic_warmup_scale": inst_repr.new_tensor(olic_scale),
            "olic_qk_std": olic_diag["olic_qk_std"].detach(),
            "olic_geom_std": olic_diag["olic_geom_std"].detach(),
            "olic_geom_qk_ratio": olic_diag["olic_geom_qk_ratio"].detach(),
            "olic_attn_entropy": olic_diag["olic_attn_entropy"].detach(),
            "olic_attn_top1_mean": olic_diag["olic_attn_top1_mean"].detach(),
            "olic_valid_obj_per_actor": olic_diag["olic_valid_obj_per_actor"].detach(),
            "olic_geom_scale": olic_diag["olic_geom_scale"].detach(),
        }

        return out
