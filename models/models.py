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

        # OLIC (Object-Conditioned Local Interaction Conditioner)
        self.use_olic = bool(getattr(args, 'use_olic', False))
        self.olic_topk_obj = int(getattr(args, 'olic_topk_obj', 6))
        self.olic_dropout_p = float(getattr(args, 'olic_dropout', args.drop_rate))
        self.olic_use_ffn = bool(getattr(args, 'olic_use_ffn', False))
        self.olic_score_use = getattr(args, 'olic_score_use', 'prune_relevance')

        if self.use_olic:
            # Object tokenization uses the same RoIAlign config as actor path.
            self.obj_fc_emb = nn.Linear(
                self.crop_size * self.crop_size * self.backbone.num_channels,
                self.hidden_dim,
            )
            self.obj_drop_emb = nn.Dropout(p=args.drop_rate)
            self.obj_box_pos_emb = MLP(4, self.hidden_dim, self.hidden_dim, 3)

            # Relevance pruning: [actor, object, geom6, score1] -> scalar
            self.olic_relevance_mlp = MLP(2 * self.hidden_dim + 7, self.hidden_dim, 1, 3)

            # Actor-object routing
            self.olic_q_actor = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_k_obj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_v_obj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_geom_bias_ao = MLP(6, self.hidden_dim, 1, 2)

            # Group-aware aggregation
            self.olic_group_query = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.olic_actor_key = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Gates
            self.olic_gate_actor = MLP(2 * self.hidden_dim, self.hidden_dim, 1, 2)
            self.olic_gate_group = MLP(2 * self.hidden_dim, self.hidden_dim, 1, 2)
            self.olic_drop = nn.Dropout(p=self.olic_dropout_p)

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
        obj_scores_bt = object_scores.reshape(bt, m).clamp(min=0.0)

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
        score_feat = torch.log(obj_scores_bt + 1e-6).unsqueeze(1).unsqueeze(-1).expand(bt, n, m, 1)

        actor_expand = actor_tokens_bt.unsqueeze(2).expand(bt, n, m, self.hidden_dim)
        obj_expand = obj_features.unsqueeze(1).expand(bt, n, m, self.hidden_dim)
        rel_in = torch.cat((actor_expand, obj_expand, geom_ao, score_feat), dim=-1)
        relevance = self.olic_relevance_mlp(rel_in).squeeze(-1)  # [BT, N, M]

        rel_valid_mask = actor_valid.unsqueeze(-1) & obj_valid.unsqueeze(1)
        relevance = relevance.masked_fill(~rel_valid_mask, -1e4)
        topk = max(1, min(self.olic_topk_obj, m))
        topk_idx = relevance.topk(k=topk, dim=-1).indices
        selected_mask = torch.zeros_like(rel_valid_mask, dtype=torch.bool)
        selected_mask.scatter_(dim=-1, index=topk_idx, value=True)
        selected_mask = selected_mask & rel_valid_mask

        q = self.olic_q_actor(actor_tokens_bt)                          # [BT, N, H]
        k = self.olic_k_obj(obj_features)                               # [BT, M, H]
        v = self.olic_v_obj(obj_features)                               # [BT, M, H]
        attn_logits = torch.einsum('bnh,bmh->bnm', q, k) / math.sqrt(self.hidden_dim)
        geom_bias = self.olic_geom_bias_ao(geom_ao).squeeze(-1)
        attn_logits = attn_logits + geom_bias

        attn = self._masked_softmax(attn_logits, selected_mask, dim=-1)
        c_actor = torch.einsum('bnm,bmh->bnh', attn, v)
        c_actor = c_actor * actor_valid.unsqueeze(-1).float()

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
        return c_actor, c_group, alpha, beta

    def forward(
        self,
        x,
        boxes,
        dummy_mask,
        mae_feats=None,
        object_boxes_xyxy=None,
        object_valid_mask=None,
        object_scores=None,
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
            actor_valid_bt = (~dummy_mask).reshape(bs * t, n)
            c_actor, c_group, alpha_o, beta_o = self._run_olic(
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

            actor_hs = actor_hs + self.olic_drop(alpha_o * c_actor)
            group_hs = group_hs + self.olic_drop(beta_o * c_group)

            if self.olic_use_ffn:
                actor_hs = actor_hs + self.olic_actor_ffn_scale * self.olic_actor_ffn(
                    self.olic_actor_ffn_norm(actor_hs)
                )
                group_hs = group_hs + self.olic_group_ffn_scale * self.olic_group_ffn(
                    self.olic_group_ffn_norm(group_hs)
                )

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
