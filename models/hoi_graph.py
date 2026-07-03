import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameHOIGraph(nn.Module):
    """
    Frame-level actor-to-actor graph modeling with multi-head attention.
    Supports additive geometry bias, hard distance mask, and logit penalty.
    """

    def __init__(self, d_model=256, nhead=4, dropout=0.1, topk=0,
                 use_geom_bias=True, use_logit_penalty=True, hard_mask_thresh=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.use_geom_bias = use_geom_bias
        self.use_logit_penalty = use_logit_penalty
        self.hard_mask_thresh = hard_mask_thresh

        # Geometry encoder -> per-head bias
        if self.use_geom_bias:
            self.geom_mlp = nn.Sequential(
                nn.Linear(5, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, nhead)
            )
        else:
            self.geom_mlp = None

        # Learnable distance temperature for soft penalty
        if self.use_logit_penalty:
            self.log_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma", torch.zeros(1))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.topk = topk  # keep top-k neighbors if >0

    def _pairwise_geometry(self, boxes):
        """
        boxes: [B*T, N, 4] with (cx, cy, w, h) in normalized coords
        return: geom_feat [B*T, N, N, 5], dist [B*T, N, N]
        """
        cx, cy, w, h = boxes.unbind(-1)
        dx = cx.unsqueeze(2) - cx.unsqueeze(1)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)
        log_w = torch.log(torch.clamp(w.unsqueeze(2) / (w.unsqueeze(1) + 1e-6), min=1e-6))
        log_h = torch.log(torch.clamp(h.unsqueeze(2) / (h.unsqueeze(1) + 1e-6), min=1e-6))
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        geom = torch.stack([dx, dy, log_w, log_h, dist], dim=-1)
        return geom, dist

    def forward(self, x, boxes, attn_mask=None):
        """
        x: [B*T, N, D]
        boxes: [B*T, N, 4] (cx, cy, w, h) normalized
        attn_mask: [B*T, N, N] bool, True indicates positions to mask
        """
        b_tokens, n, d = x.shape
        assert d == self.d_model, "Feature dim mismatch"

        # project to heads
        q = self.q_proj(x).view(b_tokens, n, self.nhead, self.dk)
        k = self.k_proj(x).view(b_tokens, n, self.nhead, self.dk)
        v = self.v_proj(x).view(b_tokens, n, self.nhead, self.dk)

        # [B*T, n, n, nhead] geom bias
        geom_feat, dist = self._pairwise_geometry(boxes)
        if self.use_geom_bias:
            geom_bias = self.geom_mlp(geom_feat)  # [B*T, n, n, nhead]
        else:
            geom_bias = None

        # attention scores
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.dk)  # [B*T, h, n, n]
        if geom_bias is not None:
            scores = scores + geom_bias.permute(0, 3, 1, 2)  # align head dim

        # soft distance penalty
        if self.use_logit_penalty:
            sigma = torch.exp(self.log_sigma) + 1e-6
            scores = scores - (dist.unsqueeze(1) ** 2) / (sigma ** 2)

        if self.hard_mask_thresh is not None:
            hard_mask = dist > self.hard_mask_thresh
            diag_idx = torch.arange(n, device=dist.device)
            hard_mask[:, diag_idx, diag_idx] = False
            scores = scores.masked_fill(hard_mask.unsqueeze(1), float("-inf"))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        if self.topk and self.topk > 0 and self.topk < n:
            # keep top-k per head/query, mask the rest
            topk_values, topk_idx = torch.topk(scores, self.topk, dim=-1)
            new_mask = torch.ones_like(scores, dtype=torch.bool)
            new_mask.scatter_(-1, topk_idx, False)
            scores = scores.masked_fill(new_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        
        # Handle NaN if all keys are masked (e.g. for dummy actors)
        if torch.isnan(attn).any():
            attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)  # [B*T, n, h, dk]
        out = out.reshape(b_tokens, n, d)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out))

        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))

        return x, attn


class FrameInteractionGraph(nn.Module):
    """
    Interaction-anchored frame graph.

    This keeps the original actor-actor STIR score and adds an optional
    anchor-induced edge bias from table/service objects. The first version is
    anchor-only: small objects are intentionally not used for pair edges.
    """

    def __init__(
        self,
        d_model=256,
        nhead=4,
        dropout=0.1,
        topk=0,
        use_geom_bias=True,
        use_logit_penalty=True,
        hard_mask_thresh=None,
        use_anchors=True,
        anchor_scale_max=0.5,
        anchor_scale_init=-6.0,
        anchor_bias_clip=2.0,
        anchor_attn_tau=3.0,
        anchor_source="gdino",
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.use_geom_bias = use_geom_bias
        self.use_logit_penalty = use_logit_penalty
        self.hard_mask_thresh = hard_mask_thresh
        self.topk = topk

        if self.use_geom_bias:
            self.geom_mlp = nn.Sequential(
                nn.Linear(5, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, nhead),
            )
        else:
            self.geom_mlp = None

        if self.use_logit_penalty:
            self.log_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma", torch.zeros(1))

        self.use_anchors = bool(use_anchors)
        self.anchor_source = str(anchor_source).lower()
        self.anchor_scale_max = float(anchor_scale_max)
        if self.anchor_scale_max <= 0.0:
            self.anchor_scale_max = 0.5
        self.anchor_bias_clip = float(anchor_bias_clip)
        if self.anchor_bias_clip <= 0.0:
            self.anchor_bias_clip = 2.0
        self.anchor_attn_tau = float(anchor_attn_tau)
        if self.anchor_attn_tau <= 0.0:
            self.anchor_attn_tau = 3.0

        # Actor-anchor association. Object tokens are only used to compute
        # smooth anchor association, not as direct actor residuals.
        self.anchor_q_actor = nn.Linear(d_model, d_model)
        self.anchor_k_obj = nn.Linear(d_model, d_model)
        self.anchor_geom_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        # Anchor evidence modulates actor-actor edge logits per head.
        self.anchor_bias_mlp = nn.Sequential(
            nn.Linear(7, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, nhead),
        )
        self.anchor_scale_logit = nn.Parameter(
            torch.full((nhead,), float(anchor_scale_init), dtype=torch.float32)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def _pairwise_geometry(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        dx = cx.unsqueeze(2) - cx.unsqueeze(1)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)
        log_w = torch.log(torch.clamp(w.unsqueeze(2) / (w.unsqueeze(1) + 1e-6), min=1e-6))
        log_h = torch.log(torch.clamp(h.unsqueeze(2) / (h.unsqueeze(1) + 1e-6), min=1e-6))
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        geom = torch.stack([dx, dy, log_w, log_h, dist], dim=-1)
        return geom, dist

    @staticmethod
    def _xyxy_to_cxcywh(boxes):
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        return torch.stack((cx, cy, w, h), dim=-1)

    @staticmethod
    def _actor_object_geometry(actor_cxcywh, obj_xyxy):
        obj_cxcywh = FrameInteractionGraph._xyxy_to_cxcywh(obj_xyxy)
        ax, ay, aw, ah = actor_cxcywh.unbind(-1)
        ox, oy, ow, oh = obj_cxcywh.unbind(-1)

        dx = ax.unsqueeze(-1) - ox.unsqueeze(-2)
        dy = ay.unsqueeze(-1) - oy.unsqueeze(-2)
        log_w = torch.log((aw.unsqueeze(-1) / ow.unsqueeze(-2)).clamp(min=1e-6))
        log_h = torch.log((ah.unsqueeze(-1) / oh.unsqueeze(-2)).clamp(min=1e-6))
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)

        ax1 = ax - 0.5 * aw
        ay1 = ay - 0.5 * ah
        ax2 = ax + 0.5 * aw
        ay2 = ay + 0.5 * ah
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
        return torch.stack((dx, dy, log_w, log_h, dist, iou), dim=-1)

    @staticmethod
    def _masked_softmax(logits, mask, dim=-1):
        logits = logits.masked_fill(~mask, -1e4)
        probs = torch.softmax(logits, dim=dim)
        probs = probs * mask.float()
        denom = probs.sum(dim=dim, keepdim=True).clamp(min=1e-6)
        return probs / denom

    @staticmethod
    def _masked_mean(values, mask):
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        return (values * mask_f).sum() / denom

    def _zero_diag(self, mask):
        n = mask.shape[-1]
        eye = torch.eye(n, device=mask.device, dtype=torch.bool).unsqueeze(0)
        return mask & (~eye)

    def _anchor_edge_bias(
        self,
        x,
        boxes,
        geom_feat,
        actor_valid_mask,
        object_tokens,
        object_boxes_xyxy,
        object_scores,
        object_family_id,
        object_valid_mask,
    ):
        b_tokens, n, _ = x.shape
        zero_pair = x.new_zeros(b_tokens, n, n)
        zero_diag = {
            "interaction_anchor_bias_mean": x.new_tensor(0.0),
            "interaction_anchor_bias_abs_mean": x.new_tensor(0.0),
            "interaction_anchor_bias_max": x.new_tensor(0.0),
            "interaction_anchor_bias_min": x.new_tensor(0.0),
            "interaction_anchor_bias_pos_ratio": x.new_tensor(0.0),
            "interaction_anchor_bias_neg_ratio": x.new_tensor(0.0),
            "interaction_anchor_shared_table_mean": x.new_tensor(0.0),
            "interaction_anchor_shared_service_mean": x.new_tensor(0.0),
            "interaction_anchor_scale_mean": (
                self.anchor_scale_max * torch.sigmoid(self.anchor_scale_logit)
            ).mean(),
            "interaction_anchor_top1_mean": x.new_tensor(0.0),
            "interaction_anchor_valid_per_actor": x.new_tensor(0.0),
        }

        if (
            (not self.use_anchors)
            or object_tokens is None
            or object_boxes_xyxy is None
            or object_scores is None
            or object_family_id is None
            or object_valid_mask is None
        ):
            return zero_pair.unsqueeze(1).expand(b_tokens, self.nhead, n, n), zero_pair, zero_pair, zero_diag

        obj_valid = object_valid_mask.bool()
        family = object_family_id.long()
        table_mask = obj_valid & (family == 5)
        if self.anchor_source == "yolo":
            service_mask = torch.zeros_like(obj_valid)
        else:
            service_mask = obj_valid & (family == 4)
        anchor_mask = table_mask | service_mask

        if not anchor_mask.any():
            return zero_pair.unsqueeze(1).expand(b_tokens, self.nhead, n, n), zero_pair, zero_pair, zero_diag

        q = self.anchor_q_actor(x)
        k = self.anchor_k_obj(object_tokens)
        qk_logits = torch.einsum("bnh,bmh->bnm", q, k) / math.sqrt(self.d_model)
        geom_ao = self._actor_object_geometry(boxes, object_boxes_xyxy)
        geom_logits = self.anchor_geom_mlp(geom_ao).squeeze(-1)

        rel_valid = actor_valid_mask.unsqueeze(-1) & anchor_mask.unsqueeze(1)
        anchor_logits = (qk_logits + geom_logits) / self.anchor_attn_tau
        anchor_attn = self._masked_softmax(anchor_logits, rel_valid, dim=-1)

        score = object_scores.float().clamp(min=0.0)

        def _shared(mask):
            weighted = anchor_attn * (score * mask.float()).unsqueeze(1)
            return torch.einsum("bim,bjm->bij", anchor_attn, weighted)

        shared_table = _shared(table_mask)
        shared_service = _shared(service_mask)
        anchor_input = torch.cat(
            (shared_table.unsqueeze(-1), shared_service.unsqueeze(-1), geom_feat),
            dim=-1,
        )
        raw_bias = torch.tanh(self.anchor_bias_mlp(anchor_input))
        if self.anchor_bias_clip > 0:
            raw_bias = raw_bias.clamp(min=-self.anchor_bias_clip, max=self.anchor_bias_clip)
        scale = self.anchor_scale_max * torch.sigmoid(self.anchor_scale_logit)
        scaled_bias = raw_bias * scale.view(1, 1, 1, self.nhead)
        scaled_bias = scaled_bias.permute(0, 3, 1, 2)
        diag_mask = torch.eye(n, device=scaled_bias.device, dtype=torch.bool).view(1, 1, n, n)
        scaled_bias = scaled_bias.masked_fill(diag_mask, 0.0)

        pair_valid = self._zero_diag(actor_valid_mask.unsqueeze(1) & actor_valid_mask.unsqueeze(2))
        head_pair_valid = pair_valid.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        valid_bias = scaled_bias[head_pair_valid] if pair_valid.any() else scaled_bias.new_zeros(0)
        if valid_bias.numel() > 0:
            zero_diag.update(
                {
                    "interaction_anchor_bias_mean": valid_bias.mean(),
                    "interaction_anchor_bias_abs_mean": valid_bias.abs().mean(),
                    "interaction_anchor_bias_max": valid_bias.max(),
                    "interaction_anchor_bias_min": valid_bias.min(),
                    "interaction_anchor_bias_pos_ratio": (valid_bias > 0).float().mean(),
                    "interaction_anchor_bias_neg_ratio": (valid_bias < 0).float().mean(),
                }
            )
        actor_valid_f = actor_valid_mask.float()
        actor_count = actor_valid_f.sum().clamp(min=1.0)
        anchor_top1 = anchor_attn.max(dim=-1).values
        zero_diag.update(
            {
                "interaction_anchor_shared_table_mean": self._masked_mean(shared_table, pair_valid),
                "interaction_anchor_shared_service_mean": self._masked_mean(shared_service, pair_valid),
                "interaction_anchor_scale_mean": scale.mean(),
                "interaction_anchor_top1_mean": (anchor_top1 * actor_valid_f).sum() / actor_count,
                "interaction_anchor_valid_per_actor": (
                    (rel_valid.float().sum(dim=-1) * actor_valid_f).sum() / actor_count
                ),
            }
        )
        return scaled_bias, shared_table, shared_service, zero_diag

    def forward(
        self,
        x,
        boxes,
        attn_mask=None,
        actor_valid_mask=None,
        object_tokens=None,
        object_boxes_xyxy=None,
        object_scores=None,
        object_family_id=None,
        object_valid_mask=None,
    ):
        """
        x: [B*T, N, D]
        boxes: [B*T, N, 4] (cx, cy, w, h) normalized
        object_*: [B*T, M, ...] normalized object context
        attn_mask: [B*T, N, N] bool, True indicates positions to mask
        actor_valid_mask: [B*T, N] bool, True indicates real actors
        """
        b_tokens, n, d = x.shape
        assert d == self.d_model, "Feature dim mismatch"
        if actor_valid_mask is None:
            if attn_mask is not None:
                actor_valid_mask = ~(attn_mask.all(dim=-1))
            else:
                actor_valid_mask = torch.ones(b_tokens, n, device=x.device, dtype=torch.bool)

        q = self.q_proj(x).view(b_tokens, n, self.nhead, self.dk)
        k = self.k_proj(x).view(b_tokens, n, self.nhead, self.dk)
        v = self.v_proj(x).view(b_tokens, n, self.nhead, self.dk)

        geom_feat, dist = self._pairwise_geometry(boxes)
        geom_bias = self.geom_mlp(geom_feat) if self.use_geom_bias else None

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.dk)
        if geom_bias is not None:
            scores = scores + geom_bias.permute(0, 3, 1, 2)

        if self.use_logit_penalty:
            sigma = torch.exp(self.log_sigma) + 1e-6
            scores = scores - (dist.unsqueeze(1) ** 2) / (sigma ** 2)

        anchor_bias, shared_table, shared_service, diag = self._anchor_edge_bias(
            x=x,
            boxes=boxes,
            geom_feat=geom_feat,
            actor_valid_mask=actor_valid_mask,
            object_tokens=object_tokens,
            object_boxes_xyxy=object_boxes_xyxy,
            object_scores=object_scores,
            object_family_id=object_family_id,
            object_valid_mask=object_valid_mask,
        )
        scores = scores + anchor_bias

        if self.hard_mask_thresh is not None:
            hard_mask = dist > self.hard_mask_thresh
            diag_idx = torch.arange(n, device=dist.device)
            hard_mask[:, diag_idx, diag_idx] = False
            scores = scores.masked_fill(hard_mask.unsqueeze(1), float("-inf"))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        if self.topk and self.topk > 0 and self.topk < n:
            topk_values, topk_idx = torch.topk(scores, self.topk, dim=-1)
            new_mask = torch.ones_like(scores, dtype=torch.bool)
            new_mask.scatter_(-1, topk_idx, False)
            scores = scores.masked_fill(new_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        out = out.reshape(b_tokens, n, d)
        out = self.out_proj(out)
        x = self.norm1(x + self.dropout(out))

        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))

        diag["interaction_stir_enabled"] = x.new_tensor(1.0)
        return x, attn, diag


class TemporalSelfAttention(nn.Module):
    """
    Temporal self-attention over per-actor (or per-token) sequences of length T.
    """

    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, pos=None, key_padding_mask=None):
        """
        x: [B*N, T, D]
        pos: [1, T, D] optional positional encoding broadcast to batch
        key_padding_mask: [B*N, T] optional
        """
        if pos is not None:
            x_qk = x + pos
        else:
            x_qk = x

        attn_out, attn = self.mha(x_qk, x_qk, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))

        return x, attn


class TemporalEncoder(nn.Module):
    """
    Temporal encoder with TCN and stacked self-attention layers.
    """

    def __init__(self, d_model=256, nhead=4, num_layers=3, tcn_kernel_size=3, tcn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        # TCN block
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=tcn_kernel_size, padding=tcn_kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(tcn_dropout)
        )

        self.layers = nn.ModuleList([
            TemporalSelfAttention(d_model, nhead, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, pos=None, key_padding_mask=None):
        """
        x: [B*N, T, D]
        """
        # Apply TCN
        # x is [Batch, Time, Dim], Conv1d needs [Batch, Dim, Time]
        x_t = x.permute(0, 2, 1)
        x_t = self.tcn(x_t)
        x = x + x_t.permute(0, 2, 1)  # Add residual connection from TCN

        attn = None
        for layer in self.layers:
            x, attn = layer(x, pos, key_padding_mask)

        return x, attn
