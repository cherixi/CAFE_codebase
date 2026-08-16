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


class StaticDynamicTemporalPool(nn.Module):
    """Pool temporal tokens through appearance and first-order dynamics."""

    def __init__(
        self,
        d_model=256,
        hidden_dim=64,
        dropout=0.1,
        dynamic_scale_init=0.1,
        dynamic_scale_max=0.5,
    ):
        super().__init__()
        if dynamic_scale_max <= 0.0:
            raise ValueError("dynamic_scale_max must be positive")
        if not 0.0 < dynamic_scale_init < dynamic_scale_max:
            raise ValueError("dynamic_scale_init must be in (0, dynamic_scale_max)")

        self.dynamic_scale_max = float(dynamic_scale_max)
        init_ratio = float(dynamic_scale_init) / self.dynamic_scale_max
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        self.dynamic_scale_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.static_score = nn.Linear(d_model, 1)
        self.dynamic_norm = nn.LayerNorm(d_model)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.dynamic_score = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.dynamic_proj.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _masked_softmax(logits, valid_mask):
        logits = logits.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        return torch.nan_to_num(weights, nan=0.0)

    def forward(self, x, key_padding_mask=None, static_logits=None):
        """
        Args:
            x: Temporal tokens with shape [B, T, D].
            key_padding_mask: Optional [B, T] mask, where True denotes padding.
            static_logits: Optional [B, T] logits from the existing temporal
                pooling head. Supplying them keeps SDTP as a residual extension
                of the established pooling path instead of replacing it.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, D] input, got {tuple(x.shape)}")

        batch_size, num_frames, _ = x.shape
        if key_padding_mask is None:
            valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=x.device)
        else:
            valid_mask = ~key_padding_mask.bool()

        if static_logits is None:
            static_logits = self.static_score(x).squeeze(-1)
        elif static_logits.shape != (batch_size, num_frames):
            raise ValueError(
                f"Expected static_logits {(batch_size, num_frames)}, "
                f"got {tuple(static_logits.shape)}"
            )
        static_weight = self._masked_softmax(static_logits, valid_mask)
        static_clip = torch.sum(x * static_weight.unsqueeze(-1), dim=1)

        delta = torch.zeros_like(x)
        if num_frames > 1:
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
        dynamic_tokens = self.dynamic_proj(self.dynamic_norm(delta))
        dynamic_logits = self.dynamic_score(torch.cat([x, delta.abs()], dim=-1)).squeeze(-1)

        dynamic_valid = valid_mask.clone()
        if num_frames > 1:
            dynamic_valid[:, 0] = False
        else:
            dynamic_valid.zero_()
        dynamic_weight = self._masked_softmax(dynamic_logits, dynamic_valid)
        dynamic_clip = torch.sum(dynamic_tokens * dynamic_weight.unsqueeze(-1), dim=1)

        gate = torch.sigmoid(self.fusion_gate(torch.cat([static_clip, dynamic_clip], dim=-1)))
        dynamic_scale = self.dynamic_scale_max * torch.sigmoid(self.dynamic_scale_logit)
        dynamic_residual = dynamic_scale * gate * dynamic_clip
        clip = static_clip + dynamic_residual

        eps = torch.finfo(x.dtype).eps if x.dtype.is_floating_point else 1e-6
        dynamic_ratio = dynamic_residual.norm(dim=-1) / static_clip.norm(dim=-1).clamp_min(eps)
        static_entropy = -(static_weight * static_weight.clamp_min(eps).log()).sum(dim=1)
        dynamic_entropy = -(dynamic_weight * dynamic_weight.clamp_min(eps).log()).sum(dim=1)
        diagnostics = {
            "gate_mean": gate.mean().detach(),
            "dynamic_scale": dynamic_scale.detach(),
            "dynamic_ratio": dynamic_ratio.mean().detach(),
            "static_entropy": static_entropy.mean().detach(),
            "dynamic_entropy": dynamic_entropy.mean().detach(),
        }
        return clip, diagnostics
