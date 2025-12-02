import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameHOIGraph(nn.Module):
    """
    Frame-level actor-to-actor graph modeling with masked self-attention.
    """

    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, attn_mask=None):
        """
        x: [B*T, N, D]
        attn_mask: [B*T, N, N] bool, True indicates positions to mask
        """
        b_tokens, n, d = x.shape
        assert d == self.d_model, "Feature dim mismatch"

        scores = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(d)  # [B*T, N, N]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, x)  # [B*T, N, D]
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
