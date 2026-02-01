import torch
import torch.nn as nn


class VideoMAEAdapter(nn.Module):
    def __init__(self, global_dim, hidden_dim, fusion="adaptive_two_branch", dropout=0.1):
        super().__init__()
        self.fusion = fusion

        # Project global feature to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Static concat projection (shared for actor/group)
        if fusion == "static_concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.concat_proj = None

        # Adaptive shared gate
        if fusion == "adaptive_shared":
            self.shared_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        else:
            self.shared_gate = None

        # Adaptive two-branch gates (actor + group)
        if fusion == "adaptive_two_branch":
            self.actor_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
            self.group_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        else:
            self.actor_gate = None
            self.group_gate = None

    def forward(self, actor_tokens, group_tokens, global_feat):
        """
        :param actor_tokens: [B, N, C]
        :param group_tokens: [B, K, C]
        :param global_feat: [B, D]
        """
        global_emb = self.proj(global_feat)  # [B, C]

        if self.fusion == "static_add":
            return (actor_tokens + global_emb.unsqueeze(1),
                    group_tokens + global_emb.unsqueeze(1))

        if self.fusion == "static_concat":
            _, n, _ = actor_tokens.shape
            _, k, _ = group_tokens.shape
            global_actor = global_emb.unsqueeze(1).expand(-1, n, -1)
            global_group = global_emb.unsqueeze(1).expand(-1, k, -1)
            actor_out = self.concat_proj(torch.cat([actor_tokens, global_actor], dim=-1))
            group_out = self.concat_proj(torch.cat([group_tokens, global_group], dim=-1))
            return actor_out, group_out

        if self.fusion == "static_pool":
            actor_pool = actor_tokens.mean(dim=1)  # [B, C]
            group_pool = group_tokens.mean(dim=1)  # [B, C]
            actor_delta = (actor_pool + global_emb).unsqueeze(1)
            group_delta = (group_pool + global_emb).unsqueeze(1)
            return (actor_tokens + actor_delta,
                    group_tokens + group_delta)

        if self.fusion == "adaptive_shared":
            _, n, _ = actor_tokens.shape
            _, k, _ = group_tokens.shape
            global_actor = global_emb.unsqueeze(1).expand(-1, n, -1)
            global_group = global_emb.unsqueeze(1).expand(-1, k, -1)
            alpha = self.shared_gate(torch.cat([actor_tokens, global_actor], dim=-1))
            beta = self.shared_gate(torch.cat([group_tokens, global_group], dim=-1))
            actor_out = actor_tokens + alpha * global_actor
            group_out = group_tokens + beta * global_group
            return actor_out, group_out

        if self.fusion == "adaptive_two_branch":
            _, n, _ = actor_tokens.shape
            _, k, _ = group_tokens.shape
            global_actor = global_emb.unsqueeze(1).expand(-1, n, -1)
            global_group = global_emb.unsqueeze(1).expand(-1, k, -1)
            alpha = self.actor_gate(torch.cat([actor_tokens, global_actor], dim=-1))
            beta = self.group_gate(torch.cat([group_tokens, global_group], dim=-1))
            actor_out = actor_tokens + alpha * global_actor
            group_out = group_tokens + beta * global_group
            return actor_out, group_out

        raise ValueError(f"Unknown VideoMAE fusion mode: {self.fusion}")
