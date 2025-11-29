import torch
import torch.nn as nn

class VideoMAEAdapter(nn.Module):
    def __init__(self, global_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # 1. 降维: 768 -> 256
        self.proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 门控系数生成器 (输出 0~1 的标量)
        self.actor_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        self.group_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    def forward(self, actor_tokens, group_tokens, global_feat):
        """
        :param actor_tokens: [B, N, C]
        :param group_tokens: [B, K, C]
        :param global_feat: [B, 768]
        """
        # global_feat: [B, 768]
        global_emb = self.proj(global_feat) # [B, C]
        
        # --- 增强 Actor ---
        # actor_tokens: [B, N, C]
        B, N, C = actor_tokens.shape
        global_expanded_actor = global_emb.unsqueeze(1).repeat(1, N, 1) # [B, N, C]
        
        # 计算融合系数 alpha
        alpha = self.actor_gate(torch.cat([actor_tokens, global_expanded_actor], dim=-1))
        actor_out = actor_tokens + alpha * global_expanded_actor # 残差融合
        
        # --- 增强 Group ---
        # group_tokens: [B, K, C]
        B, K, _ = group_tokens.shape
        global_expanded_group = global_emb.unsqueeze(1).repeat(1, K, 1) # [B, K, C]
        
        # 计算融合系数 beta
        beta = self.group_gate(torch.cat([group_tokens, global_expanded_group], dim=-1))
        group_out = group_tokens + beta * global_expanded_group
        
        return actor_out, group_out
