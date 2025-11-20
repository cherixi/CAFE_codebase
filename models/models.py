import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign

from .backbone import build_backbone
from .hoi_graph import FrameHOIGraph, TemporalSelfAttention
from .feed_forward import MLP


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

        # Frame-level HOI graph and temporal modeling
        self.frame_graph = FrameHOIGraph(self.hidden_dim, dropout=args.drop_rate)
        self.temporal_encoder = TemporalSelfAttention(self.hidden_dim, nhead=args.gar_nheads, dropout=args.drop_rate)
        self.time_pos_emb = nn.Embedding(self.num_frame, self.hidden_dim)

        self.num_group_tokens = args.num_group_tokens
        self.group_query_emb = nn.Embedding(self.num_group_tokens, self.hidden_dim)
        
        # Group activity classfication head
        self.group_emb = nn.Linear(self.hidden_dim, self.num_class + 1)

        # Temporal weighting for actors
        self.actor_time_pool = nn.Linear(self.hidden_dim, 1)
        
        # Distance mask threshold
        self.distance_threshold = args.distance_threshold

        # Membership prediction heads
        self.actor_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.group_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)

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

    def forward(self, x, boxes, dummy_mask):
        """
        :param x: [B, T, 3, H, W]
        :param boxes: [B, T, N, 4]
        :param dummy_mask: [B, N]
        :return:
        """
        bs, t, _, h, w = x.shape
        n = boxes.shape[2]

        boxes = torch.reshape(boxes, (-1, 4))                                           # [b x t x n, 4]
        boxes_flat = boxes.clone().detach()
        boxes_idx = [i * torch.ones(n, dtype=torch.int) for i in range(bs * t)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes.device)
        boxes_idx_flat = torch.reshape(boxes_idx, (bs * t * n, ))                       # [b x t x n]

        features, pos = self.backbone(x)
        _, c, oh, ow = features.shape                                                   # [b x t, d, oh, ow]

        src = self.input_proj(features)
        src = torch.reshape(src, (bs, t, -1, oh, ow))                                   # [b, t, c, oh, ow]

        # calculate distance & distance mask
        boxes_center = boxes.clone().detach()
        boxes_center = torch.reshape(boxes_center[:, :2], (-1, n, 2))
        boxes_distance = self.calculate_pairwise_distnace(boxes_center)

        distance_mask = (boxes_distance > self.distance_threshold)

        # ignore dummy boxes (padded boxes to match the number of actors)
        dummy_mask = dummy_mask.unsqueeze(1).repeat(1, t, 1).reshape(-1, n)
        actor_dummy_mask = (~dummy_mask.unsqueeze(2)).float() @ (~dummy_mask.unsqueeze(1)).float()
        dummy_diag = (dummy_mask.unsqueeze(2).float() @ dummy_mask.unsqueeze(1).float()).nonzero(as_tuple=True)
        actor_mask = ~(actor_dummy_mask.bool())
        actor_mask[dummy_diag] = False
        actor_mask = distance_mask | actor_mask

        boxes_flat[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * ow
        boxes_flat[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * oh
        boxes_flat[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * ow
        boxes_flat[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * oh

        boxes_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False

        # extract actor features
        # torchvision RoIAlign expects List[Tensor[N, 4]], so we split by batch
        boxes_list = [boxes_flat[boxes_idx_flat == i] for i in range(bs * t)]
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

        # frame-level HOI graph
        actor_graph_in = actor_features.reshape(bs * t, n, self.hidden_dim)
        actor_graph_out, _ = self.frame_graph(actor_graph_in, attn_mask=actor_mask)
        actor_graph_out = actor_graph_out.reshape(bs, t, n, self.hidden_dim)

        # temporal encoder across frames per actor
        temporal_in = actor_graph_out.permute(0, 2, 1, 3).reshape(bs * n, t, self.hidden_dim)  # [b*n, t, c]
        time_pos = self.time_pos_emb.weight[:t].unsqueeze(0)                                   # [1, t, c]
        temporal_out, _ = self.temporal_encoder(temporal_in, pos=time_pos)

        # temporal weighting per actor
        actor_time_logits = self.actor_time_pool(temporal_out).squeeze(-1)                     # [b*n, t]
        actor_time_weight = torch.softmax(actor_time_logits, dim=1).unsqueeze(-1)              # [b*n, t, 1]
        actor_clip = (temporal_out * actor_time_weight).sum(dim=1).reshape(bs, n, self.hidden_dim)

        # group tokens attend to actor clip features
        group_queries = self.group_query_emb.weight.unsqueeze(0).repeat(bs, 1, 1)              # [b, k, c]
        group_attn = torch.softmax(torch.matmul(group_queries, actor_clip.transpose(1, 2)) / math.sqrt(self.hidden_dim), dim=-1)
        group_repr_raw = torch.bmm(group_attn, actor_clip)                                      # [b, k, c]
        group_repr = F.normalize(group_repr_raw, p=2, dim=2)

        # normalize
        inst_repr = F.normalize(actor_clip, p=2, dim=2)

        # prediction heads
        outputs_class = self.class_emb(actor_clip)                      # [b, n, num_class+1]
        outputs_group_class = self.group_emb(group_repr_raw)            # [b, k, num_class+1]

        outputs_actor_emb = self.actor_match_emb(inst_repr)
        outputs_group_emb = self.group_match_emb(group_repr)

        membership = torch.bmm(outputs_group_emb, outputs_actor_emb.transpose(1, 2))
        membership = F.softmax(membership, dim=1)

        out = {
            "pred_actions": outputs_class,
            "pred_activities": outputs_group_class,
            "membership": membership.reshape(bs, self.num_group_tokens, self.num_boxes),
            "actor_embeddings": F.normalize(actor_clip, p=2, dim=2),
        }

        return out
