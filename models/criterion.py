# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/HOTR)
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import copy
import numpy as np

from torch import nn

from util import box_ops
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, weight_dict, eos_coef, losses, group_losses=None,
                 group_matcher=None, args=None):
        """ Create the criterion.
        Parameters:
        num_classes: number of object categories, omitting the special no-object category
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        eos_coef: relative classification weight applied to the no-group activity class category
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        group_losses: list of all the group losses to be applied. See get_group_loss for list of available group losses.
        group_matcher: module able to compute a matching between targets and predictions
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef

        self.group_losses = group_losses
        self.group_matcher = group_matcher

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        empty_group_weight = torch.ones(self.num_classes + 1)
        empty_group_weight[-1] = args.group_eos_coef
        self.register_buffer('empty_group_weight', empty_group_weight)

        self.num_boxes = args.num_boxes

        # option
        self.temperature = args.temperature
        self.use_pairwise_refiner = bool(getattr(args, 'use_pairwise_refiner', True))
        self.use_attach_head = bool(getattr(args, 'use_attach_head', False))
        self.membership_member_margin = float(getattr(args, 'membership_member_margin', 0.5))
        self.membership_outlier_margin = float(getattr(args, 'membership_outlier_margin', 0.0))

    #######################################################################################################################
    # * Individual Losses
    #######################################################################################################################
    def loss_labels(self, outputs, targets, num_boxes, log=True):
        """Individual action classification loss (NLL)"""
        assert 'pred_actions' in outputs
        src_logits = outputs['pred_actions']
        target_classes = torch.cat([v["actions"] for v in targets], dim=0)

        loss_ce = 0.0

        src_logits_log = None
        tgt_classes_log = None

        for batch_idx in range(src_logits.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)
            src_logit = src_logits[batch_idx][non_dummy_idx].unsqueeze(0)
            target_class = target_classes[batch_idx][non_dummy_idx].unsqueeze(0)
            loss_ce += F.cross_entropy(src_logit.transpose(1, 2), target_class, self.empty_weight)

            if src_logits_log is None:
                src_logits_log = src_logit.squeeze(0)
                tgt_classes_log = target_class.squeeze(0)
            else:
                src_logits_log = torch.cat([src_logits_log, src_logit.squeeze(0)], dim=0)
                tgt_classes_log = torch.cat([tgt_classes_log, target_class.squeeze(0)], dim=0)

        loss_ce /= src_logits.shape[0]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits_log, tgt_classes_log)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, num_boxes):
        pred_logits = outputs['pred_actions']
        device = pred_logits.device
        # tgt_lengths = torch.as_tensor([len(v["actions"]) for v in targets], device=device)
        tgt_lengths = torch.as_tensor([len(k) for v in targets for k in v["actions"]], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    #######################################################################################################################
    # * Group Losses
    #######################################################################################################################
    def loss_group_labels(self, outputs, targets, group_indices, log=True):
        """ Group activity classification loss (NLL)"""
        assert 'pred_activities' in outputs
        src_logits = outputs['pred_activities']

        idx = self._get_src_permutation_idx(group_indices)
        flatten_targets = [u for t in targets for u in t["activities"]]
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_group_weight)
        losses = {'loss_group_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['group_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_group_cardinality(self, outputs, targets, group_indices):
        pred_logits = outputs['pred_activities']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(k) for v in targets for k in v["activities"]], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'group_cardinality_error': card_err}
        return losses

    def loss_group_code(self, outputs, targets, group_indices, log=True):
        """Membership loss"""
        sim = outputs['membership']

        idx = self._get_src_permutation_idx(group_indices)

        # Binary cross entropy loss
        flatten_targets = [u for t in targets for u in t["members"]]

        # target_members_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)]).type(torch.FloatTensor).to(sim.device)
        target_members_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)])
        target_members = torch.full(sim.shape, 0.0, dtype=torch.float, device=sim.device)
        target_members[idx] = target_members_o

        loss_membership = 0.0
        for batch_idx in range(sim.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)
            sim_batch = sim[batch_idx].transpose(0, 1)[non_dummy_idx].transpose(0, 1).unsqueeze(0)

            target_members_batch = target_members[batch_idx].transpose(0, 1)[non_dummy_idx].transpose(0, 1).unsqueeze(0)

            loss_membership += F.binary_cross_entropy(sim_batch, target_members_batch)
        loss_membership /= sim.shape[0]

        losses = {'loss_group_code': loss_membership}
        return losses

    def loss_group_consistency(self, outputs, targets, group_indices):
        """Group consistency loss"""
        actor_embeds = outputs['actor_embeddings']

        consistency_loss = 0.0

        for batch_idx in range(actor_embeds.shape[0]):
            membership = targets[batch_idx]["membership"][0]
            actor_embed = actor_embeds[batch_idx]                           # [n, f]

            cos = nn.CosineSimilarity(dim=-1)
            sim = cos(actor_embed.unsqueeze(1), actor_embed.unsqueeze(0)) / self.temperature

            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)

            N = len(non_dummy_idx[0])

            non_dummy_membership = membership[non_dummy_idx]

            group_count = 0

            for actor_idx in range(N):
                group_id = non_dummy_membership[actor_idx]

                if group_id != -1:
                    # Use tensor operations to avoid deprecated indexing warning
                    pos_indices = (non_dummy_membership == group_id).nonzero(as_tuple=True)[0]
                    positive_idx = pos_indices[pos_indices != actor_idx]
                    positive_samples = sim[actor_idx][positive_idx]

                    negative_idx = (non_dummy_membership != group_id).nonzero(as_tuple=True)
                    negative_samples = sim[actor_idx][negative_idx]

                    nominator = torch.exp(positive_samples)
                    denominator = torch.exp(torch.cat((positive_samples, negative_samples)))
                    loss_partial = -torch.log(torch.sum(nominator) / torch.sum(denominator))
                    group_count += 1

                    consistency_loss += loss_partial

            consistency_loss /= group_count

        consistency_loss /= actor_embeds.shape[0]
        losses = {'loss_consistency': consistency_loss}
        return losses

    def loss_pairwise_group(self, outputs, targets, group_indices=None, log=True):
        pair_logits = outputs['pairwise_affinity_logits']
        pair_valid = outputs['pairwise_valid_mask']

        loss_pair = pair_logits.new_tensor(0.0)
        pos_mean_acc = pair_logits.new_tensor(0.0)
        neg_mean_acc = pair_logits.new_tensor(0.0)
        valid_batches = 0

        for batch_idx in range(pair_logits.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze().bool()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)[0]
            if non_dummy_idx.numel() <= 1:
                continue

            membership = targets[batch_idx]["membership"][0][non_dummy_idx]
            pair_logits_batch = pair_logits[batch_idx][non_dummy_idx][:, non_dummy_idx]
            pair_valid_batch = pair_valid[batch_idx][non_dummy_idx][:, non_dummy_idx]

            tgt_i = membership.unsqueeze(1)
            tgt_j = membership.unsqueeze(0)
            positive = (tgt_i == tgt_j) & (tgt_i >= 0) & pair_valid_batch
            negative = (tgt_i != tgt_j) & pair_valid_batch
            outlier_pair = (tgt_i < 0) & (tgt_j < 0) & pair_valid_batch
            negative = negative | outlier_pair
            valid_mask = positive | negative
            if valid_mask.sum() == 0:
                continue

            target_pair = positive.float()
            loss_pair = loss_pair + F.binary_cross_entropy_with_logits(
                pair_logits_batch[valid_mask],
                target_pair[valid_mask],
            )

            pair_prob = torch.sigmoid(pair_logits_batch)
            if positive.any():
                pos_mean_acc = pos_mean_acc + pair_prob[positive].mean()
            if negative.any():
                neg_mean_acc = neg_mean_acc + pair_prob[negative].mean()
            valid_batches += 1

        if valid_batches > 0:
            loss_pair = loss_pair / valid_batches
            pos_mean = pos_mean_acc / valid_batches
            neg_mean = neg_mean_acc / valid_batches
        else:
            pos_mean = pair_logits.new_tensor(0.0)
            neg_mean = pair_logits.new_tensor(0.0)

        losses = {
            'loss_pairwise_group': loss_pair,
            'pair_pos_mean': pos_mean.detach(),
            'pair_neg_mean': neg_mean.detach(),
            'pair_gap': (pos_mean - neg_mean).detach(),
        }
        return losses

    def loss_attach(self, outputs, targets, group_indices=None, log=True):
        if 'attach_logits' not in outputs:
            zero = outputs['membership'].new_tensor(0.0)
            return {
                'loss_attach': zero,
                'attach_pos_mean': zero.detach(),
                'attach_neg_mean': zero.detach(),
                'attach_gap': zero.detach(),
                'attach_acc': zero.detach(),
            }

        attach_logits = outputs['attach_logits']
        loss_attach = attach_logits.new_tensor(0.0)
        pos_sum = attach_logits.new_tensor(0.0)
        neg_sum = attach_logits.new_tensor(0.0)
        correct_sum = attach_logits.new_tensor(0.0)
        valid_sum = attach_logits.new_tensor(0.0)
        pos_count = 0
        neg_count = 0
        valid_batches = 0

        for batch_idx in range(attach_logits.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze().bool()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)[0]
            if non_dummy_idx.numel() == 0:
                continue

            membership = targets[batch_idx]["membership"][0][non_dummy_idx]
            labels = (membership >= 0).float()
            logits = attach_logits[batch_idx][non_dummy_idx]

            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            pos_mask = labels > 0.5
            neg_mask = ~pos_mask
            if pos_mask.any() and neg_mask.any():
                batch_loss = 0.5 * bce[pos_mask].mean() + 0.5 * bce[neg_mask].mean()
            else:
                batch_loss = bce.mean()
            loss_attach = loss_attach + batch_loss
            valid_batches += 1

            prob = torch.sigmoid(logits)
            if pos_mask.any():
                pos_sum = pos_sum + prob[pos_mask].sum()
                pos_count += int(pos_mask.sum().item())
            if neg_mask.any():
                neg_sum = neg_sum + prob[neg_mask].sum()
                neg_count += int(neg_mask.sum().item())
            pred = prob > 0.5
            correct_sum = correct_sum + (pred == pos_mask).float().sum()
            valid_sum = valid_sum + labels.new_tensor(float(labels.numel()))

        if valid_batches > 0:
            loss_attach = loss_attach / valid_batches

        pos_mean = pos_sum / max(pos_count, 1)
        neg_mean = neg_sum / max(neg_count, 1)
        attach_acc = correct_sum / valid_sum.clamp(min=1.0)
        return {
            'loss_attach': loss_attach,
            'attach_pos_mean': pos_mean.detach(),
            'attach_neg_mean': neg_mean.detach(),
            'attach_gap': (pos_mean - neg_mean).detach(),
            'attach_acc': attach_acc.detach(),
        }

    def loss_membership_margin(self, outputs, targets, group_indices, log=True):
        membership_logits = outputs.get('membership_logits_refined', None)
        if membership_logits is None:
            membership_logits = outputs.get('membership_logits_base', None)
        if membership_logits is None:
            membership_logits = outputs['membership']

        idx = self._get_src_permutation_idx(group_indices)
        flatten_targets = [u for t in targets for u in t["members"]]
        target_members_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)])
        target_members = torch.full(membership_logits.shape, 0.0, dtype=torch.float, device=membership_logits.device)
        target_members[idx] = target_members_o.float()

        member_loss_sum = membership_logits.new_tensor(0.0)
        outlier_loss_sum = membership_logits.new_tensor(0.0)
        member_violation_sum = membership_logits.new_tensor(0.0)
        outlier_violation_sum = membership_logits.new_tensor(0.0)
        member_count = membership_logits.new_tensor(0.0)
        outlier_count = membership_logits.new_tensor(0.0)

        for batch_idx in range(membership_logits.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze().bool()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)[0]
            if non_dummy_idx.numel() == 0:
                continue

            logits_b = membership_logits[batch_idx][:, non_dummy_idx]  # [G, N_valid]
            target_b = target_members[batch_idx][:, non_dummy_idx] > 0.5
            membership = targets[batch_idx]["membership"][0][non_dummy_idx]

            member_actor_mask = (membership >= 0) & target_b.any(dim=0)
            if member_actor_mask.any():
                member_logits = logits_b[:, member_actor_mask].transpose(0, 1)  # [N_member, G]
                pos_mask = target_b[:, member_actor_mask].transpose(0, 1)
                pos_idx = pos_mask.float().argmax(dim=1)
                actor_idx = torch.arange(member_logits.shape[0], device=member_logits.device)
                correct_logits = member_logits[actor_idx, pos_idx]
                wrong_logits = member_logits.masked_fill(pos_mask, -1e4).max(dim=1).values
                member_margin_raw = self.membership_member_margin - correct_logits + wrong_logits
                member_loss_sum = member_loss_sum + F.softplus(member_margin_raw).sum()
                member_violation_sum = member_violation_sum + (member_margin_raw > 0.0).float().sum()
                member_count = member_count + member_logits.new_tensor(float(member_logits.shape[0]))

            outlier_actor_mask = membership < 0
            if outlier_actor_mask.any():
                outlier_logits = logits_b[:, outlier_actor_mask]
                outlier_max_logits = outlier_logits.max(dim=0).values
                outlier_margin_raw = outlier_max_logits - self.membership_outlier_margin
                outlier_loss_sum = outlier_loss_sum + F.softplus(outlier_margin_raw).sum()
                outlier_violation_sum = outlier_violation_sum + (outlier_margin_raw > 0.0).float().sum()
                outlier_count = outlier_count + outlier_logits.new_tensor(float(outlier_logits.shape[1]))

        member_loss = member_loss_sum / member_count.clamp(min=1.0)
        outlier_loss = outlier_loss_sum / outlier_count.clamp(min=1.0)
        member_part = (member_count > 0).float()
        outlier_part = (outlier_count > 0).float()
        active_parts = (member_part + outlier_part).clamp(min=1.0)
        loss_membership_margin = (member_loss * member_part + outlier_loss * outlier_part) / active_parts

        member_active = member_violation_sum / member_count.clamp(min=1.0)
        outlier_active = outlier_violation_sum / outlier_count.clamp(min=1.0)
        return {
            'loss_membership_margin': loss_membership_margin,
            'member_margin_loss': member_loss.detach(),
            'outlier_margin_loss': outlier_loss.detach(),
            'member_margin_active': member_active.detach(),
            'outlier_margin_active': outlier_active.detach(),
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # *****************************************************************************
    # >>> DETR Losses
    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    # >>> Group Losses
    def get_group_loss(self, loss, outputs, targets, group_indices, **kwargs):
        loss_map = {
            'group_labels': self.loss_group_labels,
            'group_cardinality': self.loss_group_cardinality,
            'group_code': self.loss_group_code,
            'group_consistency': self.loss_group_consistency,
            'pairwise_group': self.loss_pairwise_group,
            'attach': self.loss_attach,
            'membership_margin': self.loss_membership_margin,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, group_indices, **kwargs)

    # *****************************************************************************

    def forward(self, outputs, targets, log=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        num_boxes = sum(len(u) for t in targets for u in t["actions"])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        sim = outputs['membership']
        bs, num_queries, num_clip_boxes = sim.shape

        for tgt in targets:
            tgt["dummy_idx"] = torch.ones_like(tgt["actions"], dtype=int)
            for box_idx in range(num_clip_boxes):
                if bool(tgt["actions"][0, box_idx] == self.num_classes + 1):
                    tgt["dummy_idx"][0, box_idx] = 0

        input_targets = [copy.deepcopy(target) for target in targets]
        group_indices = self.group_matcher(outputs_without_aux, input_targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes))

        # Group activity detection losses
        for loss in self.group_losses:
            losses.update(self.get_group_loss(loss, outputs, targets, group_indices))

        return losses
