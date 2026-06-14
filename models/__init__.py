from .group_matcher import build_group_matcher
from .criterion import SetCriterion
from .models import GADTR


def build_model(args):
    model = GADTR(args)

    losses = ['labels', 'cardinality']
    group_losses = ['group_labels', 'group_cardinality', 'group_code', 'group_consistency']
    if getattr(args, 'use_pairwise_refiner', True):
        if getattr(args, 'use_query_conditioned_pmr', False):
            group_losses.append('query_pairwise_group')
        else:
            group_losses.append('pairwise_group')
    if getattr(args, 'use_attach_head', False):
        group_losses.append('attach')
    if float(getattr(args, 'membership_margin_loss_coef', 0.0)) > 0.0:
        group_losses.append('membership_margin')

    # Set loss coefficients
    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef
    if getattr(args, 'use_pairwise_refiner', True):
        if getattr(args, 'use_query_conditioned_pmr', False):
            weight_dict['loss_query_pairwise_group'] = args.query_pairwise_loss_coef
        else:
            weight_dict['loss_pairwise_group'] = args.pairwise_loss_coef
    if getattr(args, 'use_attach_head', False):
        weight_dict['loss_attach'] = args.attach_loss_coef
    if float(getattr(args, 'membership_margin_loss_coef', 0.0)) > 0.0:
        weight_dict['loss_membership_margin'] = args.membership_margin_loss_coef

    # Group matching
    group_matcher = build_group_matcher(args)

    # Loss functions
    criterion = SetCriterion(args.num_class, weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses, group_losses=group_losses, group_matcher=group_matcher, args=args)

    criterion.cuda()

    return model, criterion
