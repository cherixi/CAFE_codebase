from .group_matcher import build_group_matcher
from .criterion import SetCriterion
from .models import GADTR


def _resolve_conditional_defaults(args):
    use_interaction_stir = bool(getattr(args, 'use_interaction_stir', False))
    if not hasattr(args, 'use_olic') or getattr(args, 'use_olic') is None:
        args.use_olic = not use_interaction_stir
    if not hasattr(args, 'use_pairwise_refiner') or getattr(args, 'use_pairwise_refiner') is None:
        args.use_pairwise_refiner = not use_interaction_stir
    return args


def build_model(args):
    args = _resolve_conditional_defaults(args)
    model = GADTR(args)

    losses = ['labels', 'cardinality']
    group_losses = ['group_labels', 'group_cardinality', 'group_code', 'group_consistency']
    if args.use_pairwise_refiner:
        group_losses.append('pairwise_group')

    # Set loss coefficients
    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef
    if args.use_pairwise_refiner:
        weight_dict['loss_pairwise_group'] = args.pairwise_loss_coef

    # Group matching
    group_matcher = build_group_matcher(args)

    # Loss functions
    criterion = SetCriterion(args.num_class, weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses, group_losses=group_losses, group_matcher=group_matcher, args=args)

    criterion.cuda()

    return model, criterion
