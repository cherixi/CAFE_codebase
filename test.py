# gadtr-hoi-mapping
import argparse
import os
import time

import torch

from util import experiment
import util.misc as utils
from util.test_eval_utils import (
    build_eval_metrics,
    build_eval_model_and_criterion,
    build_test_loader,
    collect_prediction_rows,
    create_metric_logger,
    evaluate_prediction_file,
    get_name_to_vid,
    resolve_eval_args,
    run_eval_batch,
    setup_eval_environment,
    update_metric_logger,
    write_prediction_rows,
)


ACTIVITIES = ['Queueing', 'Ordering', 'Drinking', 'Working', 'Fighting', 'Selfie', 'Individual', 'No']


def build_parser():
    parser = argparse.ArgumentParser(description='Group Activity Detection train code')

    # Dataset specification
    parser.add_argument('--dataset', default='cafe', type=str, help='dataset name')
    parser.add_argument('--val_mode', action='store_true')
    parser.add_argument('--split', default='place', type=str, help='dataset split. place or view')
    parser.add_argument('--data_path', default='/home/ziyang/aixi/Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/', type=str, help='data path')
    parser.add_argument('--tracks_source', default='gt', type=str, choices=['gt', 'pred'],
                        help='which track pkl to load under data_path/cafe: gt_tracks.pkl or pred_tracks_aligned_to_gt_slots.pkl')
    parser.add_argument('--tracks_pkl_path', default='', type=str,
                        help='optional explicit pkl path; if set, it overrides --tracks_source')
    parser.add_argument('--image_width', default=1120, type=int, help='Image width to resize (1120 for DINOv2, 1280 for ResNet)')
    parser.add_argument('--image_height', default=630, type=int, help='Image height to resize (630 for DINOv2, 720 for ResNet)')
    parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
    parser.add_argument('--num_frame', default=5, type=int, help='number of frames for each clip')
    parser.add_argument('--num_class', default=6, type=int, help='number of activity classes')
    parser.add_argument('--no_mae', action='store_true', help='disable VideoMAE enhancement')
    parser.add_argument('--mae_version', default='v2', type=str, choices=['v1', 'v2'], help='VideoMAE version: v1 (768) or v2 (1408)')
    parser.add_argument('--videomae_feats_path', default='./videomae_features_giant', type=str, help='path to videomae features')
    parser.add_argument('--mae_fusion', default='adaptive_two_branch', type=str,
                        choices=['none', 'static_add', 'static_concat', 'static_pool', 'adaptive_shared', 'adaptive_two_branch'],
                        help='VideoMAE fusion mode')

    # Backbone parameters
    parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone (resnet18, resnet50, dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)')
    parser.add_argument('--dilation', action='store_true', help='use dilation or not')
    parser.add_argument('--frozen_batch_norm', action='store_true', help='use frozen batch normalization')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone parameters (for DINOv2)')
    parser.add_argument('--unfreeze_blocks', default=0, type=int, help='number of last transformer blocks to unfreeze (for DINOv2 partial finetuning)')
    parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

    # RoI Align parameters
    parser.add_argument('--num_boxes', default=14, type=int, help='maximum number of actors')
    parser.add_argument('--crop_size', default=5, type=int, help='roi align crop size')

    # Group Transformer
    parser.add_argument('--gar_nheads', default=4, type=int, help='number of heads')
    parser.add_argument('--gar_enc_layers', default=6, type=int, help='number of group transformer layers')
    parser.add_argument('--gar_ffn_dim', default=512, type=int, help='feed forward network dimension')
    parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
    parser.add_argument('--num_group_tokens', default=12, type=int, help='number of group tokens')
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--group_threshold', default=0.5, type=float, help='post processing threshold')
    parser.add_argument('--distance_threshold', default=0.2, type=float, help='distance mask threshold')
    parser.add_argument('--hoi_nheads', default=4, type=int, help='number of heads for HOI graph')
    parser.add_argument('--hoi_topk', default=0, type=int, help='topk for HOI graph sparsity (0 for full)')
    parser.add_argument('--hoi_mode', default='penalty', type=str,
                        choices=['none', 'bias', 'hard_mask', 'penalty'],
                        help='HOI graph ablation mode')
    parser.add_argument('--hoi_hard_thresh', default=None, type=float,
                        help='distance threshold for hard_mask mode (if None, use distance_threshold)')
    parser.add_argument('--temporal_agg_mode', default='learned_pool', type=str,
                        choices=['learned_pool', 'frame_mean_main'],
                        help='temporal aggregation mode: learned pooling (default) or main-style frame mean ablation')
    parser.add_argument('--temporal_layers', default=3, type=int, help='number of temporal attention layers')
    parser.add_argument('--tcn_kernel_size', default=3, type=int, help='kernel size for TCN')
    parser.add_argument('--tcn_dropout', default=0.1, type=float, help='dropout for TCN')

    # OLIC
    olic_group = parser.add_mutually_exclusive_group()
    olic_group.add_argument('--use_olic', dest='use_olic', action='store_true',
                            help='enable OLIC object-conditioned branch')
    olic_group.add_argument('--no_olic', dest='use_olic', action='store_false',
                            help='disable OLIC object-conditioned branch')
    parser.set_defaults(use_olic=True)
    group_olic_group = parser.add_mutually_exclusive_group()
    group_olic_group.add_argument('--disable_group_olic', dest='disable_group_olic', action='store_true',
                                  help='disable group-side OLIC fusion and keep actor-side OLIC only')
    group_olic_group.add_argument('--enable_group_olic', dest='disable_group_olic', action='store_false',
                                  help='enable group-side OLIC fusion')
    parser.set_defaults(disable_group_olic=True)
    parser.add_argument('--object_tracks_pkl', default='', type=str,
                        help='path to object track pkl; default: <data_path>/cafe/object_tracks_gdino_swinb_localmix_membership.pkl')
    parser.add_argument('--num_object_boxes', default=20, type=int, help='fixed number of object boxes per frame')
    parser.add_argument('--olic_topk_obj', default=6, type=int,
                        help='(deprecated) top-k objects per actor for relevance pruning; pruning is disabled in current stable path')
    parser.add_argument('--olic_dropout', default=-1.0, type=float,
                        help='dropout for OLIC residual branch; <0 means use drop_rate')
    parser.add_argument('--olic_use_ffn', action='store_true',
                        help='enable optional OLIC FFN residual block')
    parser.add_argument('--olic_ffn_scale_init', default=0.0, type=float,
                        help='initial residual scale gamma for optional OLIC FFN')
    parser.add_argument('--olic_score_use', default='prune_relevance', type=str,
                        choices=['prune_relevance'],
                        help='how detector score is used in OLIC (v1 fixed to prune_relevance)')
    parser.add_argument('--olic_res_scale_init', default=0.0, type=float,
                        help='initial residual scale gamma for OLIC residual fusion')
    parser.add_argument('--olic_gate_init_bias', default=-4.0, type=float,
                        help='initial bias for OLIC gate heads')
    parser.add_argument('--olic_warmup_epochs', default=0, type=int,
                        help='warmup epochs for OLIC (in test this is typically 0/full)')
    parser.add_argument('--olic_attn_tau', default=2.0, type=float,
                        help='softmax temperature for actor-object routing')
    parser.add_argument('--anchor_attn_tau', default=3.0, type=float,
                        help='softmax temperature for anchor-object routing')
    dual_olic_group = parser.add_mutually_exclusive_group()
    dual_olic_group.add_argument('--use_dual_object_channels', dest='use_dual_object_channels', action='store_true',
                                 help='split objects into small-object OLIC and anchor-aware PMR channels')
    dual_olic_group.add_argument('--no_dual_object_channels', dest='use_dual_object_channels', action='store_false',
                                 help='disable dual object channels and fall back to single-channel OLIC')
    parser.set_defaults(use_dual_object_channels=True)
    parser.add_argument('--olic_geom_scale_init', default=1.0, type=float,
                        help='initial scale for geometry bias term in OLIC routing')
    parser.add_argument('--olic_geom_scale_max', default=2.0, type=float,
                        help='maximum geometry scale for OLIC routing')
    pairwise_group = parser.add_mutually_exclusive_group()
    pairwise_group.add_argument('--use_pairwise_refiner', dest='use_pairwise_refiner', action='store_true',
                                help='enable pairwise membership refiner')
    pairwise_group.add_argument('--no_pairwise_refiner', dest='use_pairwise_refiner', action='store_false',
                                help='disable pairwise membership refiner')
    parser.set_defaults(use_pairwise_refiner=True)
    objrel_group = parser.add_mutually_exclusive_group()
    objrel_group.add_argument('--pairwise_use_object_relation', dest='pairwise_use_object_relation', action='store_true',
                              help='use actor-side object summaries in pairwise affinity')
    objrel_group.add_argument('--no_pairwise_use_object_relation', dest='pairwise_use_object_relation', action='store_false',
                              help='disable object relation feature in pairwise affinity')
    parser.set_defaults(pairwise_use_object_relation=True)
    small_objrel_group = parser.add_mutually_exclusive_group()
    small_objrel_group.add_argument('--pairwise_use_small_object_relation', dest='pairwise_use_small_object_relation', action='store_true',
                                    help='use small-object clip relation in pairwise affinity')
    small_objrel_group.add_argument('--no_pairwise_use_small_object_relation', dest='pairwise_use_small_object_relation', action='store_false',
                                    help='disable small-object clip relation in pairwise affinity')
    parser.set_defaults(pairwise_use_small_object_relation=True)
    anchor_objrel_group = parser.add_mutually_exclusive_group()
    anchor_objrel_group.add_argument('--pairwise_use_anchor_relation', dest='pairwise_use_anchor_relation', action='store_true',
                                     help='use shared table/service anchor relation in pairwise affinity')
    anchor_objrel_group.add_argument('--no_pairwise_use_anchor_relation', dest='pairwise_use_anchor_relation', action='store_false',
                                     help='disable shared table/service anchor relation in pairwise affinity')
    parser.set_defaults(pairwise_use_anchor_relation=True)
    parser.add_argument('--pmr_anchor_source', default='auto', type=str, choices=['auto', 'gdino', 'yolo'],
                        help='anchor relation source for PMR: gdino keeps shared_table/shared_service, yolo uses table-only anchor relation, auto infers from object_tracks_pkl')
    geomrel_group = parser.add_mutually_exclusive_group()
    geomrel_group.add_argument('--pairwise_use_geom_relation', dest='pairwise_use_geom_relation', action='store_true',
                               help='use clip-level geometry in pairwise affinity')
    geomrel_group.add_argument('--no_pairwise_use_geom_relation', dest='pairwise_use_geom_relation', action='store_false',
                               help='disable geometry feature in pairwise affinity')
    parser.set_defaults(pairwise_use_geom_relation=True)
    parser.add_argument('--pairwise_loss_coef', default=0.25, type=float,
                        help='loss weight for pairwise same-group supervision')
    parser.add_argument('--pairwise_refine_scale', default=0.5, type=float,
                        help='residual scale for pairwise membership refinement')
    attach_group = parser.add_mutually_exclusive_group()
    attach_group.add_argument('--use_attach_head', dest='use_attach_head', action='store_true',
                              help='enable actor-level attach/outlier head')
    attach_group.add_argument('--no_attach_head', dest='use_attach_head', action='store_false',
                              help='disable attach/outlier head for old checkpoints')
    parser.set_defaults(use_attach_head=True)
    parser.add_argument('--attach_infer_mode', default='joint', type=str,
                        choices=['membership_only', 'attach_only', 'joint'],
                        help='score used by group_threshold at inference')
    attach_gate_group = parser.add_mutually_exclusive_group()
    attach_gate_group.add_argument('--attach_gate_in_pmr', dest='attach_gate_in_pmr', action='store_true',
                                   help='gate PMR propagation by attach probability')
    attach_gate_group.add_argument('--no_attach_gate_in_pmr', dest='attach_gate_in_pmr', action='store_false',
                                   help='do not gate PMR propagation by attach probability')
    parser.set_defaults(attach_gate_in_pmr=False)
    attach_detach_group = parser.add_mutually_exclusive_group()
    attach_detach_group.add_argument('--attach_gate_detach', dest='attach_gate_detach', action='store_true',
                                     help='detach attach probability before using it as PMR gate')
    attach_detach_group.add_argument('--no_attach_gate_detach', dest='attach_gate_detach', action='store_false',
                                     help='allow PMR gate gradients to flow into attach head')
    parser.set_defaults(attach_gate_detach=True)

    # Box noise ablation
    parser.add_argument('--box_noise_policy', default='none', type=str,
                        choices=['none', 'infer_only', 'train_and_infer'],
                        help='box noise policy: none / infer_only / train_and_infer')
    parser.add_argument('--box_noise_seed', default=1, type=int,
                        help='base seed for deterministic box noise sampling')
    parser.add_argument('--box_noise_center_std', default=0.10, type=float,
                        help='center offset noise std, relative to box size')
    parser.add_argument('--box_noise_scale_std', default=0.08, type=float,
                        help='log-scale noise std for box size')
    parser.add_argument('--box_noise_aspect_std', default=0.08, type=float,
                        help='log-aspect-ratio noise std')
    parser.add_argument('--box_noise_min_size', default=1e-4, type=float,
                        help='minimum normalized box size after noise')
    parser.add_argument('--box_noise_max_size', default=1.0, type=float,
                        help='maximum normalized box size after noise')

    # Loss option
    parser.add_argument('--temperature', default=0.2, type=float, help='consistency loss temperature')

    # Loss coefficients
    parser.add_argument('--ce_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--group_eos_coef', default=1, type=float)
    parser.add_argument('--group_ce_loss_coef', default=1, type=float)
    parser.add_argument('--group_code_loss_coef', default=5, type=float)
    parser.add_argument('--consistency_loss_coef', default=2, type=float)
    parser.add_argument('--attach_loss_coef', default=0.5, type=float)

    # Matcher
    parser.add_argument('--set_cost_group_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_membership', default=1, type=float,
                        help="Membership coefficient in the matching cost")

    # Training parameters
    parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
    parser.add_argument('--batch', default=16, type=int, help='Batch size')
    parser.add_argument('--test_batch', default=16, type=int, help='Test batch size')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--eval_num_workers', default=2, type=int,
                        help='DataLoader workers for evaluation')
    parser.add_argument('--eval_persistent_workers', action='store_true',
                        help='keep eval DataLoader workers alive when eval_num_workers > 0')
    parser.add_argument('--eval_prefetch_factor', default=2, type=int,
                        help='prefetch factor when eval_num_workers > 0')

    # GPU
    parser.add_argument('--device', default="0, 1", type=str, help='GPU device')
    parser.add_argument('--distributed', action='store_true')

    # Load model
    parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

    # Visualization
    parser.add_argument('--result_path', default="./outputs/")

    # Evaluation
    parser.add_argument('--groundtruth', default='./evaluation/gt_tracks.txt', type=argparse.FileType("r"))
    parser.add_argument('--labelmap', default='./label_map/group_action_list.pbtxt', type=argparse.FileType("r"))
    parser.add_argument('--giou_thresh', default=1.0, type=float)
    parser.add_argument('--eval_type', default="gt_base", type=str, help='gt_based or detection_based')
    parser.add_argument('--eval_image_width', default=1280, type=int, help='Image width for evaluation (must match gt_tracks.txt)')
    parser.add_argument('--eval_image_height', default=720, type=int, help='Image height for evaluation (must match gt_tracks.txt)')

    return parser


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack([image for image in batch[0]])
    return tuple(batch)


@torch.no_grad()
def validate(test_loader, model, criterion, metrics, args, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = create_metric_logger()
    header = 'Evaluation Inference: '
    print_freq = len(test_loader)
    name_to_vid = get_name_to_vid()
    file_path = os.path.join(output_dir, f'pred_group_test_{args.split}.txt')

    print(f"Processing {len(test_loader)} batches...")
    print(f"Saving predictions to: {file_path}")

    rows = []
    for i, (images, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        if i % max(1, len(test_loader) // 10) == 0:
            print(f"  Progress: {i}/{len(test_loader)} batches ({100 * i // len(test_loader)}%)")

        clean_boxes, outputs, loss_dict = run_eval_batch(model, criterion, images, targets, infos, args)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        update_metric_logger(metric_logger, loss_dict_reduced, criterion.weight_dict, outputs, args)
        rows.extend(collect_prediction_rows(clean_boxes, infos, outputs, args, name_to_vid=name_to_vid))

    write_prediction_rows(rows, file_path, args.group_threshold, args)

    metric_logger.synchronize_between_processes()
    print("\nAveraged stats:", metric_logger)

    print("\nEvaluating predictions...")
    result = evaluate_prediction_file(metrics, file_path)
    print("Evaluation completed!")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_eval_args(args)
    setup_eval_environment(args)

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_GAD_[%s]' % (args.dataset, time_str)
    output_dir = os.path.join(args.result_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    experiment.save_args(os.path.join(output_dir, "args.json"), vars(args))

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Output path: {output_dir}")
    print(f"Dataset: {args.dataset}, Split: {args.split}")
    print(f"Model path: {args.model_path}")
    print("=" * 60)

    print("\n[1/6] Setting random seed...")
    print("[2/6] Loading dataset...")
    test_set, test_loader = build_test_loader(args, collate_fn)
    print(f"    Dataset loaded: {len(test_set)} test samples")
    print("[3/6] Creating data loader...")
    print(f"    Data loader created: {len(test_loader)} batches")

    print("[4/6] Building model...")
    model, criterion = build_eval_model_and_criterion(args)
    print("    Model built and moved to GPU")

    print("[5/6] Model weights loaded")
    print("[6/6] Initializing evaluation metrics...")
    metrics = build_eval_metrics(args)
    print("    Metrics initialized")

    print("\n" + "=" * 60)
    print("Starting evaluation...")
    print("=" * 60 + "\n")
    _, result = validate(test_loader, model, criterion, metrics, args, output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("group mAP at 1.0: %.2f" % result['group_mAP_1.0'])
    print("group mAP at 0.5: %.2f" % result['group_mAP_0.5'])
    print("outlier mIoU: %.2f" % result['outlier_mIoU'])
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
