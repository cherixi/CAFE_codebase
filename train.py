# gadtr-hoi-mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import os
import math
import sys
import copy
import time
import random
import numpy as np
import argparse
import gc

from models import build_model
from util.utils import *
import util.misc as utils
import util.logger as loggers
from dataloader.dataloader import read_dataset
import evaluation.cafe_eval as evaluation
from util import experiment
from util.box_noise import apply_box_noise

parser = argparse.ArgumentParser(description='Group Activity Detection train code', add_help=False)

# Dataset specification
parser.add_argument('--dataset', default='cafe', type=str, help='dataset name')
parser.add_argument('--val_mode', action='store_true')
parser.add_argument('--split', default='place', type=str, help='dataset split. place or view')
# parser.add_argument('--data_path', default='/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/', type=str, help='data path')
parser.add_argument('--data_path', default='/home/ziyang/aixi/Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/', type=str, help='data path')
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
parser.add_argument('--freeze_backbone', action='store_true', help='freeze ALL backbone parameters (for DINOv2)')
parser.add_argument('--unfreeze_blocks', default=0, type=int, help='number of last transformer blocks to unfreeze (for DINOv2 partial finetuning, 0=freeze all if freeze_backbone, or finetune all)')
parser.add_argument('--backbone_lr_scale', default=0.1, type=float, help='backbone learning rate = base_lr * backbone_lr_scale (for layer-wise LR)')
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

# HOI Graph
parser.add_argument('--hoi_nheads', default=4, type=int, help='number of heads for HOI graph')
parser.add_argument('--hoi_topk', default=0, type=int, help='topk for HOI graph sparsity (0 for full)')
parser.add_argument('--hoi_mode', default='penalty', type=str,
                    choices=['none', 'bias', 'hard_mask', 'penalty'],
                    help='HOI graph ablation mode')
parser.add_argument('--hoi_hard_thresh', default=None, type=float,
                    help='distance threshold for hard_mask mode (if None, use distance_threshold)')

# Temporal Modeling
parser.add_argument('--temporal_layers', default=3, type=int, help='number of temporal attention layers')
parser.add_argument('--tcn_kernel_size', default=3, type=int, help='kernel size for TCN')
parser.add_argument('--tcn_dropout', default=0.1, type=float, help='dropout for TCN')
parser.add_argument('--temporal_agg_mode', default='learned_pool', type=str,
                    choices=['learned_pool', 'frame_mean_main'],
                    help='temporal aggregation mode: learned pooling (default) or main-style frame mean ablation')

# Loss option
parser.add_argument('--temperature', default=0.2, type=float, help='consistency loss temperature')

# Loss coefficients (Individual)
parser.add_argument('--ce_loss_coef', default=1, type=float)
parser.add_argument('--eos_coef', default=1, type=float,
                    help="Relative classification weight of the no-object class")

# Loss coefficients (Group)
parser.add_argument('--group_eos_coef', default=1, type=float)
parser.add_argument('--group_ce_loss_coef', default=1, type=float)
parser.add_argument('--group_code_loss_coef', default=5, type=float)
parser.add_argument('--consistency_loss_coef', default=2, type=float)

# Matcher (Group)
parser.add_argument('--set_cost_group_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_membership', default=1, type=float,
                    help="Membership coefficient in the matching cost")

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
parser.add_argument('--skip_test_epochs', default=8, type=int,
                    help='skip validation for the first N epochs to save time')
parser.add_argument('--batch', default=16, type=int, help='Batch size')
parser.add_argument('--test_batch', default=16, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-5, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-4, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=4, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

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

# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')
parser.add_argument('--distributed', action='store_true')

# Load model
parser.add_argument('--load_model', action='store_true', help='load model')
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

args = parser.parse_args()
path = None

SEQS_CAFE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

ACTIVITIES = ['Queueing', 'Ordering', 'Drinking', 'Working', 'Fighting', 'Selfie', 'Individual', 'No']


def main():
    global args, path

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_GAD_[%s]' % (args.dataset, time_str)
    save_path = './result/%s' % exp_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Logic for use_mae
    if args.no_mae:
        args.mae_fusion = 'none'
    args.use_mae = (not args.no_mae) and args.mae_fusion != 'none'

    if args.use_mae:
        print_log(save_path, f"----------------------------------------------------------------")
        print_log(save_path, f"VideoMAE Enhancement: ENABLED")
        print_log(save_path, f"Version: {args.mae_version.upper()} (Dim: {1408 if args.mae_version == 'v2' else 768})")
        print_log(save_path, f"Fusion: {args.mae_fusion}")
        print_log(save_path, f"Feature Path: {args.videomae_feats_path}")
        print_log(save_path, f"----------------------------------------------------------------")
    else:
        print_log(save_path, f"----------------------------------------------------------------")
        print_log(save_path, f"VideoMAE Enhancement: DISABLED")
        print_log(save_path, f"----------------------------------------------------------------")

    # Set MAE dimension
    if args.use_mae:
        args.mae_dim = 1408 if args.mae_version == 'v2' else 768
    else:
        args.mae_dim = 0

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set = read_dataset(args)

    # for variable length input
    if args.distributed:
        sampler_train = data.DistributedSampler(train_set, shuffle=True)
        sampler_test = data.DistributedSampler(test_set, shuffle=False)
    else:
        sampler_train = data.RandomSampler(train_set)
        sampler_test = data.RandomSampler(test_set)

    batch_sampler_train = data.BatchSampler(sampler_train, args.batch, drop_last=True)

    # 优化 DataLoader 配置以防止内存问题
    # - num_workers=2: 降低并发进程数，减少内存压力
    # - pin_memory=False: 关闭锁页内存，减少系统内存占用
    # - persistent_workers=True: 保持 worker 存活，避免每个 Epoch 重新 fork 进程
    # - prefetch_factor=2: 限制预取数量
    train_loader = data.DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, 
                                   num_workers=2, 
                                   pin_memory=False,
                                   persistent_workers=True,
                                   prefetch_factor=2)
    test_loader = data.DataLoader(test_set, args.test_batch, sampler=sampler_test, drop_last=False,
                                  collate_fn=collate_fn, 
                                  num_workers=2, 
                                  pin_memory=False,
                                  persistent_workers=True,
                                  prefetch_factor=2)

    model, criterion = build_model(args)
    model = torch.nn.DataParallel(model).cuda()

    # get the number of model parameters
    total_params = sum([p.data.nelement() for p in model.parameters()])
    trainable_params = sum([p.data.nelement() for p in model.parameters() if p.requires_grad])
    frozen_params = total_params - trainable_params
    
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, f'Total parameters: {total_params:,}')
    print_log(save_path, f'Trainable parameters: {trainable_params:,}')
    print_log(save_path, f'Frozen parameters: {frozen_params:,}')
    if total_params > 0:
        print_log(save_path, f'Frozen ratio: {frozen_params / total_params * 100:.1f}%')

    # define loss function and optimizer with layer-wise learning rate
    # 分层学习率：Backbone 参数使用较小学习率，其他参数使用基础学习率
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # DataParallel 包装后，参数名称会有 'module.' 前缀
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # 计算 Backbone 学习率
    backbone_lr = args.lr * args.backbone_lr_scale
    
    print_log(save_path, '--------------------Learning Rate Configuration--------------------')
    print_log(save_path, f'Head learning rate: {args.lr}')
    print_log(save_path, f'Backbone learning rate: {backbone_lr} (scale: {args.backbone_lr_scale})')
    print_log(save_path, f'Backbone params count: {sum(p.numel() for p in backbone_params):,}')
    print_log(save_path, f'Head params count: {sum(p.numel() for p in head_params):,}')
    
    # 构建参数组
    param_groups = [
        {'params': head_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': backbone_lr}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)

    # 注意: CyclicLR 对多参数组的支持有限，这里改用 CosineAnnealingLR 或保持 CyclicLR
    # CyclicLR 会按比例缩放各参数组的学习率
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                  base_lr=[args.lr, backbone_lr], 
                                                  max_lr=[args.max_lr, args.max_lr * args.backbone_lr_scale], 
                                                  step_size_up=args.lr_step,
                                                  step_size_down=args.lr_step_down, 
                                                  mode='triangular2',
                                                  cycle_momentum=False)

    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    path = args.result_path + exp_name
    if not os.path.exists(path):
        os.makedirs(path)

    metrics = evaluation.GAD_Evaluation(args)

    # experiment logging
    history = {"train": [], "val": []}
    best = {}
    args_json_path = os.path.join(save_path, "args.json")
    experiment.save_args(args_json_path, vars(args))

    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        
        # 每个 Epoch 结束后强制垃圾回收，防止内存累积
        gc.collect()
        torch.cuda.empty_cache()
        
        print_log(save_path, 'Loss: %.4f' % (train_log['loss']))
        print_log(save_path, 'Group class error: %.2f' % (train_log['group_class_error']))
        print('Current learning rate is %f' % scheduler.get_last_lr()[0])
        scheduler.step()

        # record train metrics
        history = experiment.update_history(history, "train", epoch, train_log)

        if epoch > args.skip_test_epochs and epoch % args.test_freq == 0:
            print_log(save_path, '----- %s at epoch #%d' % ("Test", epoch))
            test_log, result = validate(test_loader, model, criterion, metrics, epoch)
            print_log(save_path, 'Loss: %.4f' % (test_log['loss']))
            print_log(save_path, 'Group class error: %.2f' % (test_log['group_class_error']))
            print_log(save_path, "group mAP at 1.0: %.2f" % result['group_mAP_1.0'])
            print_log(save_path, "group mAP at 0.5: %.2f" % result['group_mAP_0.5'])
            print_log(save_path, "outlier mIoU: %.2f" % result['outlier_mIoU'])

            # merge metrics
            val_metrics = dict(test_log)
            val_metrics.update({
                "group_mAP_1.0": result['group_mAP_1.0'],
                "group_mAP_0.5": result['group_mAP_0.5'],
                "outlier_mIoU": result['outlier_mIoU'],
            })
            history = experiment.update_history(history, "val", epoch, val_metrics)
            best = experiment.update_best(best, epoch, {**val_metrics, "loss": test_log['loss']})

            # save summary and curves
            summary_path = os.path.join(save_path, "summary.json")
            experiment.save_summary(summary_path, vars(args), history, best)
            curves_path = os.path.join(save_path, "curves.png")
            experiment.plot_curves(curves_path, history)
            print_log(save_path, f"Updated summary and curves at epoch {epoch}")

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            result_path = save_path + '/epoch%d.pth' % epoch
            torch.save(state, result_path)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    criterion.train()

    # logger
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    space_fmt = str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)
    print_freq = len(train_loader)

    for i, (images, targets, infos) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        images = images.cuda()  # [B, T, 3, H, W]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        clean_boxes = torch.stack([t['boxes'] for t in targets])
        boxes = apply_box_noise(clean_boxes, infos, args, phase='train')
        dummy_mask = torch.stack([t['actions'] == args.num_class + 1 for t in targets]).squeeze()
        
        mae_feats = None
        if args.use_mae and 'mae_feats' in targets[0]:
             mae_feats = torch.stack([t['mae_feats'] for t in targets])

        num_batch = images.shape[0]
        num_frame = images.shape[1]

        # compute output
        outputs = model(images, boxes, dummy_mask, mae_feats)

        loss_dict = criterion(outputs, targets, log=False)
        weight_dict = criterion.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        
        # 关键修复：强制将所有 Tensor 转换为 Python float，彻底断开计算图引用
        # 这可以防止 metric_logger 持有计算图导致的内存泄漏
        loss_dict_reduced_unscaled = {f'{k}_unscaled': (v.item() if isinstance(v, torch.Tensor) else v)
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: ((v * weight_dict[k]).item() if isinstance(v, torch.Tensor) else (v * weight_dict[k]))
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled  # 已经是 float 了

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        # 确保传入 logger 的全是 float，不持有任何 Tensor 引用
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        gce = loss_dict_reduced['group_class_error']
        metric_logger.update(group_class_error=(gce.item() if isinstance(gce, torch.Tensor) else gce))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 显式删除大对象，帮助 GC 回收
        del images, targets, boxes, clean_boxes, outputs, loss, loss_dict

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(test_loader, model, criterion, metrics, epoch):
    model.eval()
    criterion.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    metric_logger.add_meter('group_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Evaluation Inference: '

    print_freq = len(test_loader)
    name_to_vid = {name: i + 1 for i, name in enumerate(SEQS_CAFE)}
    file_path = path + '/pred_group_epoch_%d.txt' % epoch

    for i, (images, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        images = images.cuda()  # [B, T, 3, H, W]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        clean_boxes = torch.stack([t['boxes'] for t in targets])
        boxes = apply_box_noise(clean_boxes, infos, args, phase='infer')
        dummy_mask = torch.stack([t['actions'] == args.num_class + 1 for t in targets]).squeeze()
        
        mae_feats = None
        if args.use_mae and 'mae_feats' in targets[0]:
             mae_feats = torch.stack([t['mae_feats'] for t in targets])

        # compute output
        outputs = model(images, boxes, dummy_mask, mae_feats)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        metric_logger.update(group_class_error=loss_dict_reduced['group_class_error'])

        # Keep original coordinates for evaluation alignment.
        make_txt(clean_boxes, infos, outputs, name_to_vid, file_path)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    detections = open(file_path, "r")
    result = metrics.evaluate(detections)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, result


def make_txt(boxes, infos, outputs, name_to_vid, file_path):
    for b in range(boxes.shape[0]):
        for t in range(boxes.shape[1]):
            # 使用评估基准尺寸（与 gt_tracks.txt 一致），而不是模型输入尺寸
            image_w, image_h = args.eval_image_width, args.eval_image_height

            pred_group_actions = outputs['pred_activities'][b]
            pred_group_actions = F.softmax(pred_group_actions, dim=1)
            members = outputs['membership'][b]

            pred_membership = torch.argmax(members.transpose(0, 1), dim=1).detach().cpu()
            keep_membership = members.transpose(0, 1).max(-1).values > args.group_threshold
            pred_group_action = torch.argmax(pred_group_actions, dim=1).detach().cpu()

            for box_idx in range(boxes.shape[2]):
                x, y, w, h = boxes[b][t][box_idx]
                x1, y1, x2, y2 = (x - w / 2) * image_w, (y - h / 2) * image_h, (x + w / 2) * image_w, (
                            y + h / 2) * image_h

                pred_group_id = pred_membership[box_idx]
                pred_group_action_idx = pred_group_action[pred_group_id]
                pred_group_action_prob = pred_group_actions[pred_group_id][pred_group_action_idx]

                if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                    if pred_group_action_idx != (pred_group_actions.shape[-1] - 1):
                        if bool(keep_membership[box_idx]) is False:
                            pred_group_id = -1
                            pred_group_action_idx = args.num_class

                    pred_list = [name_to_vid[infos[b]['vid']], infos[b]['sid'], infos[b]['fid'][t],
                                 int(x1), int(y1), int(x2), int(y2),
                                 int(pred_group_id), int(pred_group_action_idx) + 1,
                                 float(pred_group_action_prob)]
                    str_to_be_added = [str(k) for k in pred_list]
                    str_to_be_added = (" ".join(str_to_be_added))

                    f = open(file_path, "a+")
                    f.write(str_to_be_added + "\r\n")
                    f.close()


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack([image for image in batch[0]])
    return tuple(batch)


if __name__ == '__main__':
    main()
