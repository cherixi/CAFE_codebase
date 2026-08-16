#!/usr/bin/env bash
set -euo pipefail

variant="${1:-sdtp}"
devices="${2:-0,1,2,3}"
data_path="${CAFE_DATA_PATH:-/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/}"
feature_path="${CAFE_VIDEOMAE_PATH:-./videomae_features_giant}"

common_args=(
  --split place
  --data_path "$data_path"
  --backbone dinov2_vitb14
  --unfreeze_blocks 2
  --frozen_batch_norm
  --batch 16
  --num_frame 8
  --test_batch 52
  --device "$devices"
  --videomae_feats_path "$feature_path"
  --mae_fusion static_pool
  --hoi_mode none
  --no_olic
  --no_pairwise_refiner
  --temporal_agg_mode learned_pool
  --label_smoothing 0.05
  --skip_test_epochs 5
  --random_seed 1
)

case "$variant" in
  sdtp)
    extra_args=(
      --use_sdtp
      --sdtp_scope actor
      --sdtp_hidden_dim 64
      --sdtp_dynamic_scale_init 0.02
      --sdtp_dynamic_scale_max 0.1
    )
    ;;
  ablation)
    extra_args=(--no_sdtp)
    ;;
  *)
    echo "Usage: $0 {sdtp|ablation} [gpu_ids]" >&2
    exit 2
    ;;
esac

exec python train.py "${common_args[@]}" "${extra_args[@]}"
