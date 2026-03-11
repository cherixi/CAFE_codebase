# Box Noise Ablation Changes (support/gadtr-raw)

This document describes the box-noise ablation implementation on branch `support/gadtr-raw`.

## Goal

Add deterministic box-noise perturbation with two policies:

1. `train_and_infer`
2. `infer_only`

And keep evaluation coordinates aligned with existing protocol by exporting clean boxes.

## Modified Files

1. `util/box_noise.py` (new)
2. `train.py`
3. `test.py`

## 1) `util/box_noise.py`

Added:

- `_stable_seed(base_seed, vid, sid, fids)`
- `should_apply_box_noise(args, phase)`
- `_sample_noise(shape, seed, device)`
- `apply_box_noise(batch_boxes, infos, args, phase)`

Input format:

- `batch_boxes`: `[B, T, N, 4]`, normalized `(cx, cy, w, h)`.

Noise composition:

- center offset: relative to `w/h`
- scale perturbation: multiplicative in log-space
- aspect perturbation: multiplicative in log-space

Determinism:

- seed is derived from `box_noise_seed + vid + sid + fid-list`.

Bounds:

- invalid/dummy boxes unchanged (`w<=1e-6 or h<=1e-6`)
- `w/h` clamped to `[box_noise_min_size, box_noise_max_size]`
- `cx/cy` clamped so boxes stay in normalized range

## 2) `train.py` changes

Imported `apply_box_noise`.

Added CLI args:

- `--box_noise_policy {none,infer_only,train_and_infer}`
- `--box_noise_seed`
- `--box_noise_center_std`
- `--box_noise_scale_std`
- `--box_noise_aspect_std`
- `--box_noise_min_size`
- `--box_noise_max_size`

Forward paths:

- train step:
  - `clean_boxes = torch.stack([t['boxes'] for t in targets])`
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='train')`
- validation step:
  - same, with `phase='infer'`
  - `make_txt(clean_boxes, ...)` for evaluation alignment

## 3) `test.py` changes

Imported `apply_box_noise`.

Added same 7 CLI args as `train.py`.

Inference path:

- `clean_boxes = torch.stack([t['boxes'] for t in targets])`
- `boxes = apply_box_noise(clean_boxes, infos, args, phase='infer')`
- `make_txt(clean_boxes, ...)` to keep clean evaluation coordinates

## Defaults

- `box_noise_center_std = 0.10`
- `box_noise_scale_std = 0.08`
- `box_noise_aspect_std = 0.08`
- `box_noise_min_size = 1e-4`
- `box_noise_max_size = 1.0`
- `box_noise_seed = 1`

## Usage

### A) Inference-only noise with existing checkpoint

```bash
python test.py \
  --model_path /path/to/epochXX.pth \
  --box_noise_policy infer_only \
  --box_noise_seed 1 \
  --box_noise_center_std 0.10 \
  --box_noise_scale_std 0.08 \
  --box_noise_aspect_std 0.08
```

### B) Train + infer noise

```bash
python train.py \
  --box_noise_policy train_and_infer \
  --box_noise_seed 1 \
  --box_noise_center_std 0.10 \
  --box_noise_scale_std 0.08 \
  --box_noise_aspect_std 0.08
```

### C) Baseline

```bash
python test.py --box_noise_policy none
```

## Notes

- `infer_only` also affects validation in `train.py` because validation calls `phase='infer'`.
- Noise determinism depends on sample identity (`vid/sid/fids`). If sampled `fids` change, noise changes accordingly.
