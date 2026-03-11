# Box Noise Ablation Changes (feat/dinov2-finetune)

This document records the deterministic box-noise ablation implementation and train/test alignment changes.

## Goals

1. Add deterministic box noise with two policies:
- `train_and_infer`
- `infer_only`

2. Keep evaluation coordinates unchanged:
- model input can use noisy boxes,
- `pred txt` export must use original clean boxes.

3. Align `test.py` to `train.py` for non-train/infer logic:
- same key model args,
- same distributed sampler behavior,
- same strict checkpoint load strategy,
- same MAE forward path.

---

## Modified Files

1. `util/box_noise.py` (new)
2. `train.py`
3. `test.py`

---

## 1) New utility: `util/box_noise.py`

Added functions:

- `_stable_seed(base_seed, vid, sid, fids)`
- `should_apply_box_noise(args, phase)`
- `_sample_noise(shape, seed, device)`
- `apply_box_noise(batch_boxes, infos, args, phase)`

### Input/Output

- Input boxes shape: `[B, T, N, 4]`, normalized `(cx, cy, w, h)`.
- Output: same shape/type with deterministic perturbation applied.

### Policy

- `none`: no perturbation
- `infer_only`: perturb only when `phase == 'infer'`
- `train_and_infer`: perturb when `phase in {'train', 'infer'}`

### Noise composition

- Center offsets scaled by box size (`w/h`) using Gaussian noise.
- Multiplicative size and aspect-ratio perturbations in log-space.
- Invalid/dummy boxes (`w<=1e-6` or `h<=1e-6`) remain unchanged.
- `w/h` clamped to `[box_noise_min_size, box_noise_max_size]`.
- `cx/cy` clamped so full box remains in normalized image bounds.

### Determinism

Seed hash key uses:

- `box_noise_seed`
- `vid`, `sid`
- sampled `fid` list

So the same sample identity receives the same perturbation.

---

## 2) Changes in `train.py`

### New CLI args

- `--box_noise_policy {none,infer_only,train_and_infer}`
- `--box_noise_seed`
- `--box_noise_center_std`
- `--box_noise_scale_std`
- `--box_noise_aspect_std`
- `--box_noise_min_size`
- `--box_noise_max_size`

### Forward paths

- Train step:
  - `clean_boxes = torch.stack([t['boxes'] for t in targets])`
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='train')`

- Validation step:
  - `clean_boxes = torch.stack([t['boxes'] for t in targets])`
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='infer')`
  - `make_txt(clean_boxes, infos, outputs, ...)` for eval alignment

---

## 3) Changes in `test.py`

### Added/Aligned args to match train-side model config

- MAE-related:
  - `--no_mae`
  - `--mae_version`
  - `--videomae_feats_path`
- Backbone-related:
  - `--unfreeze_blocks`
- HOI/Temporal-related:
  - `--hoi_nheads`
  - `--hoi_topk`
  - `--temporal_layers`
  - `--tcn_kernel_size`
  - `--tcn_dropout`
- Box-noise args (same 7 as train):
  - `--box_noise_policy`
  - `--box_noise_seed`
  - `--box_noise_center_std`
  - `--box_noise_scale_std`
  - `--box_noise_aspect_std`
  - `--box_noise_min_size`
  - `--box_noise_max_size`

### Inference path updates

- `args.use_mae` / `args.mae_dim` computed same way as train.
- Distributed sampler logic aligned:
  - `DistributedSampler` when `--distributed`,
  - otherwise `RandomSampler`.
- DataLoader config aligned to train validation:
  - `num_workers=2`
  - `pin_memory=False`
  - `persistent_workers=True`
  - `prefetch_factor=2`
- Checkpoint loading aligned (strict full state_dict):
  - `checkpoint = torch.load(args.model_path)`
  - `model.load_state_dict(checkpoint['state_dict'])`
- Forward uses MAE features when enabled:
  - `outputs = model(images, boxes, dummy_mask, mae_feats)`
- Box-noise infer path:
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='infer')`
- Export txt uses clean boxes:
  - `make_txt(clean_boxes, infos, outputs, ...)`

---

## Default Noise Strengths

- `box_noise_center_std = 0.10`
- `box_noise_scale_std = 0.08`
- `box_noise_aspect_std = 0.08`
- `box_noise_min_size = 1e-4`
- `box_noise_max_size = 1.0`
- `box_noise_seed = 1`

---

## Usage Examples

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

---

## Notes

1. `infer_only` also affects validation in `train.py` (`phase='infer'`).
2. Noise reproducibility depends on sample identity (`vid/sid/fids`); if sampled frame ids differ, perturbation differs.
