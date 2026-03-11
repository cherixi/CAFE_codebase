# Box Noise Ablation Changes (Verified Against Current Code)

This document is a checked, code-aligned specification of the box-noise ablation feature.
It covers what was implemented, where it is implemented, and what caveats matter for reproducibility.

---

## 1. Scope and Goal

Implemented two ablation modes for box perturbation:

1. `train_and_infer`
2. `infer_only`

with these constraints:

- Mixed noise sources: center shift + scale + aspect-ratio change.
- Noise in normalized box space (`cx, cy, w, h`), not absolute pixels.
- Noisy boxes are clamped to valid image bounds.
- Deterministic noise sampling under fixed seed + fixed sample identity.
- Evaluation text export uses clean/original boxes for coordinate alignment with existing evaluator.

Status: **implemented and verified in code**.

---

## 2. Modified Files

1. [util/box_noise.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/util/box_noise.py) (new)
2. [train.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/train.py)
3. [test.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/test.py)

---

## 3. Implementation Details

## 3.1 `util/box_noise.py`

Implemented functions:

- `_stable_seed(base_seed, vid, sid, fids)`
- `should_apply_box_noise(args, phase)`
- `_sample_noise(shape, seed, device)`
- `apply_box_noise(batch_boxes, infos, args, phase)`

### Policy behavior

`should_apply_box_noise(args, phase)`:

- `none`: never apply
- `infer_only`: apply only when `phase == 'infer'`
- `train_and_infer`: apply when `phase in {'train', 'infer'}`

### Noise equations

Input boxes: `[B, T, N, 4]` as normalized `(cx, cy, w, h)`.

For each sample `b`:

- `eps_center ~ N(0,1)` shape `[T, N, 2]`
- `eps_scale ~ N(0,1)` shape `[T, N]`
- `eps_aspect ~ N(0,1)` shape `[T, N]`

Then:

- `cx' = cx + eps_center_x * center_std * w`
- `cy' = cy + eps_center_y * center_std * h`
- `size_scale = exp(eps_scale * scale_std)`
- `ratio_scale = exp(eps_aspect * aspect_std)`
- `w' = w * size_scale * ratio_scale`
- `h' = h * size_scale / ratio_scale`

### Validity and clamping

- Invalid/dummy boxes (`w <= 1e-6` or `h <= 1e-6`) are unchanged.
- `w', h'` are clamped to `[box_noise_min_size, box_noise_max_size]`.
- `cx', cy'` are clamped so full box remains in `[0,1]`.

### Determinism

Per-sample seed is generated from:

- `box_noise_seed` (fallback to `random_seed` if missing),
- `vid`, `sid`, and full `fid` list from `infos`.

So if sample identity (`vid/sid/fids`) is unchanged, noise is identical across runs.

---

## 3.2 `train.py` integration

### New CLI args (7)

- `--box_noise_policy`
- `--box_noise_seed`
- `--box_noise_center_std`
- `--box_noise_scale_std`
- `--box_noise_aspect_std`
- `--box_noise_min_size`
- `--box_noise_max_size`

### Train forward path

- Build clean boxes from targets:
  - `clean_boxes = torch.stack([t['boxes'] for t in targets])`
- Model input uses:
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='train')`

### Validation-in-train path

- Model input uses:
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='infer')`
- Export for evaluation uses:
  - `make_txt(clean_boxes, infos, outputs, ...)`

This preserves coordinate compatibility with the original evaluator.

---

## 3.3 `test.py` integration

### New CLI args (same 7)

- `--box_noise_policy`
- `--box_noise_seed`
- `--box_noise_center_std`
- `--box_noise_scale_std`
- `--box_noise_aspect_std`
- `--box_noise_min_size`
- `--box_noise_max_size`

### Inference path

- Build `clean_boxes` from targets.
- Feed model with:
  - `boxes = apply_box_noise(clean_boxes, infos, args, phase='infer')`
- Write eval text with:
  - `make_txt(clean_boxes, infos, outputs, ...)`

---

## 4. Default Noise Strengths

- `box_noise_center_std = 0.10`
- `box_noise_scale_std = 0.08`
- `box_noise_aspect_std = 0.08`
- `box_noise_min_size = 1e-4`
- `box_noise_max_size = 1.0`

---

## 5. Verified Consistency Check (Current Codebase)

Checked against current code:

- [train.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/train.py)
- [test.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/test.py)
- [util/box_noise.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/util/box_noise.py)

Confirmed:

1. All 7 `box_noise_*` args exist in both `train.py` and `test.py`.
2. `apply_box_noise(..., phase='train')` used in training forward.
3. `apply_box_noise(..., phase='infer')` used in validation/test forward.
4. `make_txt(clean_boxes, ...)` is used in train validation and test inference.
5. Evaluation coordinates are not replaced by noisy coordinates.

---

## 6. Important Caveats (for reproducibility)

1. Deterministic noise depends on `fids`.
- Seed key includes `fid` list.
- If frame sampling changes, noise will also change.

2. In this dataset, test-time frame selection is still random-sampled by dataset logic.
- If you require fully fixed noise + fixed frames across runs, frame sampling must also be fixed.

3. `infer_only` policy also applies to `train.py` validation path.
- Because validation calls `phase='infer'`.
- This is usually desired for evaluating infer-time robustness.

---

## 7. Usage Recipes

## A. Inference-only noise on existing checkpoint

```bash
python test.py \
  --model_path /path/to/epochXX.pth \
  --box_noise_policy infer_only \
  --box_noise_seed 1 \
  --box_noise_center_std 0.10 \
  --box_noise_scale_std 0.08 \
  --box_noise_aspect_std 0.08
```

## B. Train + infer noise

```bash
python train.py \
  --box_noise_policy train_and_infer \
  --box_noise_seed 1 \
  --box_noise_center_std 0.10 \
  --box_noise_scale_std 0.08 \
  --box_noise_aspect_std 0.08
```

## C. Baseline without noise

```bash
python test.py --box_noise_policy none
```

---

## 8. Porting Checklist for Other Branches

1. Copy [box_noise.py](/c:/Users/xiaic/Desktop/CODE/CAFE_codebase/util/box_noise.py).
2. Add the 7 CLI args in both train/test entry files.
3. Keep both tensors in forward path:
- `clean_boxes` for export/eval.
- `boxes = apply_box_noise(clean_boxes, infos, args, phase=...)` for model input.
4. Ensure `make_txt` uses `clean_boxes`.
5. Validate with a quick sanity run:
- `--box_noise_policy none` vs `infer_only` and compare metric drop.

