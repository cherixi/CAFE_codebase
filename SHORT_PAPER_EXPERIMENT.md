# Lightweight Static-Dynamic GAD Experiment

This branch starts from the model-code parent of the best archived mAP@1.0 run
(`59eb232`) but intentionally removes the major research modules from the short-paper
configuration.

## Active Pipeline

```text
DINOv2 actor features
-> Kim-style group decoder
-> static VideoMAE context fusion
-> generic temporal encoder
-> Static-Dynamic Temporal Pooling (SDTP)
-> original activity and membership heads
```

The training command explicitly disables:

- frame-level STIR/HOI graph (`--hoi_mode none`)
- adaptive two-branch LRCC (`--mae_fusion static_pool`)
- OLIC (`--no_olic`)
- PMR (`--no_pairwise_refiner`)
- all Attach, Margin, qPMR, and IA-STIR descendants, which are absent from this base commit

SDTP keeps the original learned temporal pooling as its static path and adds a bounded
first-order feature-difference residual. After the first actor+group experiment showed
group-gate saturation, the safe default applies the dynamic residual to actor tokens only.
Its scale is initialized to 0.02 and constrained below 0.1.

Label smoothing is an optional training trick and is not part of SDTP.

## Controlled Runs

```bash
bash scripts/run_short_paper_sdtp.sh sdtp "0,1,2,3"
bash scripts/run_short_paper_sdtp.sh ablation "0,1,2,3"
```

The two commands differ only in `--use_sdtp` versus `--no_sdtp`. Both use the same
backbone, static context fusion, temporal encoder, label smoothing, split, and seed.

Run the focused check before training:

```bash
python scripts/check_sdtp_smoke.py
```

For parallel screening, `scripts/run_agile_experiment.py` provides tagged variants and
historical-curve early stopping. It evaluates every epoch and stops only clearly collapsed
or non-viable runs; the target is Practical GAD on split-by-place (10.85 / 30.90 / 63.84).
