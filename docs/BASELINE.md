# Recorded postcvpr baseline

The original `train.py`, YAML and all six physical criterion formulas remain available.
`train_baseline.py` adds recording and a fixed validation protocol. Runner observation is opt-in:
it does not change loss values, rollout detach, clipping, Adam or per-substep updates.

## Run

```bash
conda activate hood_clean
export HOOD_PROJECT=/home/zhudf/projects/HOOD
export HOOD_DATA=/home/zhudf/data/hood_data/hood_data
# CPU 0 avoided the interpreter crashes seen on this server. Not a portable requirement.
taskset -c 0 python -u train_baseline.py config=baseline/postcvpr_seed57
```

CLI overrides take precedence for this entrypoint, e.g. `seed=58 experiment.name=baseline_repeat`.
The base YAML uses seed 57, 60,000 batches, one graph per batch, original postcvpr physics,
workers=0 and cuda:0. Workers/device reflect server operation; they do not change the constitutive models.
Seeds are set BEFORE model/dataset construction. Historical train.py seeded AFTER construction,
so new training cannot reproduce the old initialization or recover historical loss curves.
No stronger claim of bitwise deterministic CUDA execution is made.

Each run gets an exclusive directory under `$HOOD_DATA/experiments/`:

```
baseline_postcvpr_seed57_<UTC timestamp>_<unique ID>/
  config.yaml                 # effective, fully merged training configuration
  metadata.json               # code commit, state, seed, runtime, hardware, provenance
  environment.txt             # installed package versions
  assets.json                 # split, SMPL and garment asset hashes
  training_data_manifest.json # hashes of every SMPL sequence used for training
  validation_config.yaml
  validation_manifest.json    # fixed data hashes, material, seed, motion IDs, frame limits
  checkpoints/step_*.pth       # weights, stats, optimizer, scheduler, step, epoch, commit
  logs/console.log
  logs/train_substeps.csv      # every rollout substep, all physical losses, LR and pre-clip norm
  logs/train.csv               # both mean across rollout substeps and last-substep losses
  logs/validation.csv          # each fixed sequence at each evaluation step
  validation/step_*/           # per-frame CSV, summary JSON, mesh trajectory NPZ
```

Default recording refuses dirty Git checkouts. `recording.require_clean=false` is for local
checks only and marks the run non-reproducible; tracked diffs and untracked path names are recorded.
Neither secrets nor the full environment variables are captured. Long runs should be launched
as an independent server service; completion/failure is reflected in metadata.

## Fixed validation

The three tshirt sequences end in CMU motion IDs `01_01`, `07_02`, `104_54`.
The entrypoint checks these MOTIONS do not occur in the training split, including other shapes/garments.
Original sequence betas are used, noise is zero, material is fixed, graph centers are reproducible.
120 predictions are evaluated at step 0, every 5,000 batches, and at the final step.
Validation uses a separate runner with copied weights/statistics under no_grad, preserving training
Python/NumPy/CPU/CUDA RNG states and training normalizer/cache state. No optimizer runs in validation.

Physical losses are reported per frame, with initialization separated from regular-frame means.
The original valid_rollout sets iter=0: validation collision weight is therefore always
weight_start (5,000), independent of the training ramp. This stays fixed across checkpoints.
Additional metrics: approximate penetration fraction/depth, principal strain/stretch and pinned error.
Penetration is the original nearest-face-center normal approximation: NOT exact SDF or CCD.
No VTO garment ground-truth MSE is reported. These three sequences are a reproducible small benchmark,
not a claim of broad validation across all garments or independently calibrated material accuracy.

## Evaluate a saved checkpoint and view results

```bash
taskset -c 0 python train_baseline.py config=baseline/postcvpr_seed57 mode=validate \
  experiment.name=legacy_baseline_eval checkpoint=/absolute/path/step_0000060000.pth
python plot_baseline.py /absolute/path/to/run
python render_baseline.py /absolute/path/to/run/validation/step_0000060000/SEQUENCE_tshirt_rollout.npz
```

Legacy checkpoints are marked as missing training-commit provenance. Their filenames supply the step;
evaluation does not fabricate training history. A true fixed comparison uses the same manifest, geometry,
material, seed, metrics and compute budget across methods. Total loss alone is not comparable after changing
energy models. Track each physical term and geometry plus rollout video.

Checkpoints include optimizer/scheduler state, but exact mid-epoch resumption is deliberately not implemented:
sampler/worker RNG state is absent. The new trainer refuses resume-looking training settings rather than
silently restarting a warm-start run. The original train.py remains available for legacy usage.

## Short integration check

```bash
taskset -c 0 python train_baseline.py config=baseline/postcvpr_seed57 \
  experiment.name=baseline_smoke experiment.max_iter=102 validation.n_steps=4 validation.every=100
```

This exercises warmup, actual backward/Adam, checkpoints and isolated fixed validation without a full training run.
Run `python -m unittest discover -s tests` for recording/RNG checks.
Set `HOOD_INTEGRATION=1` to also check exact CPU training equivalence with/without
logging and repeated CUDA fixed validation without mutating training state.
