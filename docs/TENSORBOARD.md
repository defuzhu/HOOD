# Recorded training curves

`train_baseline.py` writes both CSV and TensorBoard by default. The YAML setting
`recording.tensorboard: true` makes this explicit. Install the dependency in the
active environment with `python -m pip install tensorboard==2.20.0` (also in
`hood.yml`). Each run writes its own `tensorboard/` directory.

Historical completed runs can be imported with:

```bash
python csv_to_tensorboard.py /absolute/path/to/experiment
```

The converter refuses to append to an existing output. Use `--output` with a new
directory if a second import is needed. It preserves CSV steps, includes source
hashes and original training commit, and does not rewrite CSV/checkpoints/config.
Event wall times are import times; use the Step axis, not Relative time, for
historical imports. An interrupted conversion can leave partial output: retry
to a new directory, do not use an incomplete import as a full training record.

```bash
tensorboard --logdir "$HOOD_DATA/experiments" --host 127.0.0.1 --port 6006
```

For a remote server, forward port 6006 through SSH from the viewing computer,
then open http://127.0.0.1:6006 locally. A single log root includes future runs.

- `train/*_loss_mean`: total and each physical term, averaged across the current
  outer step's rollout substeps. `*_last` is its last substep, in separate charts.
- `optimizer/lr_used_mean`, `lr_used_last`: actual learning rates used by Adam.
  Warmup has no optimizer update, so it has no fabricated `lr_used` value.
- `optimizer/lr_next_*`: scheduled next rate, including warmup; also logs the
  collision weight and gradient norms before clipping.
- `validation/<sequence>__<garment>/*`: independent held-out sequence metrics.
  `validation/sequence_mean/*` is an equal-sequence mean, not frame weighting or
  a global maximum. `*_regular_mean` excludes the initialization frame.

The x-axis is the outer `global_step`, not the count of optimizer updates.
Separate tags generate separate charts; selecting multiple runs overlays the
same metric for ablations. Raw CSV retains every rollout substep.

New recorded runs also include `train_state/optimizer_step_count`, actual updates
in the batch, synchronized batch time, and `contact/*` raw loss/geometry/barrier
diagnostics. See [EXPERIMENT_SYSTEM.md](EXPERIMENT_SYSTEM.md) for complete fields
and comparisons aligned by actual optimizer updates.

Baseline penetration metrics named `approx_*` use the historical plane
approximation. Never compare these numerically against the new `mesh_*` metric
as though they were the same measurement. Changing contact potential also makes
raw total/contact losses across variants non-equivalent quality measures.
