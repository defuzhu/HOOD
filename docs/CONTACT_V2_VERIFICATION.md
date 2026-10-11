# Verification record (2026-10-11)

Implementation commit: `67d81ba` (TensorBoard support: `7153f0a`).
Environment: `hood_clean`; real CUDA access on an RTX 3090 Ti.

## Results

- All 18 relevant tests passed with CUDA enabled, including CPU/CUDA distance
  and gradient parity, unchanged legacy criterion, relaxed potential derivatives,
  area subdivision, weight ramp, and CSV/live TensorBoard equivalence.
- Real-data smoke run completed three optimizer steps with `warmup_steps=0`,
  plus three fixed validation sequences at steps 0 and 3 (three frames each).
  This deliberately shortened run tests execution; it is not a baseline-quality
  experiment or evidence of long-run stability.
- Contact losses for the three sampled training batches were approximately
  0.0580, 17.9468 and 1.8470; pre-clipping gradient norms were 0.000370, 1303.51
  and 381.18. They were finite. Different batches/materials are not a loss trend.
- The checkpoint reloaded successfully. Maximum position differences versus
  the saved step-3 rollouts were 5.96e-8, 8.94e-8 and 8.94e-8 m, within the
  1e-6 m comparison tolerance. CUDA scatter operations were not configured for
  bitwise reproducibility; exact equality is not claimed.
- Live total/loss, learning-rate and fixed-validation scalar records were
  verified in the smoke run's TensorBoard events.

Run directories under `$HOOD_DATA/experiments`:

```
smoke_relaxed_barrier_v2_seed57_20261011T052648Z_d0736119
smoke_relaxed_barrier_v2_reload_seed57_20261011T052805Z_cb882bee
```

The previous claim that this server had no usable GPU was caused by sandbox
device visibility. Controlled host access finds two RTX 3090 Ti GPUs with working
CUDA. Unit checks inside the restricted sandbox can still skip CUDA.

## Baseline TensorBoard conversion

Source run: `baseline_postcvpr_seed57_20261006T081735Z_094bb347`.
Its recorded training commit is `452769b`, not its final directory suffix.

- Converted 60,000 training rows, 250,004 rollout-substep rows and 39 validation
  rows into 193 scalar tags. The latter are three sequences at 13 checkpoints.
- Verified 60,000 total-loss points (steps 1..60000), 59,901 actual learning-rate
  points (optimizer enabled at step 100), and 13 aggregate validation points
  (steps 0..60000).
- Checked final scalar values against CSV. Original CSV/checkpoints/config were
  preserved; `tensorboard/csv_import.json` records source CSV checksums.
- TensorBoard loaded the run and served HTTP 200 at localhost port 6006.

No full V2 training or statistical baseline comparison has been completed.
Use the controlled comparison protocol in `CONTACT_POTENTIAL_V2.md` before
claiming superiority or noninferiority.
