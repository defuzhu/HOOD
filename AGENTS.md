# Research workflow

- Keep `main` at the official HOOD baseline. Research changes belong to `my_method`
  (isolated work branches may be merged into it).
- Make one commit per independently explainable change. Keep original methods
  selectable; control new modules with YAML and use separate ablation configs.
- Every training run gets a unique experiment directory with commit, resolved
  config, seed, checkpoints, CSV logs and fixed validation results.
- All future recorded training must also write TensorBoard scalars for total
  loss, each physical loss, learning rate and fixed validation metrics. Keep
  CSV as the source record. Use identical tag names across baseline and variants.
- Do not overwrite experiments or present a changed loss value as evidence of
  better simulation. Compare common geometry metrics and rollout videos.
