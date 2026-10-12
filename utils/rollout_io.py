"""Portable, configurable names for saved garment trajectories."""
import re
from pathlib import Path

DEFAULT_ROLLOUT_TEMPLATE = '{experiment}_seed{seed}_{sequence}_{garment}_rollout.npz'


def safe_name(value):
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('.')
    if not name:
        raise ValueError('Empty output name')
    return name


def rollout_filename(experiment, seed, sequence, garment, template=DEFAULT_ROLLOUT_TEMPLATE):
    name = template.format(experiment=safe_name(experiment), seed=int(seed),
                           sequence=safe_name(sequence), garment=safe_name(garment))
    if Path(name).name != name or '/' in name or '\\' in name or not name.endswith('.npz'):
        raise ValueError('rollout_name_template must produce one .npz filename, without directories')
    return name
