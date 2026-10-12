"""Validate external standard-SMPL inputs without altering their source files."""
import pickle
from pathlib import Path

import numpy as np


def read_parameters(path):
    """PKL must be trusted; NPZ/NPY are loaded without object deserialization."""
    path = Path(path).expanduser()
    if path.suffix.lower() == '.pkl':
        with path.open('rb') as stream:
            return pickle.load(stream)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    raise ValueError('Use a .pkl dictionary, numeric .npz, or numeric .npy array')


def numeric(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError('SMPL parameters contain NaN or Inf')
    return result


def normalize_smpl_sequence(parameters, betas_override=None, start=0, stop=None, stride=1):
    """Accept split axis-angle SMPL fields or full pose [T,72]/[T,24,3].

    One constant ten-dimensional body shape is supported per rollout. No
    implicit SMPL-X conversion, coordinate transform or FPS resampling occurs.
    Returns a fresh standard dictionary, source-frame indices and assumptions.
    """
    data = dict(parameters) if isinstance(parameters, dict) else {'poses': parameters}
    notes = []
    if 'body_pose' in data:
        body = numeric(data['body_pose'])
        if body.ndim == 3 and body.shape[1:] == (23, 3):
            body = body.reshape(len(body), 69)
        if body.ndim != 2 or body.shape[1] != 69:
            raise ValueError('body_pose must be [T,69] or [T,23,3] standard-SMPL axis angles')
        root = numeric(data['global_orient']) if 'global_orient' in data else np.zeros((len(body), 3), np.float32)
        if 'global_orient' not in data:
            notes.append('Missing global_orient: using zero root rotation')
    else:
        key = 'poses' if 'poses' in data else 'pose'
        if key not in data:
            raise ValueError('Missing body_pose or poses/pose')
        pose = numeric(data[key])
        if pose.ndim == 3 and pose.shape[1:] == (24, 3):
            pose = pose.reshape(len(pose), 72)
        if pose.ndim != 2 or pose.shape[1] != 72:
            raise ValueError('Full pose must be [T,72] or [T,24,3]; SMPL-X/SMPL-H/BVH need explicit conversion')
        root, body = pose[:, :3], pose[:, 3:]
    count = len(body)
    transl_key = 'transl' if 'transl' in data else 'trans'
    transl = numeric(data[transl_key]) if transl_key in data else np.zeros((count, 3), np.float32)
    if transl_key not in data:
        notes.append('Missing transl/trans: using zero global translation')
    for name, value in [('global_orient', root), ('transl', transl)]:
        if value.shape != (count, 3):
            raise ValueError(f'{name} must be [T,3], matching body_pose')
    beta = betas_override if betas_override is not None else data.get('betas', np.zeros(10, np.float32))
    if isinstance(beta, (str, Path)):
        beta = read_parameters(beta)
    if isinstance(beta, dict):
        if 'betas' not in beta:
            raise ValueError('Body parameter file must have a betas field')
        beta = beta['betas']
    beta = numeric(beta)
    if beta.shape == (1, 10):
        beta = beta[0]
    if beta.shape != (10,):
        raise ValueError('Use one constant body shape: betas [10] or [1,10]')
    if betas_override is None and 'betas' not in data:
        notes.append('Missing betas: using zero body shape')
    if not isinstance(start, int) or start < 0 or not isinstance(stride, int) or stride <= 0:
        raise ValueError('start must be a nonnegative integer; stride a positive integer')
    if stop is not None and (not isinstance(stop, int) or stop > count or stop <= start):
        raise ValueError('stop must satisfy start < stop <= source frame count')
    indices = np.arange(count)[slice(start, stop, stride)]
    if len(indices) < 3:
        raise ValueError('HOOD needs at least three selected SMPL frames (two context frames)')
    used = {'body_pose', 'global_orient', 'poses', 'pose', 'transl', 'trans', 'betas'}
    ignored = sorted(set(data) - used)
    if ignored:
        notes.append('Unused source fields: ' + ', '.join(ignored))
    return {'body_pose': body[indices].copy(), 'global_orient': root[indices].copy(),
            'transl': transl[indices].copy(), 'betas': beta.copy()}, indices, notes
