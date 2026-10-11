"""Batch-boundary reproducible sampling and checkpoint state for recorded runs."""
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from utils.common import save_checkpoint


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same number of visible CUDA devices')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class ResumableSampler(Sampler):
    """Mirror default RandomSampler's global seed draw and randperm exactly.

    Cursor advances only after a fully completed batch, never at prefetch/yield.
    Recorded training explicitly requires batch_size=1 and num_workers=0.
    """
    def __init__(self, length, state=None):
        self.length, self.order, self.cursor, self.epoch = length, None, 0, 0
        if state:
            if state['length'] != length:
                raise ValueError('Dataset length changed since checkpoint')
            self.order, self.cursor, self.epoch = state['order'], state['cursor'], state['epoch']
            if not 0 <= self.cursor <= self.length or self.order is None or len(self.order) != length:
                raise ValueError('Invalid checkpoint sampler state')

    def __iter__(self):
        if self.order is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
            self.order = torch.randperm(self.length, generator=generator).tolist()
        return iter(self.order[self.cursor:])

    def __len__(self):
        return self.length - self.cursor

    def commit(self):
        if self.order is None or self.cursor >= self.length:
            raise RuntimeError('No current sample to commit')
        self.cursor += 1

    def next_epoch(self):
        if self.cursor != self.length:
            raise RuntimeError('Cannot skip unfinished epoch')
        self.epoch += 1
        self.order, self.cursor = None, 0

    def state_dict(self):
        return {'length': self.length, 'order': self.order, 'cursor': self.cursor, 'epoch': self.epoch}


def loader_iterator(dataset, sampler, restoring=False):
    loader = DataLoader(dataset, batch_size=1, num_workers=0, sampler=sampler)
    if not restoring:
        return iter(loader)
    # DataLoader iterator construction consumes a base seed even without workers.
    # It is unused here. Do not consume it a second time halfway through an epoch.
    state = torch.get_rng_state()
    iterator = iter(loader)
    torch.set_rng_state(state)
    return iterator


def move_tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, defaultdict):
        return defaultdict(value.default_factory, {key: move_tree(item, device) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    return copy.deepcopy(value)


def dataset_cache(dataset):
    garments = dataset.loader.garment_builder.garments_dict
    return {name: copy.deepcopy(g.get('coarse_edges', {})) for name, g in garments.items()}


def restore_dataset_cache(dataset, cache):
    garments = dataset.loader.garment_builder.garments_dict
    for name, edges in cache.items():
        garments[name]['coarse_edges'] = copy.deepcopy(edges)


def data_fingerprint(assets, manifest):
    record = {'assets': {key: value['sha256'] for key, value in assets.items()},
              'sequences': [(s['id'], s['sha256']) for s in manifest['sequences']]}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def save_recorded_checkpoint(runner, aux, cfg, run, sampler, dataset, progress, fingerprint):
    destination = run.root / 'checkpoints' / f'step_{progress["global_step"]:010d}.pth'
    if destination.exists():
        return destination
    temporary = destination.with_suffix('.pth.tmp')
    save_checkpoint(runner, aux, cfg, str(temporary))
    state = torch.load(temporary, map_location='cpu')
    state.update(checkpoint_schema=2, **progress, seed=int(cfg.seed),
                 git_commit=run.metadata['git_commit'], source_run=str(run.root),
                 data_fingerprint=fingerprint, sampler=sampler.state_dict(), rng_state=rng_state(),
                 cloth_cache=move_tree(runner.cloth_obj.cache, 'cpu'),
                 dataset_coarse_cache=dataset_cache(dataset))
    torch.save(state, temporary)
    temporary.replace(destination)
    return destination


def restore_training(state, runner, aux, dataset, fingerprint, commit, allow_code_change=False):
    if state.get('checkpoint_schema') != 2:
        raise ValueError('This checkpoint lacks RNG/sampler/cache state. It can be evaluated, '
                         'but cannot be exactly resumed. Use init_checkpoint for a separately labelled fine-tune.')
    if state['data_fingerprint'] != fingerprint:
        raise ValueError('Training data/assets differ from checkpoint')
    if state['git_commit'] != commit and not allow_code_change:
        raise ValueError('Code commit differs. Check out the checkpoint commit, or explicitly set '
                         'resume_allow_code_change=true (not exact reproducibility).')
    runner.load_state_dict(state['training_module'])
    for key in ('optimizer', 'scheduler'):
        aux[key].load_state_dict(state[key])
    runner.cloth_obj.cache = move_tree(state['cloth_cache'], cfg_device(runner))
    restore_dataset_cache(dataset, state['dataset_coarse_cache'])


def cfg_device(runner):
    return runner.mcfg.device
