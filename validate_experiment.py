"""Independent common-protocol evaluation of any recorded/legacy HOOD checkpoint."""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from train_baseline import main
from utils.defaults import DEFAULTS


if __name__ == '__main__':
    cli = OmegaConf.from_cli()
    if not cli.get('checkpoint'):
        raise ValueError('checkpoint=/absolute/path/model.pth is required')
    state = torch.load(cli.checkpoint, map_location='cpu')
    stored = OmegaConf.create(state.get('config', {}))
    if not stored.get('config') and not cli.get('config'):
        raise ValueError('A checkpoint without config needs config=baseline/postcvpr_seed57 or its original config')
    protocol = Path(cli.get('protocol', Path(DEFAULTS.project_dir) / 'configs/evaluation/common_physics.yaml'))
    options = OmegaConf.merge(stored, {'mode': 'validate', 'step_start': 0,
                                     'recording': {'require_clean': True, 'tensorboard': True},
                                     'experiment': {'name': 'common_evaluation', 'checkpoint_path': None},
                                     'validation': OmegaConf.load(protocol)['validation']}, cli)
    options['mode'] = 'validate'
    for key in ('run_dir', 'resume', 'init_checkpoint'):
        options.pop(key, None)
    main(options)
