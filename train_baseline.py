"""Recorded postcvpr training, leaving train.py and all physics formulas intact."""
import contextlib
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.arguments import create_modules, load_params
from utils.common import add_field_to_pyg_batch, move2device, save_checkpoint
from utils.defaults import DEFAULTS
from utils.experiment import Experiment, file_record, seed_all, write_json
from utils.fixed_validation import FixedValidation


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def checkpoint(runner, aux, cfg, experiment, step, epoch):
    path = experiment.root / 'checkpoints' / f'step_{step:010d}.pth'
    save_checkpoint(runner, aux, cfg, str(path))
    state = torch.load(path, map_location=cfg.device)
    state.update(global_step=step, epoch=epoch, seed=int(cfg.seed),
                 git_commit=experiment.metadata['git_commit'])
    # Epoch sampler/worker state is not saved: this is NOT exact mid-epoch resumption.
    torch.save(state, path)
    return path


def main():
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    cli = OmegaConf.from_cli()
    config_name = cli.get('config', 'baseline/postcvpr_seed57')
    modules, cfg = load_params(config_name, overrides=cli)
    if list(cfg.runner) != ['postcvpr']:
        raise ValueError('Recorded baseline currently supports the postcvpr runner only')
    if cfg.mode not in ['train', 'validate']:
        raise ValueError('mode must be train or validate')
    if cfg.mode == 'train' and (cfg.step_start != 0 or cfg.experiment.checkpoint_path):
        raise ValueError('Fresh training only. Use mode=validate for existing checkpoints; '
                         'exact resume is not implemented.')
    if cfg.mode == 'validate' and not cfg.get('checkpoint'):
        raise ValueError('mode=validate requires checkpoint=/absolute/path/to/model.pth')
    if cfg.dataloader.batch_size != 1:
        raise ValueError('HOOD supports batch_size=1')
    if cfg.validation.enabled and (cfg.validation.every <= 0 or cfg.validation.n_steps <= 0):
        raise ValueError('Validation intervals and n_steps must be positive')
    if cfg.experiment.save_checkpoint_every <= 0 or cfg.experiment.n_epochs <= 0:
        raise ValueError('Checkpoint interval and n_epochs must be positive')
    if cfg.experiment.max_iter is not None and cfg.experiment.max_iter <= 0:
        raise ValueError('max_iter must be positive')
    seed_all(int(cfg.seed))  # Includes model initialization, unlike the historical train.py.
    dataloader, runner_module, runner, aux = create_modules(modules, cfg)
    run = Experiment(cfg, DEFAULTS.project_dir, DEFAULTS.experiment_root)
    print('RUN_DIR=' + str(run.root), flush=True)
    started = time.monotonic()
    step, epoch = 0, 0
    try:
        with (run.root / 'logs' / 'console.log').open('x') as logfile, \
                contextlib.redirect_stdout(Tee(sys.stdout, logfile)), \
                contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            assets = {'training_split': file_record(Path(DEFAULTS.aux_data) / cfg.dataloader.dataset.postcvpr.split_path)}
            for key in ['smpl_model', 'garment_dict_file', 'obstacle_dict_file']:
                assets[key] = file_record(Path(DEFAULTS.aux_data) / cfg.dataloader.dataset.postcvpr[key])
            write_json(run.root / 'assets.json', assets)
            if cfg.mode == 'train':
                dataset_cfg = cfg.dataloader.dataset.postcvpr
                split = Path(DEFAULTS.aux_data) / dataset_cfg.split_path
                with split.open() as stream:
                    sequence_ids = sorted({row['id'] for row in csv.DictReader(stream)})
                data_root = Path(DEFAULTS.data_root) / dataset_cfg.data_root
                write_json(run.root / 'training_data_manifest.json',
                           {'sequences': [{'id': sid, **file_record(data_root / (sid + '.pkl'))}
                                          for sid in sequence_ids]})
            if cfg.mode == 'validate':
                state = torch.load(cfg.checkpoint, map_location=cfg.device)
                runner.load_state_dict(state.get('training_module', state))
                # Legacy checkpoint filenames are informational, never silently claim step=0.
                if 'global_step' in state:
                    step = int(state['global_step'])
                else:
                    stem = Path(cfg.checkpoint).stem
                    if not stem.startswith('step_') or not stem[5:].isdigit():
                        raise ValueError('Legacy checkpoint requires a step_NNN.pth filename')
                    step = int(stem[5:])
                run.metadata['evaluated_checkpoint'] = file_record(cfg.checkpoint)
                run.metadata['checkpoint_training_commit'] = state.get('git_commit', 'unrecorded_legacy')
            validator = FixedValidation(modules, cfg, run.root) if cfg.validation.enabled else None
            validation_log = None
            if validator:
                from utils.experiment import CSVLog
                validation_log = CSVLog(run.root / 'logs' / 'validation.csv')
            def validate(current_step):
                if validator:
                    summaries = validator.evaluate(runner, current_step)
                    for summary in summaries:
                        validation_log.write({'global_step': current_step, **summary})
                    run.tensorboard.validation([{'global_step': current_step, **s} for s in summaries])
                    print(f'VALIDATION step={current_step} sequences={len(summaries)}', flush=True)
            try:
                if cfg.mode == 'validate':
                    validate(step)
                else:
                    validate(0)
                    if cfg.detect_anomaly:
                        torch.autograd.set_detect_anomaly(True)
                    runner.train()
                    for epoch in range(cfg.experiment.n_epochs):
                        for sample in dataloader.create_dataloader():
                            if cfg.experiment.max_iter is not None and step >= cfg.experiment.max_iter:
                                break
                            step += 1
                            sample = move2device(sample, cfg.device)
                            sample = add_field_to_pyg_batch(sample, 'iter', [step], 'cloth', reference_key=None)
                            rolls = min(1 + step // runner.mcfg.increase_roll_every, runner.mcfg.roll_max)
                            optimized = step >= runner.mcfg.warmup_steps
                            rows = []
                            def observe(substep, state, losses, lr_used, grad_norm):
                                scalar = {k: float(v.detach()) for k, v in losses.items()}
                                if not all(np.isfinite(v) for v in scalar.values()):
                                    raise FloatingPointError(f'Non-finite physical loss at step {step}')
                                weights = [c.get_weight(step) for c in runner.criterion_dict.values() if hasattr(c, 'get_weight')]
                                row = {
                                    'global_step': step, 'epoch': epoch, 'rollout_substep': substep,
                                    'roll_steps': rolls, 'optimized': int(optimized),
                                    'sequence': sample.sequence_name[0], 'garment': sample.garment_name[0],
                                    'timestep': float(state['cloth'].timestep[0, 0]),
                                    'lr_used': lr_used, 'lr_next': aux['optimizer'].param_groups[0]['lr'],
                                    'grad_norm_before_clip': None if grad_norm is None else float(grad_norm),
                                    'collision_weight': weights[0] if weights else None,
                                    'lame_mu': float(state['cloth'].lame_mu[0]),
                                    'lame_lambda': float(state['cloth'].lame_lambda[0]),
                                    'bending_coeff': float(state['cloth'].bending_coeff[0]),
                                    **scalar, 'total_physical_loss': sum(scalar.values()),
                                }
                                rows.append(row)
                                run.substeps.write(row)
                            runner(sample, roll_steps=rolls,
                                   optimizer=aux['optimizer'] if optimized else None,
                                   scheduler=aux['scheduler'] if optimized else None, step_callback=observe)
                            physical_keys = [k for k in rows[0] if k.endswith('_loss')]
                            training_row = {'global_step': step, 'epoch': epoch, 'roll_steps': rolls,
                                             'optimized': int(optimized), 'elapsed_s': time.monotonic() - started,
                                             **{k + '_mean': float(np.mean([r[k] for r in rows])) for k in physical_keys},
                                             **{k + '_last': rows[-1][k] for k in physical_keys}}
                            run.train.write(training_row)
                            run.tensorboard.training(training_row)
                            run.tensorboard.optimizer(rows)
                            if step == 1 or step % 100 == 0:
                                print(f'TRAIN step={step} rolls={rolls} total={rows[-1]["total_physical_loss"]:.6g}', flush=True)
                            if step % cfg.experiment.save_checkpoint_every == 0:
                                checkpoint(runner, aux, cfg, run, step, epoch)
                            if validator and step % cfg.validation.every == 0:
                                validate(step)
                        if cfg.experiment.max_iter is not None and step >= cfg.experiment.max_iter:
                            break
                    if step % cfg.experiment.save_checkpoint_every != 0:
                        checkpoint(runner, aux, cfg, run, step, epoch)
                    if validator and step % cfg.validation.every != 0:
                        validate(step)
            finally:
                if validation_log:
                    validation_log.close()
        run.finish('completed', global_step=step, epoch=epoch, elapsed_s=time.monotonic() - started)
    except BaseException as error:
        run.finish('failed', global_step=step, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
