"""Recorded HOOD training, detached logs, batch-boundary resume and common validation."""
import contextlib
import csv
import os
import signal
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.arguments import create_modules, load_params
from utils.common import add_field_to_pyg_batch, move2device
from utils.defaults import DEFAULTS
from utils.experiment import CSVLog, Experiment, file_record, seed_all, write_json
from utils.fixed_validation import FixedValidation
from utils.resume import (ResumableSampler, data_fingerprint, loader_iterator,
                          restore_rng, restore_training, save_recorded_checkpoint)
from utils.training_observer import BatchObserver


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


def resume_config(config):
    result = OmegaConf.to_container(config, resolve=True)
    for key in ('run_dir', 'resume', 'resume_allow_code_change', 'checkpoint', 'init_checkpoint', 'protocol'):
        result.pop(key, None)
    result['experiment'].pop('max_iter', None)
    result['experiment'].pop('n_epochs', None)
    result['experiment'].pop('name', None)
    result.pop('device', None)
    return result


def main(overrides=None):
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    cli = OmegaConf.from_cli() if overrides is None else overrides
    resumed = torch.load(cli.resume, map_location='cpu') if cli.get('resume') else None
    config_name = cli.get('config', resumed['config']['config'] if resumed else 'baseline/postcvpr_seed57')
    options = OmegaConf.merge(OmegaConf.create(resumed['config']), cli) if resumed else cli
    if resumed:
        options.pop('run_dir', None)
    modules, cfg = load_params(config_name, overrides=options)
    if list(cfg.runner) != ['postcvpr']:
        raise ValueError('Recorded training supports the postcvpr runner')
    if cfg.mode not in ('train', 'validate'):
        raise ValueError('mode must be train or validate')
    if cfg.dataloader.batch_size != 1 or cfg.dataloader.num_workers != 0:
        raise ValueError('Recorded reproducible sampling requires batch_size=1, num_workers=0')
    if cfg.step_start != 0 or cfg.experiment.checkpoint_path:
        raise ValueError('Use resume= for stateful continuation, init_checkpoint= for a labelled fine-tune')
    if cfg.mode == 'validate' and not cfg.get('checkpoint'):
        raise ValueError('mode=validate requires checkpoint=/absolute/path/model.pth')
    if resumed:
        if cfg.mode != 'train' or cfg.get('init_checkpoint'):
            raise ValueError('resume is only for continuation of training')
        if resume_config(cfg) != resume_config(OmegaConf.create(resumed['config'])):
            raise ValueError('Resume changes training settings. Only name/device/budget may change; '
                             'use a separate fine-tune for different objectives/settings.')
        if cfg.experiment.max_iter is not None and cfg.experiment.max_iter <= resumed['global_step']:
            raise ValueError('Training budget is already completed; extend max_iter to continue')
    if cfg.validation.enabled and (cfg.validation.every <= 0 or cfg.validation.n_steps == 0 or cfg.validation.n_steps < -1):
        raise ValueError('Validation every must be positive; n_steps is positive or -1 for full sequences')
    if cfg.experiment.save_checkpoint_every <= 0 or cfg.experiment.n_epochs <= 0:
        raise ValueError('Checkpoint interval and epoch budget must be positive')
    if cfg.experiment.max_iter is not None and cfg.experiment.max_iter <= 0:
        raise ValueError('max_iter must be positive')
    seed_all(int(cfg.seed))
    dataloader, _, runner, aux = create_modules(modules, cfg)
    run = Experiment(cfg, DEFAULTS.project_dir, DEFAULTS.experiment_root)
    print('RUN_DIR=' + str(run.root), flush=True)
    started = time.monotonic()
    progress = {'global_step': 0, 'optimizer_step_count': 0, 'epoch': 0,
                'elapsed_s': 0., 'best_validation': None}
    validation_log, sampler = None, None
    stop = {'requested': False}
    previous_handlers = {}
    def request_stop(signum, frame):
        stop['requested'] = True
        print('STOP requested; finishing current batch, then saving a recoverable checkpoint', flush=True)
    if cfg.mode == 'train':
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        with (run.root / 'logs' / 'console.log').open('x') as logfile, \
                contextlib.redirect_stdout(Tee(sys.stdout, logfile)), \
                contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            dataset_cfg = cfg.dataloader.dataset.postcvpr
            assets = {'training_split': file_record(Path(DEFAULTS.aux_data) / dataset_cfg.split_path)}
            for key in ('smpl_model', 'garment_dict_file', 'obstacle_dict_file'):
                assets[key] = file_record(Path(DEFAULTS.aux_data) / dataset_cfg[key])
            write_json(run.root / 'assets.json', assets)
            fingerprint = None
            if cfg.mode == 'train':
                with (Path(DEFAULTS.aux_data) / dataset_cfg.split_path).open() as stream:
                    ids = sorted({row['id'] for row in csv.DictReader(stream)})
                manifest = {'sequences': [{'id': sid, **file_record(Path(DEFAULTS.data_root) / dataset_cfg.data_root / (sid + '.pkl'))}
                                          for sid in ids]}
                write_json(run.root / 'training_data_manifest.json', manifest)
                fingerprint = data_fingerprint(assets, manifest)
                if resumed:
                    restore_training(resumed, runner, aux, dataloader.dataset, fingerprint,
                                     run.metadata['git_commit'], cfg.get('resume_allow_code_change', False))
                    progress = {key: resumed[key] for key in progress}
                    run.metadata['resume'] = {'parent_run': resumed['source_run'],
                                              'from_step': progress['global_step'],
                                              'checkpoint': file_record(cli.resume),
                                              'source_commit': resumed['git_commit'],
                                              'code_changed': resumed['git_commit'] != run.metadata['git_commit']}
                elif cfg.get('init_checkpoint'):
                    state = torch.load(cfg.init_checkpoint, map_location='cpu')
                    runner.load_state_dict(state.get('training_module', state))
                    run.metadata['initialization_checkpoint'] = file_record(cfg.init_checkpoint)
                    run.metadata['training_kind'] = 'fine_tune_with_fresh_optimizer'
                sampler = ResumableSampler(len(dataloader.dataset), resumed['sampler'] if resumed else None)
            else:
                state = torch.load(cfg.checkpoint, map_location='cpu')
                runner.load_state_dict(state.get('training_module', state))
                if 'global_step' in state:
                    progress['global_step'] = int(state['global_step'])
                else:
                    stem = Path(cfg.checkpoint).stem
                    if not stem.startswith('step_') or not stem[5:].isdigit():
                        raise ValueError('Legacy checkpoint requires a step_NNN.pth filename')
                    progress['global_step'] = int(stem[5:])
                progress['optimizer_step_count'] = state.get('optimizer_step_count')
                run.metadata['evaluated_checkpoint'] = file_record(cfg.checkpoint)
                run.metadata['checkpoint_training_commit'] = state.get('git_commit', 'unrecorded_legacy')
            write_json(run.root / 'metadata.json', run.metadata)
            validator = FixedValidation(modules, cfg, run.root) if cfg.validation.enabled else None
            if validator:
                validation_log = CSVLog(run.root / 'logs' / 'validation.csv')
            def validate(step):
                if not validator:
                    return False
                summaries = validator.evaluate(runner, step)
                rows = [{'global_step': step, 'optimizer_step_count': progress['optimizer_step_count'], **s} for s in summaries]
                for row in rows:
                    validation_log.write(row)
                run.tensorboard.validation(rows)
                metric = cfg.validation.get('best_metric', 'mesh_free_vertex_penetration_fraction_regular_mean')
                improved = False
                if metric in summaries[0] and step > 0:
                    score = sum(s[metric] for s in summaries) / len(summaries)
                    if progress['best_validation'] is None or score < progress['best_validation']['value']:
                        progress['best_validation'] = {'metric': metric, 'value': score, 'global_step': step}
                        improved = True
                print(f'VALIDATION step={step} sequences={len(summaries)}', flush=True)
                return improved
            if cfg.mode == 'validate':
                validate(progress['global_step'])
            else:
                elapsed_before = float(progress['elapsed_s'])
                if resumed:
                    # Restore only after all constructors; validator construction is isolated too.
                    restore_rng(resumed['rng_state'])
                else:
                    validate(0)
                runner.train()
                if cfg.detect_anomaly:
                    torch.autograd.set_detect_anomaly(True)
                restoring_iterator = resumed is not None
                while sampler.epoch < cfg.experiment.n_epochs and not stop['requested']:
                    if sampler.order is not None and sampler.cursor == sampler.length:
                        sampler.next_epoch()
                        restoring_iterator = False
                        if sampler.epoch >= cfg.experiment.n_epochs:
                            break
                    iterator = loader_iterator(dataloader.dataset, sampler, restoring_iterator)
                    restoring_iterator = False
                    while sampler.cursor < sampler.length and not stop['requested']:
                        if cfg.experiment.max_iter is not None and progress['global_step'] >= cfg.experiment.max_iter:
                            break
                        step = progress['global_step'] + 1
                        observer = BatchObserver(runner, aux, run, step, sampler.epoch, progress['optimizer_step_count'])
                        sample = next(iterator)
                        observer.sample_index = sampler.order[sampler.cursor]
                        sample = move2device(sample, cfg.device)
                        sample = add_field_to_pyg_batch(sample, 'iter', [step], 'cloth', reference_key=None)
                        rolls = min(1 + step // runner.mcfg.increase_roll_every, runner.mcfg.roll_max)
                        optimized = step >= runner.mcfg.warmup_steps
                        runner(sample, roll_steps=rolls, optimizer=aux['optimizer'] if optimized else None,
                               scheduler=aux['scheduler'] if optimized else None, step_callback=observer.record)
                        elapsed = elapsed_before + time.monotonic() - started
                        summary = observer.finish(rolls, elapsed)
                        sampler.commit()
                        progress.update(global_step=step, epoch=sampler.epoch,
                                        optimizer_step_count=observer.updates, elapsed_s=elapsed)
                        last = ((cfg.experiment.max_iter is not None and step >= cfg.experiment.max_iter)
                                or (sampler.cursor == sampler.length and sampler.epoch + 1 >= cfg.experiment.n_epochs))
                        improved = False
                        if validator and not stop['requested'] and (step % cfg.validation.every == 0 or last):
                            improved = validate(step)
                        if step == 1 or step % 100 == 0:
                            print(f'TRAIN step={step} updates={observer.updates} rolls={rolls} total={summary["total_physical_loss_last"]:.6g}', flush=True)
                        if step % cfg.experiment.save_checkpoint_every == 0 or last or stop['requested'] or improved:
                            progress['elapsed_s'] = elapsed_before + time.monotonic() - started
                            path = save_recorded_checkpoint(runner, aux, cfg, run, sampler, dataloader.dataset, progress, fingerprint)
                            run.metadata['latest_checkpoint'] = str(path)
                            if improved:
                                write_json(run.root / 'checkpoints' / 'best_checkpoint.json',
                                           {'path': str(path), **progress['best_validation']})
                        if last:
                            break
                    if cfg.experiment.max_iter is not None and progress['global_step'] >= cfg.experiment.max_iter:
                        break
                if progress['global_step'] > 0:
                    path = save_recorded_checkpoint(runner, aux, cfg, run, sampler, dataloader.dataset, progress, fingerprint)
                    run.metadata['latest_checkpoint'] = str(path)
        run.finish('interrupted' if stop['requested'] else 'completed', **progress,
                   invocation_elapsed_s=time.monotonic() - started)
    except BaseException as error:
        run.finish('failed', **progress, error=f'{type(error).__name__}: {error}',
                   recovery_note='Resume the last completed checkpoint; no partially optimized batch is saved.')
        raise
    finally:
        if validation_log:
            validation_log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return run.root


if __name__ == '__main__':
    main()
