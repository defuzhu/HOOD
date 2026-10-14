"""Detached monitoring; this module never constructs a training objective."""
import time

import numpy as np
import torch


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class BatchObserver:
    def __init__(self, runner, aux, run, step, epoch, updates, sample_index=-1):
        self.runner, self.aux, self.run = runner, aux, run
        self.step, self.epoch, self.updates = step, epoch, updates
        self.sample_index = sample_index
        self.rows, self.physical_keys = [], None
        synchronize(runner.mcfg.device)
        self.started = self.previous = time.perf_counter()

    def record(self, substep, state, losses, lr_used, grad_norm):
        scalar = {key: float(value.detach()) for key, value in losses.items()}
        if not all(np.isfinite(value) for value in scalar.values()):
            raise FloatingPointError('Non-finite physical loss in observer')
        if self.physical_keys is None:
            self.physical_keys = list(scalar)
        diagnostics = {}
        for criterion in self.runner.criterion_dict.values():
            for key, value in getattr(criterion, 'last_diagnostics', {}).items():
                if key in scalar or key in diagnostics:
                    raise ValueError(f'Duplicate monitoring field: {key}')
                diagnostics[key] = float(value.detach()) if torch.is_tensor(value) else float(value)
        if not all(np.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError('Non-finite contact diagnostics')
        optimized = lr_used is not None
        self.updates += int(optimized)
        now = time.perf_counter()
        total = sum(scalar.values())
        gradient = None if grad_norm is None else float(grad_norm.detach())
        row = {
            'global_step': self.step, 'epoch': self.epoch, 'sample_index': self.sample_index,
            'rollout_substep': substep, 'roll_steps': None, 'optimized': int(optimized),
            'optimizer_step_count': self.updates,
            'optimizer_updates_in_batch': sum(r['optimized'] for r in self.rows) + int(optimized),
            'sequence': state.sequence_name[0], 'garment': state.garment_name[0],
            'timestep': float(state['cloth'].timestep[0, 0]),
            'learning_rate': lr_used, 'lr_used': lr_used,
            'lr_next': self.aux['optimizer'].param_groups[0]['lr'],
            'grad_norm_preclip': gradient, 'grad_norm_before_clip': gradient,
            'rollout_substep_time_ms': (now - self.previous) * 1000,
            'lame_mu': float(state['cloth'].lame_mu[0]),
            'lame_lambda': float(state['cloth'].lame_lambda[0]),
            'bending_coeff': float(state['cloth'].bending_coeff[0]),
            **scalar, 'total_loss': total, 'total_physical_loss': total, **diagnostics,
        }
        self.rows.append(row)
        self.previous = now

    def finish(self, rolls, elapsed_s):
        synchronize(self.runner.mcfg.device)
        step_time = (time.perf_counter() - self.started) * 1000
        if len(self.rows) != rolls:
            raise RuntimeError('Missing rollout observer callbacks')
        for row in self.rows:
            row['roll_steps'] = rolls
            self.run.substeps.write(row)
        physical = self.physical_keys + ['total_physical_loss', 'total_loss']
        summary = {
            'global_step': self.step, 'epoch': self.epoch, 'sample_index': self.sample_index,
            'roll_steps': rolls, 'optimized': self.rows[-1]['optimized'],
            'optimizer_step_count': self.updates,
            'optimizer_updates_in_batch': sum(row['optimized'] for row in self.rows),
            'step_time_ms': step_time, 'elapsed_s': elapsed_s,
            'learning_rate': self.rows[-1]['lr_used'],
        }
        for key in physical:
            summary[key + '_mean'] = float(np.mean([row[key] for row in self.rows]))
            summary[key + '_last'] = self.rows[-1][key]
        monitor = ['collision_raw_loss', 'collision_weight', 'grad_norm_preclip', 'learning_rate']
        monitor += [key for key in self.rows[0] if key.startswith(('contact_', 'barrier_', 'friction_')) and key not in physical]
        for key in monitor:
            values = [row.get(key) for row in self.rows if row.get(key) is not None]
            summary[key + '_mean'] = float(np.mean(values)) if values else None
            summary[key + '_last'] = values[-1] if values else None
        self.run.train.write(summary)
        self.run.tensorboard.training(summary)
        self.run.tensorboard.optimizer(self.rows)
        return summary
