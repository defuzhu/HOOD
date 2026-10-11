"""Identical scalar tags for live recorded runs and historical CSV imports."""
import math
from pathlib import Path


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite TensorBoard scalar: {value}')
    return result


class TrainingTensorBoard:
    def __init__(self, directory, enabled=True):
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(Path(directory)), flush_secs=30)

    def scalar(self, tag, value, step):
        if self.writer is not None:
            value = number(value)
            if value is not None:
                self.writer.add_scalar(tag, value, int(step))

    def training(self, row):
        for key, value in row.items():
            if key.endswith(('_loss_mean', '_loss_last')):
                self.scalar('train/' + key, value, row['global_step'])
        for key in ('roll_steps', 'optimized'):
            self.scalar('train_state/' + key, row[key], row['global_step'])

    def optimizer(self, rows):
        # One point per outer training step; rollouts have multiple Adam updates.
        rows = list(rows)
        step = rows[0]['global_step']
        for key in ('lr_used', 'lr_next', 'grad_norm_before_clip', 'collision_weight'):
            values = [number(row.get(key)) for row in rows]
            values = [v for v in values if v is not None]
            if values:
                self.scalar('optimizer/' + key + '_mean', sum(values) / len(values), step)
                self.scalar('optimizer/' + key + '_last', values[-1], step)
                if key == 'grad_norm_before_clip':
                    self.scalar('optimizer/' + key + '_max', max(values), step)

    def validation(self, rows):
        # Keep sequence identity; also report an explicitly equal-sequence mean.
        aggregate = {}
        step = None
        for row in rows:
            step = row['global_step']
            sequence = row['sequence'] + '__' + row['garment']
            for key, value in row.items():
                if key in ('global_step', 'sequence', 'garment', 'frames'):
                    continue
                value = number(value)
                if value is not None:
                    self.scalar('validation/' + sequence + '/' + key, value, step)
                    aggregate.setdefault(key, []).append(value)
        for key, values in aggregate.items():
            self.scalar('validation/sequence_mean/' + key, sum(values) / len(values), step)
        self.flush()

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
