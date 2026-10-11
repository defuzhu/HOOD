"""Run records and RNG isolation; independent of the cloth constitutive models."""
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed):
    """Validation must not change future training samples or random materials."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            seed_all(seed)
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def git_output(project, *args):
    return subprocess.check_output(['git', '-C', str(project), *args], text=True).strip()


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class CSVLog:
    def __init__(self, path):
        self.stream = Path(path).open('x', newline='')
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()


class Experiment:
    def __init__(self, cfg, project, experiment_root):
        status = git_output(project, 'status', '--porcelain')
        if cfg.recording.require_clean and status:
            raise RuntimeError('Recorded runs require a clean checkout. Commit changes first; '
                               'recording.require_clean=false is only for development checks.')
        name = cfg.experiment.name
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('experiment.name must contain only letters, digits, underscores or hyphens')
        identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        self.root = Path(experiment_root) / f'{name}_seed{cfg.seed}_{identity}'
        self.root.mkdir(parents=True, exist_ok=False)
        for child in ['logs', 'checkpoints', 'validation']:
            (self.root / child).mkdir()
        cfg.run_dir = str(self.root)
        OmegaConf.save(cfg, self.root / 'config.yaml', resolve=True)
        self.metadata = {
            'experiment': name, 'seed': int(cfg.seed), 'created_utc': identity.split('_')[0],
            'git_commit': git_output(project, 'rev-parse', 'HEAD'),
            'git_branch': git_output(project, 'rev-parse', '--abbrev-ref', 'HEAD'),
            'git_status': status, 'reproducible_code_snapshot': not bool(status),
            'argv': sys.argv, 'project': str(Path(project).resolve()),
            'python': sys.version, 'python_executable': sys.executable,
            'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'status': 'running',
        }
        write_json(self.root / 'metadata.json', self.metadata)
        if status:
            (self.root / 'working_changes.patch').write_text(git_output(project, 'diff', 'HEAD'))
        packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
        (self.root / 'environment.txt').write_text('\n'.join(packages) + '\n')
        self.train = CSVLog(self.root / 'logs' / 'train.csv')
        self.substeps = CSVLog(self.root / 'logs' / 'train_substeps.csv')
        from utils.tensorboard_logging import TrainingTensorBoard
        self.tensorboard = TrainingTensorBoard(self.root / 'tensorboard',
                                              enabled=cfg.recording.get('tensorboard', True))

    def finish(self, status, **details):
        self.metadata.update(status=status, **details)
        write_json(self.root / 'metadata.json', self.metadata)
        self.train.close()
        self.substeps.close()
        self.tensorboard.close()
