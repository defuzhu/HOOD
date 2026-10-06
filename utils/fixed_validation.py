"""Fixed held-out motion rollouts, using the original postcvpr physics."""
import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.ops import knn_points
from torch_geometric.data import Batch

from utils.arguments import create_dataloader_module, create_runner
from utils.cloth_and_material import FaceNormals, get_shape_matrix, gather_triangles
from utils.common import NodeType, gather, move2device
from utils.defaults import DEFAULTS
from utils.experiment import CSVLog, file_record, isolated_rng, write_json
from utils.validation import Config, update_config_for_validation, update_config_single_sequence


def motion_id(sequence):
    """VTO IDs encode garment/shape followed by the CMU subject and motion."""
    return '_'.join(Path(sequence).stem.split('_')[-2:])


def geometry_metrics(state):
    cloth, body = state['cloth'], state['obstacle']
    y = cloth.pred_pos
    faces = body.faces_batch.T
    centers = gather(body.target_pos, faces, 0, 1, 1).mean(dim=-2)
    normals = FaceNormals()(body.target_pos[None], faces[None])[0]
    _, indices, _ = knn_points(y[None], centers[None])
    indices = indices[0, :, 0]
    signed = ((y - centers[indices]) * normals[indices]).sum(-1)
    depth = torch.clamp(-signed, min=0)
    triangles = gather_triangles(y[None], cloth.faces_batch.T)[0]
    singular_values = torch.linalg.svdvals(get_shape_matrix(triangles) @ cloth.Dm_inv)
    handles = cloth.vertex_type[:, 0] == NodeType.HANDLE
    pin_error = torch.linalg.vector_norm(y[handles] - cloth.target_pos[handles], dim=-1)
    return {
        # This is the original face-center/normal approximation, not exact penetration or CCD.
        'approx_penetration_fraction': (signed < 0).float().mean(),
        'approx_penetration_depth_mean_m': depth.mean(),
        'approx_penetration_depth_max_m': depth.max(),
        'principal_stretch_max': singular_values.max(),
        'principal_stretch_min': singular_values.min(),
        'principal_strain_rms': ((singular_values - 1) ** 2).mean().sqrt(),
        'pinned_error_max_m': pin_error.max() if handles.any() else y.new_tensor(0.),
    }


class FixedValidation:
    def __init__(self, modules, cfg, run_dir):
        self.cfg, self.root = cfg, Path(run_dir)
        entries = list(cfg.validation.sequences)
        if not entries:
            raise ValueError('Fixed validation requires at least one sequence')
        dataset_cfg = cfg.dataloader.dataset.postcvpr
        data_root = Path(DEFAULTS.data_root) / dataset_cfg.data_root
        train_path = Path(DEFAULTS.aux_data) / dataset_cfg.split_path
        with train_path.open() as stream:
            train_motions = {motion_id(row['id']) for row in csv.DictReader(stream)}
        if cfg.validation.split_kind == 'held_out_motion':
            overlaps = [e.id for e in entries if motion_id(e.id) in train_motions]
            if overlaps:
                raise ValueError(f'Validation motions overlap training: {overlaps}')
        # Construction, graph-center selection and SMPL/LBS loading are isolated too.
        self.sequences = []
        self.records = []
        with isolated_rng(int(cfg.validation.seed)):
            material = dict(cfg.validation.material)
            options = Config(garment_dict_file=dataset_cfg.garment_dict_file,
                             smpl_model=dataset_cfg.smpl_model, **material)
            vcfg = update_config_for_validation(OmegaConf.create(OmegaConf.to_container(cfg)), options)
            self.validation_config = vcfg
            _, self.runner, _ = create_runner(modules, vcfg, create_aux_modules=False)
            self.runner.eval()
            for entry in entries:
                path = data_root / (entry.id + '.pkl')
                scfg = OmegaConf.create(OmegaConf.to_container(vcfg))
                update_config_single_sequence(scfg, str(path), entry.garment)
                data = create_dataloader_module(modules, scfg).dataset[0]
                # Keep just the requested frames on the GPU, preserving original frame indexing.
                n_steps = min(int(cfg.validation.n_steps), data['obstacle'].pos.shape[1])
                if n_steps <= 0:
                    raise ValueError('Validation n_steps must be positive')
                for obj in ['cloth', 'obstacle']:
                    for key in ['prev_pos', 'pos', 'target_pos']:
                        data[obj][key] = data[obj][key][:, :n_steps].clone()
                sequence = move2device(Batch.from_data_list([data]), cfg.device)
                self.sequences.append((entry.id, entry.garment, sequence, n_steps))
                self.records.append({'id': entry.id, 'garment': entry.garment, 'n_steps': n_steps,
                                     'motion': motion_id(entry.id), **file_record(path)})
        manifest = {'seed': int(cfg.validation.seed), 'split_kind': cfg.validation.split_kind,
                    'random_betas': False, 'noise_scale': 0, 'material': dict(cfg.validation.material),
                    'sequences': self.records, 'training_split': file_record(train_path),
                    'metric_note': 'Penetration metrics approximate nearest face centers and normals; '
                                   'not exact signed distance, CCD or supervised trajectory error.'}
        write_json(self.root / 'validation_manifest.json', manifest)
        OmegaConf.save(vcfg, self.root / 'validation_config.yaml', resolve=True)

    def evaluate(self, training_runner, step):
        destination = self.root / 'validation' / f'step_{step:010d}'
        destination.mkdir(exist_ok=False)
        summaries = []
        with isolated_rng(int(self.cfg.validation.seed)), torch.no_grad():
            self.runner.load_state_dict(training_runner.state_dict())
            self.runner.cloth_obj.cache.clear()
            original_criterion = self.runner.criterion_pass
            def criterion_with_geometry(state):
                physics = original_criterion(state)
                return {**physics, 'total_physical_loss': sum(physics.values()), **geometry_metrics(state)}
            self.runner.criterion_pass = criterion_with_geometry
            try:
                for sequence_id, garment, sequence, n_steps in self.sequences:
                    rollout = self.runner.valid_rollout(sequence.clone(), n_steps=n_steps, bare=False)
                    values = rollout['metrics']
                    frame_log = CSVLog(destination / f'{sequence_id}_{garment}_frames.csv')
                    try:
                        for frame in range(n_steps):
                            frame_log.write({'frame': frame, 'initialization_step': int(frame == 0),
                                             **{k: float(v[frame]) for k, v in values.items()}})
                    finally:
                        frame_log.close()
                    if not np.isfinite(rollout['pred']).all():
                        raise FloatingPointError(f'Non-finite rollout: {sequence_id}')
                    summary = {'sequence': sequence_id, 'garment': garment, 'frames': n_steps}
                    for key, value in values.items():
                        array = np.asarray(value, dtype=float)
                        if not np.isfinite(array).all():
                            raise FloatingPointError(f'Non-finite metric: {key}')
                        summary[key + '_mean'] = float(array.mean())
                        summary[key + '_max'] = float(array.max())
                        # The first step uses a different timestep; report regular steps separately.
                        if n_steps > 1:
                            summary[key + '_regular_mean'] = float(array[1:].mean())
                    summaries.append(summary)
                    if self.cfg.validation.save_rollouts:
                        np.savez_compressed(destination / f'{sequence_id}_{garment}_rollout.npz',
                                            pred=rollout['pred'], obstacle=rollout['obstacle'],
                                            cloth_faces=rollout['cloth_faces'], obstacle_faces=rollout['obstacle_faces'])
            finally:
                self.runner.criterion_pass = original_criterion
        write_json(destination / 'metrics.json', {'global_step': step, 'sequences': summaries})
        return summaries
