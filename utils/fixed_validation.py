"""Fixed held-out motion rollouts, using the original postcvpr physics."""
import csv
import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.ops import knn_points
from torch_geometric.data import Batch

from utils.arguments import create_dataloader_module, create_runner, load_params
from utils.cloth_and_material import FaceNormals, get_shape_matrix, gather_triangles
from utils.common import NodeType, gather, move2device
from utils.defaults import DEFAULTS
from utils.experiment import CSVLog, file_record, isolated_rng, write_json
from utils.validation import Config, update_config_for_validation, update_config_single_sequence
from utils.contact_potential import vertex_rest_areas
from utils.training_observer import synchronize
from utils.rollout_io import DEFAULT_ROLLOUT_TEMPLATE, rollout_filename


def motion_id(sequence):
    """VTO IDs encode garment/shape followed by the CMU subject and motion."""
    return '_'.join(Path(sequence).stem.split('_')[-2:])


def geometry_metrics(state, contact_query=None):
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
    metrics = {
        # This is the original face-center/normal approximation, not exact penetration or CCD.
        'approx_penetration_fraction': (signed < 0).float().mean(),
        'approx_penetration_depth_mean_m': depth.mean(),
        'approx_penetration_depth_max_m': depth.max(),
        'principal_stretch_max': singular_values.max(),
        'principal_stretch_min': singular_values.min(),
        'principal_strain_rms': ((singular_values - 1) ** 2).mean().sqrt(),
        'pinned_error_max_m': pin_error.max() if handles.any() else y.new_tensor(0.),
        'principal_extreme_strain_fraction': ((singular_values > 2) | (singular_values < .5)).float().mean(),
        # HOOD's field called velocity is a per-frame displacement, not m/s.
        'cloth_speed_rms_m_s': ((cloth.pred_pos - cloth.pos) / cloth.timestep).square().sum(-1).mean().sqrt(),
        'cloth_acceleration_rms_m_s2': ((cloth.pred_pos - cloth.pos - cloth.velocity)
                                      / cloth.timestep.square()).square().sum(-1).mean().sqrt(),
    }
    if contact_query is not None:
        signed_mesh = contact_query(y, body.target_pos, faces)['signed_distance']
        depth_mesh = (-signed_mesh).clamp_min(0)
        free = ~handles
        penetrated = signed_mesh < -1e-7
        area = vertex_rest_areas(cloth.rest_pos, cloth.faces_batch.T)
        metrics.update({
            'mesh_vertex_penetration_fraction': (signed_mesh < -1e-7).float().mean(),
            'mesh_vertex_penetration_depth_mean_m': depth_mesh.mean(),
            'mesh_vertex_penetration_depth_max_m': depth_mesh.max(),
            'mesh_vertex_clearance_min_m': signed_mesh.min(),
            'mesh_free_vertex_penetration_fraction': (signed_mesh[free] < -1e-7).float().mean()
                if free.any() else y.new_tensor(0.),
            'mesh_penetrating_depth_mean_m': depth_mesh[penetrated].mean() if penetrated.any() else y.new_tensor(0.),
            'mesh_penetrating_depth_p95_m': torch.quantile(depth_mesh[penetrated], .95) if penetrated.any() else y.new_tensor(0.),
            'mesh_penetration_area_fraction': area[penetrated].sum() / area.sum().clamp_min(1e-30),
        })
    return metrics


class FixedValidation:
    def __init__(self, modules, cfg, run_dir):
        self.cfg, self.root = cfg, Path(run_dir)
        self.contact_query = None
        from utils.contact_geometry import ClosedMeshDistance
        self.contact_query = ClosedMeshDistance()
        entries = list(cfg.validation.sequences)
        if not entries:
            raise ValueError('Fixed validation requires at least one sequence')
        self.output_names = {}
        for entry in entries:
            key = (entry.id, entry.garment)
            name = rollout_filename(cfg.validation.get('rollout_experiment') or cfg.experiment.name,
                                    cfg.seed, entry.id, entry.garment,
                                    cfg.validation.get('rollout_name_template', DEFAULT_ROLLOUT_TEMPLATE))
            if key in self.output_names or name in self.output_names.values():
                raise ValueError('Validation sequences or rollout filenames are duplicated')
            self.output_names[key] = name
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
            # The evaluator always uses the same original postcvpr constitutive
            # models and a unit-weight signed-mesh cubic diagnostic. Training
            # criteria/modules never define evaluation physics.
            common_modules, common_cfg = load_params('baseline/postcvpr_seed57')
            vcfg.criterions = common_cfg.criterions
            contact = vcfg.criterions['postcvpr.collision_penalty']
            contact.contact_geometry, contact.contact_potential = 'signed_mesh', 'cubic'
            contact.weight_start = contact.weight_max = 1.
            contact.eps = .002
            evaluation_modules = dict(modules, criterions=common_modules['criterions'])
            self.validation_config = vcfg
            _, self.runner, _ = create_runner(evaluation_modules, vcfg, create_aux_modules=False)
            self.runner.eval()
            for entry in entries:
                path = data_root / (entry.id + '.pkl')
                scfg = OmegaConf.create(OmegaConf.to_container(vcfg))
                update_config_single_sequence(scfg, str(path), entry.garment)
                data = create_dataloader_module(modules, scfg).dataset[0]
                # Keep just the requested frames on the GPU, preserving original frame indexing.
                n_steps = data['obstacle'].pos.shape[1] if cfg.validation.n_steps == -1 else min(
                    int(cfg.validation.n_steps), data['obstacle'].pos.shape[1])
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
        if self.contact_query is not None:
            manifest['contact_metric_note'] = ('mesh_* metrics use global nearest point-triangle distances '
                                              'and winding signs. Vertex samples only, not edge/face '
                                              'intersection tests or CCD; self-intersecting bodies retain '
                                              'winding semantics.')
        write_json(self.root / 'validation_manifest.json', manifest)
        definition = {
            'version': 'common_postcvpr_mesh_v2', 'seed': int(cfg.validation.seed),
            'material': dict(cfg.validation.material),
            'data': [{key: e[key] for key in ('id', 'garment', 'n_steps', 'sha256')} for e in self.records],
            'assets': {key: file_record(Path(DEFAULTS.aux_data) / dataset_cfg[key])['sha256']
                       for key in ('smpl_model', 'garment_dict_file', 'obstacle_dict_file')},
            'dataset': OmegaConf.to_container(vcfg.dataloader.dataset, resolve=True),
            'criteria': OmegaConf.to_container(vcfg.criterions, resolve=True),
            'timesteps': [float(vcfg.runner.postcvpr.initial_ts), float(vcfg.runner.postcvpr.regular_ts)],
            'geometry_source_sha256': file_record(Path(__file__).parent / 'contact_geometry.py')['sha256'],
            'metric_source_sha256': file_record(__file__)['sha256'],
        }
        # Data paths are recorded separately; semantic signature uses content.
        definition['dataset']['postcvpr'].pop('single_sequence_path', None)
        self.signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        manifest.update(evaluation_signature=self.signature, evaluation_definition=definition,
                        metric_note='Common original physics; collision_penalty_loss is a unit-weight signed '
                                    'mesh cubic diagnostic, not any variant training objective.')
        write_json(self.root / 'validation_manifest.json', manifest)
        OmegaConf.save(vcfg, self.root / 'validation_config.yaml', resolve=True)

    def evaluate(self, training_runner, step):
        destination = self.root / 'validation' / f'step_{step:010d}'
        destination.mkdir(exist_ok=False)
        summaries = []
        with isolated_rng(int(self.cfg.validation.seed)), torch.no_grad():
            self.runner.load_state_dict(training_runner.state_dict())
            self.runner.cloth_obj.cache.clear()
            with contextlib.ExitStack() as stack:
                collision_log = CSVLog(destination / 'collision_per_frame.csv')
                physics_log = CSVLog(destination / 'physics_metrics.csv')
                stack.callback(collision_log.close)
                stack.callback(physics_log.close)
                for sequence_id, garment, sequence, n_steps in self.sequences:
                    self.runner.cloth_obj.cache.clear()
                    frame_log = CSVLog(destination / f'{sequence_id}_{garment}_frames.csv')
                    stack.callback(frame_log.close)
                    def observe(frame, state, timing):
                        # criterion_pass is the common evaluator's original physics;
                        # diagnostics stay completely outside training loss_dict.
                        physics = self.runner.criterion_pass(state)
                        contact = self.runner.criterion_dict['collision_penalty'].last_diagnostics
                        metrics = {**physics, 'total_physical_loss': sum(physics.values()),
                                   **geometry_metrics(state, self.contact_query), **timing,
                                   'collision_raw_loss': contact['collision_raw_loss']}
                        metrics = {key: float(value) for key, value in metrics.items()}
                        if not all(np.isfinite(value) for value in metrics.values()):
                            raise FloatingPointError(f'Non-finite evaluation metric at frame {frame}')
                        row = {'global_step': step, 'sequence': sequence_id, 'garment': garment,
                               'frame': frame, 'initialization_step': int(frame == 0), **metrics}
                        frame_log.write(row)
                        common = {key: row[key] for key in ('global_step', 'sequence', 'garment', 'frame', 'initialization_step')}
                        collision_log.write({**common, **{key: value for key, value in metrics.items()
                                                         if key.startswith(('mesh_', 'approx_', 'collision_'))}})
                        physics_log.write({**common, **{key: value for key, value in metrics.items()
                                                       if not key.startswith(('mesh_', 'approx_', 'collision_'))}})
                        return metrics
                    device = self.cfg.device
                    synchronize(device)
                    baseline_memory = 0
                    if torch.device(device).type == 'cuda':
                        baseline_memory = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    rollout = self.runner.valid_rollout(sequence.clone(), n_steps=n_steps, bare=True, frame_callback=observe)
                    synchronize(device)
                    evaluation_seconds = time.perf_counter() - started
                    values = rollout['metrics']
                    if not np.isfinite(rollout['pred']).all():
                        raise FloatingPointError(f'Non-finite rollout: {sequence_id}')
                    summary = {'sequence': sequence_id, 'garment': garment, 'frames': n_steps}
                    summary.update(evaluation_wall_time_s=evaluation_seconds,
                                   inference_frames_total_s=sum(values['frame_inference_time_ms']) / 1000,
                                   model_parameters=sum(p.numel() for p in self.runner.model.parameters()))
                    if torch.device(device).type == 'cuda':
                        peak = torch.cuda.max_memory_allocated(device)
                        summary.update(evaluation_cuda_peak_allocated_mib=peak / 1024 ** 2,
                                       rollout_cuda_peak_increment_mib=(peak - baseline_memory) / 1024 ** 2)
                    else:
                        summary.update(evaluation_cuda_peak_allocated_mib=0., rollout_cuda_peak_increment_mib=0.)
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
                        rollout_path = destination / self.output_names[(sequence_id, garment)]
                        np.savez_compressed(rollout_path,
                                            pred=rollout['pred'], obstacle=rollout['obstacle'],
                                            cloth_faces=rollout['cloth_faces'], obstacle_faces=rollout['obstacle_faces'])
                        if self.cfg.validation.get('save_videos', False):
                            from render_baseline import render_rollout
                            videos = self.root / 'videos' / f'step_{step:010d}'
                            videos.mkdir(parents=True, exist_ok=True)
                            render_rollout(rollout_path,
                                           videos / rollout_path.with_suffix('.mp4').name,
                                           stride=int(self.cfg.validation.get('video_stride', 4)),
                                           max_frames=int(self.cfg.validation.get('video_max_frames', 60)))
        write_json(destination / 'metrics.json', {'global_step': step, 'evaluation_signature': self.signature,
                                                'sequences': summaries})
        return summaries
