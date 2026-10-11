from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight_start: float = 1e+3
    weight_max: float = 1e+5
    start_rampup_iteration: int = 50000
    n_rampup_iterations: int = 100000
    eps: float = 1e-3
    contact_geometry: str = 'legacy_plane'
    point_chunk: int = 256
    face_chunk: int = 2048
    min_triangle_area: float = 1e-12


def create(mcfg):
    return Criterion(weight_start=mcfg.weight_start, weight_max=mcfg.weight_max,
                     start_rampup_iteration=mcfg.start_rampup_iteration, n_rampup_iterations=mcfg.n_rampup_iterations,
                     eps=mcfg.eps, contact_geometry=mcfg.contact_geometry,
                     point_chunk=mcfg.point_chunk, face_chunk=mcfg.face_chunk,
                     min_triangle_area=mcfg.min_triangle_area)


class Criterion(nn.Module):
    def __init__(self, weight_start, weight_max, start_rampup_iteration, n_rampup_iterations, eps=1e-3,
                 contact_geometry='legacy_plane', point_chunk=256, face_chunk=2048, min_triangle_area=1e-12):
        super().__init__()
        self.weight_start = weight_start
        self.weight_max = weight_max
        self.start_rampup_iteration = start_rampup_iteration
        self.n_rampup_iterations = n_rampup_iterations
        self.eps = eps
        self.f_normals_f = FaceNormals()
        self.name = 'collision_penalty'
        if contact_geometry not in ['legacy_plane', 'signed_mesh']:
            raise ValueError('contact_geometry must be legacy_plane or signed_mesh')
        self.contact_geometry = contact_geometry
        self.mesh_distance = None
        if contact_geometry == 'signed_mesh':
            from utils.contact_geometry import ClosedMeshDistance
            self.mesh_distance = ClosedMeshDistance(point_chunk, face_chunk, min_triangle_area)

    def get_weight(self, iter):
        iter = iter - self.start_rampup_iteration
        iter = max(iter, 0)
        progress = iter / self.n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.weight_start + (self.weight_max - self.weight_start) * progress
        return weight

    def calc_loss(self, example):
        obstacle_next_pos = example['obstacle'].target_pos
        obstacle_curr_pos = example['obstacle'].pos
        obstacle_faces = example['obstacle'].faces_batch.T

        curr_pos = example['cloth'].pos
        next_pos = example['cloth'].pred_pos

        if self.mesh_distance is not None:
            # Re-select the globally nearest triangle at the predicted/target
            # state; unlike legacy_plane this does not lag the correspondence.
            distance = self.mesh_distance(next_pos, obstacle_next_pos, obstacle_faces)['signed_distance']
            return (self.eps - distance).clamp_min(0).pow(3).sum()

        # Find correspondences in current step
        obstacle_face_curr_pos = gather(obstacle_curr_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)
        _, nn_idx, _ = knn_points(curr_pos.unsqueeze(0), obstacle_face_curr_pos.unsqueeze(0),
                                  return_nn=True)
        nn_idx = nn_idx[0]

        # Compute distances in the new step
        obstacle_face_next_pos = gather(obstacle_next_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)
        obstacle_fn = self.f_normals_f(obstacle_next_pos.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

        nn_points = gather(obstacle_face_next_pos, nn_idx, 0, 1, 1)
        nn_normals = gather(obstacle_fn, nn_idx, 0, 1, 1)

        nn_points = nn_points[:, 0]
        nn_normals = nn_normals[:, 0]
        device = next_pos.device

        distance = ((next_pos - nn_points) * nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(self.eps - distance, torch.FloatTensor([0]).to(device))

        interpenetration = interpenetration.pow(3)
        loss = interpenetration.sum(-1)

        return loss

    def forward(self, sample):
        B = sample.num_graphs
        iter_num = sample['cloth'].iter[0].item()
        weight = self.get_weight(iter_num)

        loss_list = []
        for i in range(B):
            loss_list.append(
                self.calc_loss(sample.get_example(i)))

        loss = sum(loss_list) / B * weight

        return dict(loss=loss)
