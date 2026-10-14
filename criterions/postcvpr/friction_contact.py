"""Opt-in friction ablations; the legacy friction criterion stays unchanged."""
import math
import weakref
import torch
from torch import nn
from utils.contact_potential import vertex_rest_areas
from utils.friction_contact import (contact_kinematics, tangential_displacement,
                                    relaxed_normal_load, smooth_slip_potential)


class Criterion(nn.Module):
    name = 'friction_energy'

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.weight, self.mu, self.g = config.weight, config.mu, config.g
        self.velocity_epsilon = config.velocity_epsilon
        if self.mode not in ('geometry_mg', 'lagged_coulomb'):
            raise ValueError('Unknown friction mode')
        if not all(math.isfinite(v) and v >= 0 for v in (self.weight, self.mu, self.g)):
            raise ValueError('Friction weight/mu/g must be finite and nonnegative')
        if not math.isfinite(self.velocity_epsilon) or self.velocity_epsilon <= 0:
            raise ValueError('velocity_epsilon must be positive, in m/s')
        self._contact_ref = None
        self.record_diagnostics = False
        self.last_diagnostics = {}

    def bind_contact(self, contact):
        # One source of truth for geometry, activation distance and stiffness schedule.
        if contact is None or contact.contact_geometry != 'signed_mesh':
            raise ValueError('New friction requires signed_mesh collision geometry')
        if self.mode == 'lagged_coulomb' and contact.contact_potential != 'area_relaxed_barrier':
            raise ValueError('lagged_coulomb requires area_relaxed_barrier contact potential')
        self._contact_ref = weakref.ref(contact)

    def forward(self, sample):
        contact = self._contact_ref() if self._contact_ref else None
        if contact is None:
            raise RuntimeError('Bind the collision criterion before computing friction')
        total, diagnostics = [], []
        for index in range(sample.num_graphs):
            example = sample.get_example(index)
            cloth, body = example['cloth'], example['obstacle']
            kin = contact_kinematics(cloth.pos, body.pos, body.target_pos,
                                     body.faces_batch.T, contact.mesh_distance)
            slip = tangential_displacement(cloth.pred_pos, cloth.pos, kin)
            r = torch.linalg.vector_norm(slip, dim=-1)
            dt = cloth.timestep.reshape(-1)[0].detach()
            if not torch.isfinite(dt) or dt <= 0:
                raise ValueError('Friction requires a positive finite timestep')
            active = kin['distance'] < contact.eps
            with torch.no_grad():
                if self.mode == 'lagged_coulomb':
                    stiffness = contact.barrier_stiffness * contact.get_weight(
                        cloth.iter.reshape(-1)[0].item()) / contact.weight_start
                    area = vertex_rest_areas(cloth.rest_pos, cloth.faces_batch.T)
                    load = relaxed_normal_load(kin['distance'], area, contact.eps,
                                               stiffness, contact.barrier_transition_ratio)
                else:
                    load = cloth.v_mass.reshape(-1).detach() * self.g * active
            if self.mode == 'geometry_mg':
                # Preserve the legacy mg and floor-direction approximation for B.
                # Only geometry, timing, unit projection and contact support change.
                cosine = torch.nn.functional.cosine_similarity(
                    slip, slip.new_tensor([0., 0., 1.])[None], dim=-1).abs()
                floor_factor = ((1 - cosine.square()).abs() + 1e-4).sqrt()
                per_vertex = self.mu * load * floor_factor * r
            else:
                per_vertex = self.mu * load * smooth_slip_potential(slip, self.velocity_epsilon * dt)
            total.append(per_vertex.sum() * self.weight)
            if self.record_diagnostics:
                with torch.no_grad():
                    count = active.sum().clamp_min(1)
                    diagnostics.append({
                        'friction_active_fraction': active.float().mean(),
                        'friction_load_sum_n': load.sum(),
                        'friction_load_max_n': load.max(),
                        'friction_inside_fraction': (kin['distance'] < 0).float().mean(),
                        'friction_slip_speed_active_mean_m_s': (r.detach() * active).sum() / (dt * count),
                        'friction_low_speed_active_fraction': ((r.detach() < self.velocity_epsilon * dt) & active).sum() / count,
                    })
        self.last_diagnostics = ({key: torch.stack([d[key] for d in diagnostics]).mean()
                                  for key in diagnostics[0]} if diagnostics else {})
        return {'loss': torch.stack(total).mean()}
