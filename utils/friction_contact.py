"""Frozen closest-point kinematics and regularized Coulomb friction primitives.

Distances are in meters, loads in N, and displacement potentials in J.
These helpers do not perform collision correction or guarantee static sticking.
"""
import math
import torch


@torch.no_grad()
def contact_kinematics(position, body, body_next, faces, query):
    """Track one body material point, with face/barycentric coordinates frozen at n."""
    geometry = query(position.detach(), body.detach(), faces)
    triangles = body[faces[geometry['face_index']]]
    triangles_next = body_next[faces[geometry['face_index']]]
    bary = geometry['barycentric']
    body_delta = ((triangles_next - triangles) * bary[..., None]).sum(1)
    separation = position.detach() - geometry['closest_point']
    length = torch.linalg.vector_norm(separation, dim=-1, keepdim=True)
    face_normal = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0])
    face_normal = face_normal / face_normal.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    # Projectors are invariant to normal sign. At a nonzero edge/vertex distance,
    # use the distance direction, not an arbitrary incident triangle's normal.
    normal = torch.where(length > 1e-10, separation / length.clamp_min(1e-30), face_normal)
    if not torch.isfinite(normal).all() or (normal.norm(dim=-1) < .99).any():
        raise ValueError('Undefined friction tangent at a degenerate zero-distance contact')
    return {'distance': geometry['signed_distance'], 'normal': normal,
            'body_delta': body_delta}


def tangential_displacement(prediction, position, kinematics):
    relative = prediction - position.detach() - kinematics['body_delta']
    normal = kinematics['normal']
    return relative - (relative * normal).sum(-1, keepdim=True) * normal


def relaxed_normal_load(distance, area, dhat, stiffness, tau):
    """-dE/dd for the existing area relaxed barrier, including its quadratic branch."""
    if not 0 < tau < 1 or dhat <= 0 or stiffness <= 0:
        raise ValueError('Invalid relaxed barrier parameters')
    s = distance / dhat
    safe = s.clamp_min(tau)
    derivative = torch.log(safe) + 1 - 1 / safe
    extended = math.log(tau) + 1 - 1 / tau + (1 / tau + 1 / tau**2) * (s - tau)
    derivative = torch.where(s < tau, extended, derivative)
    return torch.where(s >= 1, torch.zeros_like(s), -area * stiffness * derivative / 2)


def smooth_slip_potential(slip, a):
    """f0(r)-f0(0), so zero slip logs zero without changing frozen-load forces.

    In the small-slip branch use r^2 directly, avoiding a norm Hessian at zero.
    """
    if not torch.isfinite(torch.as_tensor(a)).all() or (torch.as_tensor(a) <= 0).any():
        raise ValueError('Positive finite velocity threshold times timestep required')
    r2 = slip.square().sum(-1)
    r = torch.linalg.vector_norm(slip, dim=-1)
    return torch.where(r < a, r2 / a - r2 * r / (3 * a**2), r - a / 3)
