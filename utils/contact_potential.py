"""Area quadrature and a finite continuation of the book's log barrier.

This is a relaxed penalty for neural predictions that can have d <= 0. It is
NOT an interior-point barrier and does not guarantee nonintersection.
"""
import math

import torch


def vertex_rest_areas(rest_pos, faces):
    """Barycentric dual areas: each rest triangle contributes one third to each vertex."""
    with torch.no_grad():
        triangles = rest_pos.detach()[faces]
        area = torch.linalg.vector_norm(torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), dim=-1) / 2
        result = rest_pos.new_zeros(len(rest_pos))
        result.scatter_add_(0, faces.reshape(-1), (area[:, None] / 3).expand(-1, 3).reshape(-1))
    return result


def relaxed_log_shape(s, transition_ratio=0.1):
    """phi(s)=(s-1)log(s) on [tau,1), zero above 1; quadratic below tau.

    The continuation matches value, first and second derivatives at tau.
    At the activation boundary s=1 the full function is C1, as in the book.
    """
    tau = float(transition_ratio)
    if not 0 < tau < 1:
        raise ValueError('barrier_transition_ratio must be between zero and one')
    # Clamp only the inactive logarithm branch. The active continuation uses s,
    # so negative distances retain a nonzero outward descent direction.
    safe = s.clamp_min(tau)
    log_branch = (safe - 1) * torch.log(safe)
    value = (tau - 1) * math.log(tau)
    derivative = math.log(tau) + 1 - 1 / tau
    curvature = 1 / tau + 1 / tau ** 2
    delta = s - tau
    continuation = value + derivative * delta + .5 * curvature * delta.square()
    return torch.where(s >= 1, torch.zeros_like(s),
                       torch.where(s < tau, continuation, log_branch))


def area_relaxed_barrier(distance, area, dhat, stiffness, transition_ratio=0.1):
    """Sum A_i * dhat * kappa/2 * phi(d_i/dhat), with SI units giving joules."""
    if not math.isfinite(dhat) or dhat <= 0 or not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError('Barrier distance and stiffness must be positive and finite')
    return (area * dhat * stiffness / 2 * relaxed_log_shape(distance / dhat, transition_ratio)).sum()
