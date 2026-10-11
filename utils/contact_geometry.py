"""Point-to-closed-mesh distances for garment/body contact (not CCD).

Global nearest-face search uses the installed PyTorch3D kernel. Closest points
are recomputed in torch so gradients reach predicted garment vertices. Signs
come from generalized winding numbers, NOT a nearest-face-plane heuristic.
"""
import hashlib
import math

import numpy as np
import torch
from pytorch3d import _C


def closest_points_on_triangles(points, triangles, min_triangle_area=1e-12):
    """One point per triangle. Return closest points and barycentric coordinates.

    Test the constrained plane projection and all three finite segments. This
    includes face, edge and vertex regions; degenerate triangles use segments.
    """
    a, b, c = triangles.unbind(1)
    ab, ac, ap = b - a, c - a, points - a
    dot = lambda u, v: (u * v).sum(-1)
    aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    det = aa * cc - bb.square()
    area_ok = torch.linalg.cross(ab, ac).square().sum(-1) > (2 * min_triangle_area) ** 2
    # A finite denominator also on degenerate triangles avoids masked NaN gradients.
    safe_det = torch.where(area_ok & (det > 0), det, torch.ones_like(det))
    u = (cc * dot(ap, ab) - bb * dot(ap, ac)) / safe_det
    v = (aa * dot(ap, ac) - bb * dot(ap, ab)) / safe_det
    bary_plane = torch.stack((1 - u - v, u, v), -1)
    plane = a + u[:, None] * ab + v[:, None] * ac
    plane_ok = area_ok & (det > 0) & (bary_plane >= 0).all(-1)
    candidates, bary = [plane], [bary_plane]
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        start, end = triangles[:, i], triangles[:, j]
        edge = end - start
        length_sq = dot(edge, edge)
        denominator = torch.where(length_sq > 0, length_sq, torch.ones_like(length_sq))
        t = (dot(points - start, edge) / denominator).clamp(0, 1)
        candidates.append(start + t[:, None] * edge)
        basis_i = torch.nn.functional.one_hot(torch.tensor(i, device=points.device), 3).to(points.dtype)
        basis_j = torch.nn.functional.one_hot(torch.tensor(j, device=points.device), 3).to(points.dtype)
        bary.append((1 - t[:, None]) * basis_i + t[:, None] * basis_j)
    candidates = torch.stack(candidates, 1)
    squared = (points[:, None] - candidates).square().sum(-1)
    squared[:, 0] = torch.where(plane_ok, squared[:, 0], torch.full_like(squared[:, 0], float('inf')))
    selection = squared.argmin(-1)
    rows = torch.arange(len(points), device=points.device)
    return candidates[rows, selection], torch.stack(bary, 1)[rows, selection]


@torch.no_grad()
def winding_numbers(points, triangles, point_chunk=256, face_chunk=2048):
    """Sum oriented triangle solid angles / 4pi, in float64 with bounded memory.

    Classification is detached. For closed embedded surfaces, abs(w)>0.5
    means inside. Self-intersecting surfaces retain winding semantics; this is
    not the distance to the exterior boundary of a repaired union of volumes.
    """
    output = torch.zeros(len(points), dtype=torch.float64, device=points.device)
    for ps in range(0, len(points), point_chunk):
        p = points[ps:ps + point_chunk].double()
        total = torch.zeros(len(p), dtype=torch.float64, device=points.device)
        for fs in range(0, len(triangles), face_chunk):
            t = triangles[fs:fs + face_chunk].double()
            a = t[None, :, 0] - p[:, None]
            b = t[None, :, 1] - p[:, None]
            c = t[None, :, 2] - p[:, None]
            na, nb, nc = (torch.linalg.vector_norm(v, dim=-1) for v in (a, b, c))
            numerator = (a * torch.linalg.cross(b, c)).sum(-1)
            denominator = na * nb * nc + (a * b).sum(-1) * nc + (b * c).sum(-1) * na + (c * a).sum(-1) * nb
            # Undefined solid angles at zero-area faces contribute zero.
            valid = torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]).square().sum(-1) > 0
            angle = torch.where(valid[None], 2 * torch.atan2(numerator, denominator), 0.)
            total += angle.sum(-1)
        output[ps:ps + len(p)] = total / (4 * math.pi)
    return output


class ClosedMeshDistance:
    def __init__(self, point_chunk=256, face_chunk=2048, min_triangle_area=1e-12):
        if point_chunk <= 0 or face_chunk <= 0 or min_triangle_area <= 0:
            raise ValueError('Contact chunks and min_triangle_area must be positive')
        self.point_chunk, self.face_chunk = point_chunk, face_chunk
        self.min_triangle_area = min_triangle_area
        self._checked_topologies = set()

    def check_topology(self, faces):
        array = faces.detach().cpu().numpy()
        key = hashlib.sha256(array.tobytes()).hexdigest()
        if key in self._checked_topologies:
            return
        edges = np.concatenate([array[:, [0, 1]], array[:, [1, 2]], array[:, [2, 0]]])
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
        balances = np.bincount(inverse, weights=direction, minlength=len(counts))
        if np.any(counts != 2) or np.any(balances != 0):
            raise ValueError('signed_mesh contact requires a closed, consistently oriented body mesh; '
                             'open/nonmanifold meshes need a different sign construction')
        self._checked_topologies.add(key)

    def __call__(self, points, vertices, faces):
        if len(points) == 0 or len(faces) == 0:
            raise ValueError('Contact requires nonempty points and faces')
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError('Body faces must have shape [F,3]')
        self.check_topology(faces)
        triangles = vertices[faces]
        with torch.no_grad():
            if not (torch.isfinite(points).all() and torch.isfinite(vertices).all()):
                raise FloatingPointError('Non-finite contact geometry')
            # All faces participate. A nearest-centroid shortlist is NOT used.
            first = torch.zeros(1, dtype=torch.long, device=points.device)
            _, indices = _C.point_face_dist_forward(
                points.detach().float().contiguous(), first,
                triangles.detach().float().contiguous(), first, len(points), self.min_triangle_area)
            # Outside the body AABB is certainly outside a closed body's volume.
            in_box = ((points >= vertices.min(0).values) & (points <= vertices.max(0).values)).all(-1)
            winding = torch.zeros(len(points), dtype=torch.float64, device=points.device)
            winding[in_box] = winding_numbers(points[in_box], triangles, self.point_chunk, self.face_chunk)
            inside = winding.abs() > 0.5
        selected = triangles[indices]
        closest, barycentric = closest_points_on_triangles(points, selected, self.min_triangle_area)
        unsigned = torch.linalg.vector_norm(points - closest, dim=-1)
        sign = torch.where(inside, -torch.ones_like(unsigned), torch.ones_like(unsigned))
        signed = sign * unsigned
        # At exactly zero distance norm() has zero derivative. Use an outward
        # face-normal subgradient there so the cubic clearance penalty can repel.
        normal = torch.linalg.cross(selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0])
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-30)
        with torch.no_grad():
            volume6 = (triangles[:, 0] * torch.linalg.cross(triangles[:, 1], triangles[:, 2])).sum()
            orientation = torch.where(volume6 >= 0, 1., -1.)
        local = ((points - closest) * normal.detach() * orientation).sum(-1)
        signed = torch.where(unsigned > 1e-10, signed, local)
        return {'signed_distance': signed, 'unsigned_distance': unsigned,
                'closest_point': closest, 'barycentric': barycentric,
                'face_index': indices, 'inside': inside, 'winding_number': winding}
