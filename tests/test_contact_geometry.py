import math
import unittest

import torch
from pytorch3d.ops import knn_points
from torch_geometric.data import HeteroData

from criterions.postcvpr.collision_penalty import Criterion
from utils.cloth_and_material import FaceNormals
from utils.common import gather
from utils.contact_geometry import ClosedMeshDistance, closest_points_on_triangles


def tetrahedron(dtype=torch.float32, device='cpu'):
    v = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype, device=device)
    f = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device=device)
    return v, f


class ContactGeometryTest(unittest.TestCase):
    def test_face_edge_vertex_and_degenerate_triangle_closest_points(self):
        t = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]).repeat(5, 1, 1)
        t[3] = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        t[4] = 0
        p = torch.tensor([[.2, .3, .4], [.5, -.2, .3], [-.1, -.2, .3], [.4, .2, 0.], [.1, .2, .3]], requires_grad=True)
        q, bary = closest_points_on_triangles(p, t)
        expected = torch.tensor([[.2, .3, 0.], [.5, 0., 0.], [0., 0., 0.], [.4, 0., 0.], [0., 0., 0.]])
        torch.testing.assert_close(q, expected)
        self.assertTrue((bary >= 0).all())
        torch.testing.assert_close(bary.sum(-1), torch.ones(5))
        (p - q).square().sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_signed_distance_inside_outside_and_deep_inside(self):
        v, f = tetrahedron()
        p = torch.tensor([[.1, .1, .1], [.2, .2, -.3], [.5, -.2, -.3], [-.1, -.2, -.3], [.25, .25, .25]])
        result = ClosedMeshDistance(point_chunk=2, face_chunk=2)(p, v, f)
        expected = torch.tensor([-.1, .3, math.sqrt(.13), math.sqrt(.14), -.25 / math.sqrt(3)])
        torch.testing.assert_close(result['signed_distance'], expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result['winding_number'].float(), torch.tensor([1., 0., 0., 0., 1.]), atol=1e-6, rtol=1e-6)

    def test_global_triangle_search_defeats_nearest_centroid(self):
        va, fa = tetrahedron()
        va *= 100
        va[3, 2] = -100
        fa = fa[:, [0, 2, 1]]
        vb, fb = tetrahedron()
        vb = vb * .1 + torch.tensor([.1, .1, .2])
        v, f = torch.cat((va, vb)), torch.cat((fa, fb + 4))
        p = torch.tensor([[.1, .1, .0005]])
        centroids = v[f].mean(1)
        old_face = torch.cdist(p, centroids).argmin(-1)
        self.assertGreaterEqual(int(old_face), 4)
        result = ClosedMeshDistance()(p, v, f)
        self.assertLess(int(result['face_index']), 4)
        torch.testing.assert_close(result['signed_distance'], torch.tensor([.0005]))

    def test_sign_does_not_depend_on_global_face_orientation(self):
        v, f = tetrahedron()
        p = torch.tensor([[.1, .1, .1], [.2, .2, -.3]])
        query = ClosedMeshDistance()
        a, b = query(p, v, f), query(p, v, f[:, [0, 2, 1]])
        torch.testing.assert_close(a['signed_distance'], b['signed_distance'])

    def test_signed_distance_gradient_finite_difference(self):
        v, f = tetrahedron(torch.float64)
        p = torch.tensor([[.2, .2, -.3], [.12, .24, .24]], dtype=torch.float64, requires_grad=True)
        query = ClosedMeshDistance()
        self.assertTrue(torch.autograd.gradcheck(lambda x: query(x, v, f)['signed_distance'], (p,), eps=1e-6, atol=1e-5))

    def test_contact_repels_even_exactly_on_surface(self):
        v, f = tetrahedron()
        for faces in [f, f[:, [0, 2, 1]]]:
            p = torch.tensor([[.2, .2, 0.]], requires_grad=True)
            d = ClosedMeshDistance()(p, v, faces)['signed_distance']
            d.sum().backward()
            torch.testing.assert_close(p.grad, torch.tensor([[0., 0., -1.]]))

    def test_open_mesh_is_rejected(self):
        v, f = tetrahedron()
        with self.assertRaisesRegex(ValueError, 'closed'):
            ClosedMeshDistance()(torch.tensor([[.1, .1, .1]]), v, f[:3])

    def test_default_criterion_is_exactly_legacy_formula(self):
        v, f = tetrahedron()
        e = HeteroData()
        e['obstacle'].pos, e['obstacle'].target_pos = v, v + .01
        e['obstacle'].faces_batch = f.T
        e['cloth'].pos = torch.tensor([[.2, .2, -.001], [.1, .1, .1]])
        e['cloth'].pred_pos = e['cloth'].pos + .0002
        centers = gather(v, f, 0, 1, 1).mean(-2)
        _, idx, _ = knn_points(e['cloth'].pos[None], centers[None], return_nn=True)
        idx = idx[0, :, 0]
        next_centers = gather(e['obstacle'].target_pos, f, 0, 1, 1).mean(-2)
        normals = FaceNormals()(e['obstacle'].target_pos[None], f[None])[0]
        d = ((e['cloth'].pred_pos - next_centers[idx]) * normals[idx]).sum(-1)
        expected = torch.maximum(.001 - d, torch.tensor([0.])).pow(3).sum()
        actual = Criterion(1000, 100000, 50000, 100000).calc_loss(e)
        self.assertTrue(torch.equal(expected, actual))

    def test_new_criterion_uses_target_body_cubic_and_backpropagates(self):
        v, f = tetrahedron()
        e = HeteroData()
        e['obstacle'].pos, e['obstacle'].target_pos = v - 2, v
        e['obstacle'].faces_batch = f.T
        e['cloth'].pos = torch.tensor([[.2, .2, -.5]])
        e['cloth'].pred_pos = torch.tensor([[.2, .2, -.0005]], requires_grad=True)
        criterion = Criterion(1000, 100000, 50000, 100000, eps=.002, contact_geometry='signed_mesh')
        loss = criterion.calc_loss(e)
        torch.testing.assert_close(loss, torch.tensor(.0015 ** 3))
        loss.backward()
        self.assertGreater(float(e['cloth'].pred_pos.grad[0, 2]), 0)  # descent moves outward (-z)
        self.assertTrue(torch.isfinite(e['cloth'].pred_pos.grad).all())
        e['obstacle'].pos = v + 10
        self.assertEqual(float(loss), float(criterion.calc_loss(e)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_cpu_distances_and_gradients(self):
        v, f = tetrahedron()
        p = torch.tensor([[.12, .22, .3], [-.1, -.2, -.3]], requires_grad=True)
        cpu = ClosedMeshDistance()(p, v, f)['signed_distance']
        cpu.sum().backward()
        pg = p.detach().cuda().requires_grad_()
        gpu = ClosedMeshDistance()(pg, v.cuda(), f.cuda())['signed_distance']
        gpu.sum().backward()
        torch.testing.assert_close(cpu, gpu.cpu())
        torch.testing.assert_close(p.grad, pg.grad.cpu())
