import unittest
import torch

from utils.contact_potential import area_relaxed_barrier, relaxed_log_shape, vertex_rest_areas


class PotentialTest(unittest.TestCase):
    def test_finite_repulsion_inside_surface_and_outside(self):
        s = torch.tensor([-10., -.2, 0., .05, .1, .5, .99, 1., 2.], dtype=torch.double, requires_grad=True)
        energy = relaxed_log_shape(s)
        grad, = torch.autograd.grad(energy.sum(), s)
        self.assertTrue(torch.isfinite(energy).all())
        self.assertTrue((energy >= 0).all())
        self.assertTrue((grad[:7] < 0).all())
        self.assertTrue((grad[7:] == 0).all())

    def test_continuation_c2_join_and_gradcheck(self):
        s = torch.tensor([.1 - 1e-8, .1 + 1e-8], dtype=torch.double, requires_grad=True)
        value = relaxed_log_shape(s)
        grad, = torch.autograd.grad(value.sum(), s, create_graph=True)
        hess, = torch.autograd.grad(grad.sum(), s)
        for x in (value, grad, hess):
            torch.testing.assert_close(x[0], x[1], rtol=1e-5, atol=1e-6)
        points = torch.tensor([-.3, 0., .05, .4, 1.2], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(relaxed_log_shape, (points,)))

    def test_constant_gap_energy_invariant_under_triangle_subdivision(self):
        v = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.double)
        f = torch.tensor([[0, 1, 2]])
        refined = torch.cat((v, v.mean(0, keepdim=True)))
        rf = torch.tensor([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
        energies = []
        for vertices, faces in ((v, f), (refined, rf)):
            areas = vertex_rest_areas(vertices, faces)
            torch.testing.assert_close(areas.sum(), torch.tensor(.5, dtype=torch.double))
            energies.append(area_relaxed_barrier(torch.full((len(vertices),), .001, dtype=torch.double),
                                                 areas, .002, 100.))
        torch.testing.assert_close(*energies)

    def test_area_scaling_and_distance_scale(self):
        d = torch.tensor([.0005, -.0001], dtype=torch.double)
        area = torch.tensor([.1, .2], dtype=torch.double)
        e = area_relaxed_barrier(d, area, .002, 100.)
        torch.testing.assert_close(area_relaxed_barrier(d, area * 3, .002, 100.), e * 3)
        # Uniform geometric scaling: area scales c^2, layer thickness c, total c^3.
        torch.testing.assert_close(area_relaxed_barrier(d * 2, area * 4, .004, 100.), e * 8)

    def test_full_criterion_backpropagates_and_ramp_applies_once(self):
        from torch_geometric.data import HeteroData, Batch
        from tests.test_contact_geometry import tetrahedron
        from criterions.postcvpr.collision_penalty import Criterion
        v, f = tetrahedron()
        for potential in ('area_cubic', 'area_relaxed_barrier'):
            e = HeteroData()
            e['obstacle'].pos = e['obstacle'].target_pos = v
            e['obstacle'].faces_batch = f.T
            e['cloth'].pos = torch.tensor([[.2, .2, .001], [.3, .2, .001], [.2, .3, .001]])
            e['cloth'].pred_pos = e['cloth'].pos.clone().requires_grad_()
            e['cloth'].rest_pos = e['cloth'].pos.clone()
            e['cloth'].faces_batch = torch.tensor([[0], [1], [2]])
            e['cloth'].iter = torch.tensor([0])
            c = Criterion(5000., 10000., 0, 100, eps=.002, contact_geometry='signed_mesh', contact_potential=potential)
            batch = Batch.from_data_list([e])
            loss = c(batch)['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(e['cloth'].pred_pos.grad).all())
            self.assertTrue((e['cloth'].pred_pos.grad[:, 2] > 0).all())
            batch['cloth'].iter[:] = 100
            torch.testing.assert_close(c(batch)['loss'], loss.detach() * 2)


if __name__ == '__main__':
    unittest.main()
