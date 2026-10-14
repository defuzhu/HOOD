import unittest
import torch
from torch_geometric.data import HeteroData, Batch
from criterions.postcvpr.friction_energy import Config, create, Criterion as Legacy
from criterions.postcvpr.collision_penalty import Criterion as Contact
from utils.contact_potential import area_relaxed_barrier
from utils.friction_contact import (relaxed_normal_load, smooth_slip_potential,
                                    contact_kinematics, tangential_displacement)
from utils.contact_geometry import ClosedMeshDistance
from test_contact_geometry import tetrahedron


def example():
    e = HeteroData()
    v, f = tetrahedron()
    e['obstacle'].pos = v
    e['obstacle'].prev_pos = v - .5  # Deliberately different old frame.
    e['obstacle'].target_pos = v + torch.tensor([.002, 0., 0.])
    e['obstacle'].faces_batch = f.T
    x = torch.tensor([[.2, .2, -.001], [.3, .2, -.001], [.2, .3, -.001]])
    e['cloth'].pos = x
    e['cloth'].pred_pos = (x + torch.tensor([.003, 0., 0.])).requires_grad_()
    e['cloth'].rest_pos = x.clone()
    e['cloth'].v_mass = torch.full((3, 1), .001)
    e['cloth'].faces_batch = torch.tensor([[0], [1], [2]])
    e['cloth'].iter = torch.tensor([60000])
    e['cloth'].timestep = torch.tensor([[1/30]])
    return e


def contact():
    return Contact(5000., 500000., 50000, 100000, eps=.002,
                   contact_geometry='signed_mesh', contact_potential='area_relaxed_barrier')


class FrictionTest(unittest.TestCase):
    def test_load_matches_energy_derivative_and_area_stiffness_scaling(self):
        d = torch.tensor([-.002, 0., .0001, .0002, .001, .002, .004], dtype=torch.double, requires_grad=True)
        area = torch.full_like(d, .001)
        grad, = torch.autograd.grad(area_relaxed_barrier(d, area, .002, 94.8), d)
        load = relaxed_normal_load(d, area, .002, 94.8, .1)
        torch.testing.assert_close(load, -grad)
        self.assertTrue((load >= 0).all())
        torch.testing.assert_close(relaxed_normal_load(d, area*2, .002, 94.8*3, .1), load*6)

    def test_coulomb_force_bound_direction_zero_and_timestep_invariance(self):
        for dt in (1/30, 1/60, 1/3):
            slip = torch.tensor([[0., 0., 0.], [.001, 0., 0.], [.02, 0., 0.]], dtype=torch.double) * dt
            slip.requires_grad_()
            energy = .5 * 2 * smooth_slip_potential(slip, .01 * dt)
            grad, = torch.autograd.grad(energy.sum(), slip)
            torch.testing.assert_close(grad[:, 0], torch.tensor([0., .19, 1.], dtype=torch.double))
            self.assertTrue((grad[:, 1:] == 0).all())
            self.assertEqual(energy[0].item(), 0.)
            self.assertTrue((grad.norm(dim=-1) <= 1 + 1e-12).all())

    def test_barycentric_motion_common_translation_and_frozen_inputs(self):
        e = example(); c, b = e['cloth'], e['obstacle']
        kin = contact_kinematics(c.pos, b.pos, b.target_pos, b.faces_batch.T, ClosedMeshDistance())
        predicted = (c.pos + torch.tensor([.002, 0., .001])).requires_grad_()
        slip = tangential_displacement(predicted, c.pos, kin)
        torch.testing.assert_close(slip, torch.zeros_like(slip), atol=2e-8, rtol=0)
        # Deform only one body vertex: closest-point barycentric weight is 0.2,
        # not the 1/3 weight of the triangle center.
        target = b.pos.clone(); target[1, 0] += .01
        kin = contact_kinematics(c.pos, b.pos, target, b.faces_batch.T, ClosedMeshDistance())
        self.assertAlmostEqual(kin['body_delta'][0, 0].item(), .002, places=6)
        self.assertFalse(kin['normal'].requires_grad)

    def test_full_criteria_gradients_batch_reduction_diagnostics_and_no_contact(self):
        for mode in ('geometry_mg', 'lagged_coulomb'):
            col = contact(); friction = create(Config(mode=mode, mu=.5))
            friction.bind_contact(col)
            e = example(); batch = Batch.from_data_list([e])
            a = friction(batch)['loss']
            a.backward()
            self.assertTrue(torch.isfinite(e['cloth'].pred_pos.grad).all())
            self.assertGreater(e['cloth'].pred_pos.grad[:, 0].sum(), 0)
            self.assertTrue(torch.allclose(e['cloth'].pred_pos.grad[:, 2], torch.zeros(3)))
            friction.record_diagnostics = True
            b = friction(Batch.from_data_list([example(), example()]))['loss']
            torch.testing.assert_close(a.detach(), b.detach())
            self.assertTrue(friction.last_diagnostics)
            self.assertTrue(all(not x.requires_grad for x in friction.last_diagnostics.values()))
            e = example(); e['cloth'].pos[:, 2] = -.1
            e['cloth'].pred_pos = (e['cloth'].pos + .001).requires_grad_()
            zero = friction(Batch.from_data_list([e]))['loss']
            zero.backward()
            self.assertEqual(zero.item(), 0.)
            self.assertTrue((e['cloth'].pred_pos.grad == 0).all())

    def test_new_coulomb_rotation_invariance_and_ramp_once(self):
        col = contact(); friction = create(Config(mode='lagged_coulomb'))
        friction.bind_contact(col)
        e = example(); base = friction(Batch.from_data_list([e]))['loss']
        e['cloth'].iter[:] = 50000
        start = friction(Batch.from_data_list([e]))['loss']
        torch.testing.assert_close(base, start * 10.9)
        rotation = torch.tensor([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        e = example()
        for obj in ('cloth', 'obstacle'):
            for key in ('pos', 'prev_pos', 'target_pos', 'pred_pos', 'rest_pos'):
                if key in e[obj]: e[obj][key] = e[obj][key] @ rotation.T
        torch.testing.assert_close(friction(Batch.from_data_list([e]))['loss'], base, atol=1e-7, rtol=1e-5)

    def test_legacy_factory_is_unchanged_and_binding_rejects_incompatible_contact(self):
        e = Batch.from_data_list([example()])
        torch.testing.assert_close(create(Config())(e)['loss'], Legacy(1.)(e)['loss'], rtol=0, atol=0)
        friction = create(Config(mode='lagged_coulomb'))
        with self.assertRaises(ValueError): friction.bind_contact(Contact(1, 1, 0, 1))
        with self.assertRaises(RuntimeError): friction(e)


if __name__ == '__main__':
    unittest.main()
