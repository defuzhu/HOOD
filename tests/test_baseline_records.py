import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.experiment import CSVLog, Experiment, isolated_rng, seed_all
from utils.fixed_validation import motion_id
from utils.arguments import load_params


class RecordsTest(unittest.TestCase):
    def test_validation_restores_all_random_streams_even_on_failure(self):
        seed_all(57)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        seed_all(57)
        with self.assertRaises(RuntimeError):
            with isolated_rng(123):
                random.random(); np.random.rand(); torch.rand(3)
                if torch.cuda.is_available():
                    torch.rand(3, device='cuda')
                raise RuntimeError('validation failed')
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[0:2], actual[0:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_cuda_random_stream_is_preserved(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        seed_all(57)
        expected = torch.rand(3, device='cuda')
        seed_all(57)
        with isolated_rng(123):
            torch.rand(30, device='cuda')
        self.assertTrue(torch.equal(expected, torch.rand(3, device='cuda')))

    def test_cli_overrides_win_only_when_explicitly_requested(self):
        _, cfg = load_params('baseline/postcvpr_seed57', overrides=OmegaConf.create({'experiment': {'max_iter': 2}}))
        self.assertEqual(cfg.experiment.max_iter, 2)
        _, original = load_params('postcvpr')
        self.assertEqual(original.experiment.max_iter, 60000)

    def test_motion_identity_ignores_shape_and_garment(self):
        self.assertEqual(motion_id('tshirt_shape00_104_54'), motion_id('dress_shape16_104_54'))
        self.assertNotEqual(motion_id('tshirt_shape00_104_54'), motion_id('tshirt_shape00_104_11'))

    def test_unique_run_directories_and_named_losses(self):
        import os
        cfg = OmegaConf.create({'seed': 57, 'recording': {'require_clean': False},
                                'experiment': {'name': 'test_baseline'}})
        with tempfile.TemporaryDirectory() as folder:
            a = Experiment(cfg, os.environ['HOOD_PROJECT'], folder)
            b = Experiment(cfg, os.environ['HOOD_PROJECT'], folder)
            self.assertNotEqual(a.root, b.root)
            a.substeps.write({'global_step': 1, 'inertia_loss': 0.2, 'gravitational_energy_loss': -0.1})
            a.finish('completed'); b.finish('completed')
            with (a.root / 'logs/train_substeps.csv').open() as stream:
                row = next(csv.DictReader(stream))
                self.assertEqual(float(row['gravitational_energy_loss']), -0.1)
            self.assertTrue((a.root / 'config.yaml').is_file())


if __name__ == '__main__':
    unittest.main()
