"""Opt-in CUDA/data tests: HOOD_INTEGRATION=1 python -m unittest discover -s tests."""
import os
import tempfile
import unittest

import torch
from torch_geometric.data import Batch
from omegaconf import OmegaConf

from utils.arguments import create_modules, create_runner, load_params
from utils.common import add_field_to_pyg_batch, move2device
from utils.experiment import seed_all
from utils.fixed_validation import FixedValidation


@unittest.skipUnless(os.environ.get('HOOD_INTEGRATION') == '1', 'requires local HOOD data and CUDA')
class PhysicsPreservationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_all(57)
        self.modules, self.cfg = load_params('baseline/postcvpr_seed57')
        self.dataloader, _, self.runner, self.aux = create_modules(self.modules, self.cfg)

    def test_observation_preserves_losses_and_optimizer_updates(self):
        # CUDA scatter reductions can differ between identical executions. Use CPU
        # here so exact equality tests the observer rather than GPU reduction order.
        self.cfg.device = 'cpu'
        _, self.runner, self.aux = create_runner(self.modules, self.cfg)
        sample = move2device(Batch.from_data_list([self.dataloader.dataset[0]]), self.cfg.device)
        sample = add_field_to_pyg_batch(sample, 'iter', [5000], 'cloth', reference_key=None)
        _, observed, observed_aux = create_runner(self.modules, self.cfg)
        observed.load_state_dict(self.runner.state_dict())
        observed_aux['optimizer'].load_state_dict(self.aux['optimizer'].state_dict())
        observed_aux['scheduler'].load_state_dict(self.aux['scheduler'].state_dict())
        seed_all(900)
        expected = self.runner(sample.clone(), roll_steps=2, optimizer=self.aux['optimizer'], scheduler=self.aux['scheduler'])
        rows = []
        seed_all(900)
        actual = observed(sample.clone(), roll_steps=2, optimizer=observed_aux['optimizer'],
                          scheduler=observed_aux['scheduler'], step_callback=lambda i, *args: rows.append(i))
        self.assertEqual(rows, [0, 1])
        self.assertEqual(expected, actual)
        for key, value in self.runner.state_dict().items():
            self.assertTrue(torch.equal(value, observed.state_dict()[key]), key)

    def test_fixed_validation_is_repeatable_and_does_not_mutate_training(self):
        self.cfg.validation.n_steps = 2
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            (Path(folder) / 'validation').mkdir()
            validator = FixedValidation(self.modules, self.cfg, folder)
            before = {k: v.clone() for k, v in self.runner.state_dict().items()}
            first = validator.evaluate(self.runner, 0)
            second = validator.evaluate(self.runner, 1)
            self.assertEqual(first, second)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, self.runner.state_dict()[key]), key)
            self.assertEqual(self.runner.cloth_obj.cache, {})


if __name__ == '__main__':
    unittest.main()
