import random
from collections import defaultdict
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from utils.experiment import seed_all
from utils.resume import (ResumableSampler, loader_iterator, restore_rng, restore_training,
                          save_recorded_checkpoint)


class RandomDataset:
    def __init__(self):
        self.loader = SimpleNamespace(garment_builder=SimpleNamespace(garments_dict={'cloth': {'coarse_edges': {0: [1, 2]}}}))
    def __len__(self):
        return 11
    def __getitem__(self, index):
        return torch.tensor([index / 11, random.random(), np.random.rand(), float(torch.rand(()))])


class ToyRunner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 1))
        self.register_buffer('normalizer_count', torch.zeros(()))
        self.cloth_obj = SimpleNamespace(cache=defaultdict(dict))
        self.mcfg = SimpleNamespace(device='cpu')


class ResumeTest(unittest.TestCase):
    def test_fresh_loader_matches_default_shuffle_and_augmentation_two_epochs(self):
        seed_all(57)
        standard = []
        for _ in range(2):
            standard.extend(DataLoader(RandomDataset(), batch_size=1, shuffle=True, num_workers=0))
        expected_rng = torch.get_rng_state()
        seed_all(57)
        sampler, actual = ResumableSampler(11), []
        for epoch in range(2):
            if epoch:
                sampler.next_epoch()
            for sample in loader_iterator(RandomDataset(), sampler):
                actual.append(sample); sampler.commit()
        for x, y in zip(standard, actual):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_checkpoint_mid_epoch_restores_optimizer_rng_normalizer_and_cache(self):
        def initialize():
            runner = ToyRunner()
            optimizer = torch.optim.Adam(runner.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: .9 ** n)
            return runner, {'optimizer': optimizer, 'scheduler': scheduler}
        def batches(runner, aux, sampler, dataset, start, end, updates, restoring=False):
            iterator = loader_iterator(dataset, sampler, restoring)
            losses = []
            for step in range(start, end):
                sample = next(iterator)
                runner.cloth_obj.cache.setdefault('first_value', sample.clone())
                for _ in range(1 + step // 2):
                    x = sample + .01 * runner.cloth_obj.cache['first_value']
                    loss = runner.model(x).square().sum()
                    aux['optimizer'].zero_grad(); loss.backward(); aux['optimizer'].step(); aux['scheduler'].step()
                    runner.normalizer_count += 1
                    losses.append(float(loss.detach())); updates += 1
                sampler.commit()
            return losses, updates
        seed_all(57)
        dataset = RandomDataset(); runner, aux = initialize(); sampler = ResumableSampler(len(dataset))
        _, updates = batches(runner, aux, sampler, dataset, 0, 2, 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'checkpoints').mkdir()
            run = SimpleNamespace(root=root, metadata={'git_commit': 'fixture'})
            progress = dict(global_step=2, optimizer_step_count=updates, epoch=0, elapsed_s=1., best_validation=None)
            path = save_recorded_checkpoint(runner, aux, OmegaConf.create({'seed': 57}), run, sampler, dataset, progress, 'data')
            checkpoint = torch.load(path, map_location='cpu')
            # Continuing the SAME iterator must not consume another base seed.
            expected, count = batches(runner, aux, sampler, dataset, 2, 4, updates, restoring=True)
            expected_state = {k: v.clone() for k, v in runner.state_dict().items()}
            restored, restored_aux = initialize(); restored_dataset = RandomDataset()
            restore_training(checkpoint, restored, restored_aux, restored_dataset, 'data', 'fixture')
            self.assertIsInstance(restored.cloth_obj.cache, defaultdict)
            self.assertEqual(restored.cloth_obj.cache['unseen_garment'], {})
            restored_sampler = ResumableSampler(len(restored_dataset), checkpoint['sampler'])
            restore_rng(checkpoint['rng_state'])
            actual, resumed_count = batches(restored, restored_aux, restored_sampler, restored_dataset, 2, 4, updates, restoring=True)
            self.assertEqual(expected, actual)
            self.assertEqual(count, resumed_count)
            self.assertEqual(aux['scheduler'].state_dict(), restored_aux['scheduler'].state_dict())
            for k, v in expected_state.items():
                self.assertTrue(torch.equal(v, restored.state_dict()[k]), k)
            self.assertEqual(checkpoint['sampler']['cursor'], 2)
            with self.assertRaisesRegex(ValueError, 'data/assets'):
                restore_training(checkpoint, restored, restored_aux, restored_dataset, 'changed', 'fixture')
            with self.assertRaisesRegex(ValueError, 'commit differs'):
                restore_training(checkpoint, restored, restored_aux, restored_dataset, 'data', 'different')


if __name__ == '__main__':
    unittest.main()
