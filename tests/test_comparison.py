import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from compare_experiments import compare, controls, history, pair_runs


class ComparisonTest(unittest.TestCase):
    def fixture(self, root, name, seed, value):
        run = root / name
        (run / 'logs').mkdir(parents=True)
        (run / 'validation').mkdir()
        (run / 'metadata.json').write_text(json.dumps({'seed': seed, 'status': 'completed'}))
        OmegaConf.save(OmegaConf.create({'model': {'postcvpr': {}}, 'dataloader': {},
                                        'runner': {'postcvpr': {'lr': .001}},
                                        'experiment': {'max_iter': 4, 'n_epochs': 2}}), run / 'config.yaml')
        (run / 'validation_manifest.json').write_text(json.dumps({'evaluation_signature': 'common'}))
        pd.DataFrame([{'global_step': i, 'optimizer_step_count': i,
                       'stretching_energy_loss_mean': 1 / i} for i in range(1, 5)]).to_csv(run / 'logs/train.csv', index=False)
        pd.DataFrame([{'global_step': 4, 'sequence': 'motion', 'garment': 'tshirt', 'frames': 2,
                       'mesh_vertex_penetration_fraction_mean': value,
                       'mesh_vertex_penetration_depth_mean_m_mean': 0.,
                       'common_contact_slip_speed_mean_m_s_regular_mean': value}]).to_csv(run / 'logs/validation.csv', index=False)
        return run

    def test_paired_seed_statistics_and_zero_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = [self.fixture(root, f'b{s}', s, v) for s, v in [(57, .1), (58, .2)]]
            m = [self.fixture(root, f'm{s}', s, v) for s, v in [(57, .08), (58, .16)]]
            out = compare(b, m, root / 'comparison', axis='optimizer_step_count')
            stats = pd.read_csv(out / 'paired_seed_statistics.csv', header=[0, 1], index_col=[0, 1])
            row = stats.loc[(4, 'mesh_vertex_penetration_fraction_mean')]
            self.assertAlmostEqual(row[('absolute_change', 'mean')], -.03)
            self.assertAlmostEqual(row[('absolute_change', 'std')], np.sqrt(.0002))
            metrics = pd.read_csv(out / 'metrics_comparison.csv')
            self.assertTrue(metrics[metrics.metric.str.contains('depth')].relative_change_percent.isna().all())
            slip = metrics[metrics.metric.str.startswith('common_contact_')]
            self.assertTrue((slip.direction == 'descriptive').all())
            self.assertTrue(slip.relative_improvement_percent.isna().all())
            self.assertTrue((out / 'training_curves/stretching_energy_loss_mean.png').is_file())
            with self.assertRaises(FileExistsError):
                compare(b, m, out)
            with self.assertRaisesRegex(ValueError, 'seed set'):
                pair_runs(b[:1], m)

    def test_resume_clips_parent_and_rejects_changed_controls_or_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.fixture(root, 'baseline', 57, .1)
            m = self.fixture(root, 'modified', 57, .08)
            child = self.fixture(root, 'resume', 57, .08)
            info = json.loads((child / 'metadata.json').read_text())
            info['resume'] = {'parent_run': str(m), 'from_step': 2}
            (child / 'metadata.json').write_text(json.dumps(info))
            parent = json.loads((m / 'metadata.json').read_text())
            parent['initialization_checkpoint'] = {'sha256': 'fine-tune-source'}
            (m / 'metadata.json').write_text(json.dumps(parent))
            self.assertEqual(controls(child)['initialization'], 'fine-tune-source')
            parent.pop('initialization_checkpoint')
            (m / 'metadata.json').write_text(json.dumps(parent))
            pd.DataFrame([{'global_step': 3, 'optimizer_step_count': 9},
                          {'global_step': 4, 'optimizer_step_count': 10}]).to_csv(child / 'logs/train.csv', index=False)
            joined = history(child, 'train')
            self.assertEqual(joined.global_step.tolist(), [1, 2, 3, 4])
            self.assertEqual(joined.optimizer_step_count.tolist(), [1, 2, 9, 10])
            cfg = OmegaConf.load(m / 'config.yaml'); cfg.runner.postcvpr.lr = .002
            OmegaConf.save(cfg, m / 'config.yaml')
            with self.assertRaisesRegex(ValueError, 'hyperparameters'):
                compare([b], [m], root / 'confounded')
            self.assertFalse((root / 'confounded').exists())
            cfg.runner.postcvpr.lr = .001; OmegaConf.save(cfg, m / 'config.yaml')
            (m / 'validation_manifest.json').write_text(json.dumps({'evaluation_signature': 'different'}))
            with self.assertRaisesRegex(ValueError, 'signatures differ'):
                compare([b], [m], root / 'incompatible')
            self.assertFalse((root / 'incompatible').exists())


if __name__ == '__main__':
    unittest.main()
