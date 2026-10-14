import csv
import json
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from csv_to_tensorboard import convert
from utils.tensorboard_logging import TrainingTensorBoard


class TensorBoardTest(unittest.TestCase):
    def test_csv_and_live_scalars_match_including_rollout_lr_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            (root / 'metadata.json').write_text(json.dumps({'git_commit': 'fixture'}))
            training = [dict(global_step=1, roll_steps=2, optimized=1,
                             total_physical_loss_mean=3., total_physical_loss_last=4.,
                             inertia_loss_mean=1., inertia_loss_last=2.,
                             friction_load_sum_n_mean=.12, friction_load_sum_n_last=.14)]
            substeps = [dict(global_step=1, lr_used=.1, lr_next=.05, grad_norm_before_clip=3., collision_weight=5000),
                        dict(global_step=1, lr_used=.05, lr_next=.025, grad_norm_before_clip=4., collision_weight=5000)]
            validation = [dict(global_step=1, sequence='a', garment='shirt', frames=10, total_physical_loss_regular_mean=2.),
                          dict(global_step=1, sequence='b', garment='shirt', frames=10, total_physical_loss_regular_mean=4.)]
            for name, rows in [('train', training), ('train_substeps', substeps), ('validation', validation)]:
                with (root / 'logs' / (name + '.csv')).open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            live = TrainingTensorBoard(root / 'live')
            live.training(training[0])
            live.optimizer(substeps)
            live.validation(validation)
            live.close()
            convert(root, root / 'imported')
            def scalars(path):
                events = EventAccumulator(str(path), size_guidance={'scalars': 0}).Reload()
                return {tag: [(s.step, s.value) for s in events.Scalars(tag)] for tag in events.Tags()['scalars']}
            imported = scalars(root / 'imported')
            self.assertEqual(imported, scalars(root / 'live'))
            self.assertAlmostEqual(imported['friction/friction_load_sum_n_mean'][0][1], .12)
            self.assertAlmostEqual(imported['optimizer/lr_used_mean'][0][1], .075)
            self.assertEqual(imported['validation/sequence_mean/total_physical_loss_regular_mean'], [(1, 3.)])
            with self.assertRaises(FileExistsError):
                convert(root, root / 'imported')

    def test_warmup_does_not_fabricate_lr_used(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TrainingTensorBoard(directory)
            writer.optimizer([dict(global_step=1, lr_used='', lr_next=.001, grad_norm_before_clip='', collision_weight=5000)])
            writer.close()
            tags = EventAccumulator(directory).Reload().Tags()['scalars']
            self.assertNotIn('optimizer/lr_used_mean', tags)
            self.assertIn('optimizer/lr_next_last', tags)


if __name__ == '__main__':
    unittest.main()
