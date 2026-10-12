import unittest
import numpy as np
from utils.inference_inputs import normalize_smpl_sequence
from utils.rollout_io import rollout_filename


class InferenceInputsTest(unittest.TestCase):
    def test_pose_split_crop_override_and_no_mutation(self):
        pose = np.arange(8 * 72, dtype=np.float32).reshape(8, 72)
        original = pose.copy()
        data, indices, notes = normalize_smpl_sequence({'poses': pose}, [0.1] * 10, start=1, stride=2)
        np.testing.assert_array_equal(indices, [1, 3, 5, 7])
        np.testing.assert_array_equal(data['body_pose'], pose[indices, 3:])
        np.testing.assert_array_equal(data['global_orient'], pose[indices, :3])
        data['body_pose'][:] = 0
        np.testing.assert_array_equal(pose, original)
        np.testing.assert_allclose(data['betas'], 0.1)
        self.assertTrue(notes)

    def test_reject_incompatible_or_invalid_parameters(self):
        for parameters in ({'poses': np.zeros((4, 156))}, {'poses': np.zeros((2, 72))},
                           {'poses': np.full((4, 72), np.nan)},
                           {'poses': np.zeros((4, 72)), 'betas': np.zeros((4, 10))}):
            with self.subTest(parameters=list(parameters)):
                with self.assertRaises(ValueError):
                    normalize_smpl_sequence(parameters)

    def test_names_distinguish_garments_and_prevent_path_escape(self):
        name = rollout_filename('modified_contact_relaxed_barrier', 57, 'tshirt_shape00_01_01', 'longskirt')
        self.assertEqual(name, 'modified_contact_relaxed_barrier_seed57_tshirt_shape00_01_01_longskirt_rollout.npz')
        self.assertNotEqual(name, rollout_filename('modified_contact_relaxed_barrier', 57, 'tshirt_shape00_01_01', 'tshirt'))
        with self.assertRaises(ValueError):
            rollout_filename('experiment', 57, 'sequence', 'garment', '../{sequence}.npz')


if __name__ == '__main__':
    unittest.main()
