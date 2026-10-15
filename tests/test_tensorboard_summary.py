import unittest
import pandas as pd
from tensorboard_summary import validation_groups, CORE

class SummaryTest(unittest.TestCase):
    def test_matched_scopes_never_mix_missing_or_extra_sequences(self):
        rows=[dict(sequence=s,garment='tshirt') for s in CORE]
        frame=pd.DataFrame(rows+[dict(sequence='tshirt_shape00_55_27',garment='tshirt')])
        groups=dict(validation_groups(frame))
        self.assertEqual(len(groups['01_core3_tshirt']),3)
        self.assertEqual(len(groups['02_extra_tshirt_55_27']),1)
        self.assertNotIn('03_longskirt_4motions',groups)
        self.assertEqual(list(validation_groups(pd.DataFrame(rows[:2]))),[])
