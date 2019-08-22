import unittest
import pandas as pd

from mb_modelbase.models_core import utils


class TestUtils(unittest.TestCase):

    def test_all_numeric(self):
        df_num = pd.DataFrame(data={'A': [1,2,3], 'B':[5.4, 4.2, 1.9]})
        df_partial_num = pd.DataFrame(data={'A': [1, 2, 3], 'B': list('abc')})
        df_empty = pd.DataFrame()
        df_str = pd.DataFrame(data={'A': list("sdfghjkl")})

        self.assertTrue(utils.all_numeric(df_num))
        self.assertFalse(utils.all_numeric(df_partial_num))
        self.assertTrue(utils.all_numeric(df_empty))
        self.assertFalse(utils.all_numeric(df_str))




