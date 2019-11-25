import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal

from mb_modelbase.utils import data_import_utils


class MyTestCase(unittest.TestCase):
    def test_to_string_cols(self):

        df = pd.DataFrame({'A': list(range(5)), 'B': list('ABCDE')})
        df2 = data_import_utils.to_string_cols(df, inplace=False)
        df3 = data_import_utils.to_string_cols(df, columns=['A'], inplace=False)
        assert_frame_equal(df2, df3)
        self.assertTrue(np.all(df2.applymap(lambda e: type(e) == str)))
        self.assertTrue(np.all(df3.applymap(lambda e: type(e) == str)))


if __name__ == '__main__':
    unittest.main()
