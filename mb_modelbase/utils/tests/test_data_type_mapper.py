import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
from mb_modelbase.utils import data_type_mapper as dtm


class DataTypeMapperTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dtmapper = dtm.DataTypeMapper()
        dtmapper.set_map('sex', forward={'Male': 1, 'Female': 2}, backward='auto')
        dtmapper.set_map('B', forward={100: 'x', 200: 'y'}, backward='auto')
        cls._dtmmapper = dtmapper

    def test_dataframe_mapping(self):
        mapper = self._dtmmapper
        df = pd.DataFrame(data={'sex': ['Male', 'Female', 'Male'], 'B': [100, 200, 100]})

        df_conv = mapper.forward(df, inplace=False)
        df_conv_ref = pd.DataFrame(data={'sex': [1, 2, 1], 'B': ['x', 'y', 'x']})
        assert_frame_equal(df_conv_ref, df_conv)

        df_back_conv = mapper.backward(df_conv, inplace=False)
        assert_frame_equal(df, df_back_conv)

    def test_series_mapping(self):
        mapper = self._dtmmapper
        series = pd.Series(data=['Male', 'Female'], name='sex')

        series_conv = mapper.forward(series, inplace=False)
        series_conv_ref = pd.Series(data=[1, 2], name='sex')
        assert_series_equal(series_conv, series_conv_ref)

        series_back_conv  = mapper.backward(series_conv, inplace=False)
        assert_series_equal(series, series_back_conv)

    def test_dict_mapping(self):
        mapper = self._dtmmapper
        d = {'sex': 'Male'}

        d_conv = mapper.forward(d, inplace=False)
        d_conv_ref = {'sex': 1}
        self.assertEqual(d_conv, d_conv_ref)

        d_back_conv  = mapper.backward(d_conv, inplace=False)
        self.assertEqual(d, d_back_conv)

        d = {'foobar': 'bhgjdas'}
        d_conv = mapper.forward(d, inplace=False)
        self.assertEqual(d, d_conv)

    def test_dict_mapping_missing_map(self):
        mapper = self._dtmmapper
        d = {'does_not_exist_in_map': 666, 'ABC': 42.01}
        d_conv = mapper.forward(d, inplace=False)
        self.assertEqual(d_conv, d)

    def test_copy(self):
        mapper = self._dtmmapper
        mapper_copy = mapper.copy()

        df = pd.DataFrame(data={'sex': ['Male', 'Female', 'Male'], 'B': [100, 200, 100]})

        df_conv = mapper_copy.forward(df, inplace=False)
        df_conv_ref = pd.DataFrame(data={'sex': [1, 2, 1], 'B': ['x', 'y', 'x']})
        assert_frame_equal(df_conv_ref, df_conv)

        df_back_conv = mapper_copy.backward(df_conv, inplace=False)
        assert_frame_equal(df, df_back_conv)


if __name__ == '__main__':
    unittest.main()
