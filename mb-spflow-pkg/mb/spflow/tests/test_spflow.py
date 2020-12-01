import unittest

import mb.modelbase as mbase
from mb.modelbase import SPFlowModel
from mb.modelbase.utils import data_import

from . import test_allbus


class TestSPFlowModelAllbus(unittest.TestCase):
    def test_generic(self):
        df = test_allbus.categorical_as_strings()
        all_, discrete, continuous = data_import.get_columns_by_dtype(df)
        data = {
            'mixed': df
        }
        models_setup = {

        }

        models = {
            'discrete': [],
            'continuous': [],
            # 'mixed': [EmpiricalModel]
            'mixed': [SPFlowModel]
        }

        def setup(model):
            model.set_spn_type('mspn')
            model.set_var_types(test_allbus.spn_metatypes['philipp'])

        models_setup = {
            ('mixed', SPFlowModel): lambda x: setup(x)
        }

        mbase._test_all(models, models_setup, data, depth=1)


class TestSPFlowModelMPG(unittest.TestCase):
    def test_generic(self):

        raise("Not implemented: add mpg to test suite. see code.")

        df = mpg.cg_4cat3cont(do_not_change=['cylinder'])
        all_, discrete, continuous = data_import.get_columns_by_dtype(df)
        data = {
            'mixed': df
        }
        models_setup = {

        }

        models = {
            'discrete': [],
            'continuous': [],
            'mixed': [SPFlowModel]
        }

        def setup(model):
            model.set_spn_type('mspn')
            model.set_var_types(mpg.spflow_metatypes['version_A'])

        models_setup = {
            ('mixed', SPFlowModel): lambda x: setup(x)
        }

        mbase._test_all(models, models_setup, data, depth=1)


if __name__ == '__main__':
    unittest.main()
