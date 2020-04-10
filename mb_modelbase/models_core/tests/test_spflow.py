import unittest
import mb_modelbase.models_core.tests.test_models_generic as tmg
from mb_data.allbus import allbus
from mb_data.mpg import mpg
from mb_modelbase.utils import data_import_utils
from mb_modelbase.models_core.spflow import SPNModel

class TestSPFlowModelAllbus(unittest.TestCase):
    def test_generic(self):
        df = allbus.categorical_as_strings()
        all_, discrete, continuous = data_import_utils.get_columns_by_dtype(df)
        data = {
            'mixed': df
        }
        models_setup = {

        }

        models = {
            'discrete': [],
            'continuous': [],
            # 'mixed': [EmpiricalModel]
            'mixed': [SPNModel]
        }

        def setup(model):
            model.set_spn_type('mspn')
            model.set_var_types(allbus.spn_metatypes['philipp'])

        models_setup = {
            ('mixed', SPNModel): lambda x: setup(x)
        }

        tmg._test_all(models, models_setup, data, depth=1)

class TestSPFlowModelMPG(unittest.TestCase):
    def test_generic(self):
        df = mpg.cg_4cat3cont(do_not_change=['cylinder'])
        all_, discrete, continuous = data_import_utils.get_columns_by_dtype(df)
        data = {
            'mixed': df
        }
        models_setup = {

        }

        models = {
            'discrete': [],
            'continuous': [],
            'mixed': [SPNModel]
        }

        def setup(model):
            model.set_spn_type('mspn')
            model.set_var_types(mpg.spflow_metatypes['version_A'])

        models_setup = {
            ('mixed', SPNModel): lambda x: setup(x)
        }

        tmg._test_all(models, models_setup, data, depth=1)

if __name__ == '__main__':
    unittest.main()
