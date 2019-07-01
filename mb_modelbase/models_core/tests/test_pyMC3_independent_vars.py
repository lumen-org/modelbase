import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase as mbase
import unittest
from run_conf import cfg as user_cfg


# Load model. The model first has to be created by create_PyMC3_testmodels.py

testcasemodel_path = user_cfg['modules']['modelbase']['test_model_directory'] + \
                     '/pymc3_getting_started_model_independent_vars_fitted.mdl'
mymod = mbase.Model.load(testcasemodel_path)


class Test(unittest.TestCase):

    def test_data(self):
        """
        Test if data for independent variables exists
        """
        self.assertTrue(len(mymod.data['X1']) > 0, "Data for independent variables should exist:X1")
        self.assertTrue(len(mymod.data['X2']) > 0, "Data for independent variables should exist:X2")

    def test_test_data(self):
        """
        Test if test data for independent variables exists
        """
        self.assertTrue(mymod.test_data['X1'].isnull().all(), "Test data for independent variables should not exist:X1")
        self.assertTrue(mymod.test_data['X2'].isnull().all(), "Test data for independent variables should not exist:X2")

    def test_prediction_dependent(self):
        """
        Test if predictions of dependent variables work as intended
        """
        self.assertTrue(len(mymod.predict(mbase.models_core.base.Aggregation('Y'),
                                          splitby=mbase.models_core.base.Split('X1', 'equiinterval'))) > 0,
                        'It should be possible to predict a dependent variable conditioned on an independent one')
        self.assertTrue(mymod.predict(mbase.models_core.base.Aggregation('Y'),
                                         splitby=mbase.models_core.base.Split('X1', 'equiinterval')).isnull().values.all(),
                       'prediction of a dependent variable conditioned on an independent one should not contain only NaNs')
        self.assertTrue(len(mymod.predict(mbase.models_core.base.Aggregation('Y'),
                                          splitby=mbase.models_core.base.Split('alpha', 'equiinterval'))) > 0,
                        'It should be possible to predict a dependent variable conditioned on another dependent variable')
        self.assertTrue(mymod.predict(mbase.models_core.base.Aggregation('Y'),
                                          splitby=mbase.models_core.base.Split('alpha', 'equiinterval')).isnull().values.all(),
                        'prediction of a dependent variable conditioned on another dependent one should not contain only NaNs')

    def test_prediction_independent(self):
        """
        Test if predictions of independent variables work as intended
        """
        self.assertTrue(mymod.predict(mbase.models_core.base.Aggregation('X1'),
                                      splitby=mbase.models_core.base.Split('Y', 'equiinterval')).isnull().all()[0],
                        'There should be no predictions for an independent variable conditioned on a dependent one')
        self.assertTrue(mymod.predict(mbase.models_core.base.Aggregation('X1'),
                                      splitby=mbase.models_core.base.Split('X2', 'equiinterval')).isnull().all()[0],
                        'There should be no predictions for an independent variable conditioned on a dependent one')

    def test_conditioning(self):
        # How is the feature implemented that sets an arbitrary interval for a variable?
        # It is not the conditioning method, I think, because that does not change the model
        pass

if __name__ == "__main__":

    unittest.main()
