import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase as mbase
import unittest
from mb_modelbase.models_core.base import Density, Split


# Generate data and load model
np.random.seed(123)
alpha, sigma = 1, 1
beta_0 = 1
beta_1 = 2.5
size = 100
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2
Y = alpha + beta_0 * X1 + beta_1 * X2 + np.random.randn(size) * sigma
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
testcasemodel_path = '/home/guet_jn/Desktop/mb_data/data_models/pymc3_getting_started_model_independent_vars_fitted.mdl'
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
        self.assertTrue(len(mymod.data['X1']) == 0, "Test data for independent variables should not exist:X1")
        self.assertTrue(len(mymod.data['X2']) == 0, "Test data for independent variables should not exist:X2")

    def test_samples(self):
        """
        Test if samples were drawn for independent variables. There should be no samples for these variables,
        which automatically ensures that no marginal distribution and no density for those variables can be computed.
        """
        self.assertTrue(len(mymod.samples['X1']) == 0, "There should be no samples for independent variables: X1")
        self.assertTrue(len(mymod.samples['X2']) == 0, "There should be no samples for independent variables: X2")

    def test_prediction(self):
        """
        Test if predictions work as intended
        """
        self.assertIsNone(mymod.predict(Density('X1', 'Y'), splitby=Split('Y', 'equiinterval')), 'It should not be possible to predict an independent variable')
        self.assertIsNone(mymod.predict(Density('X1', 'Y'), splitby=Split('X1', 'equiinterval')), 'It should be possible to predict a dependent variable')

    def test_conditioning(self):
        # How is the feature implemented that sets an arbitrary interval for a variable?
        # It is not the conditioning method, I think, because that does not change the model
        pass

if __name__ == "__main__":

    unittest.main()
