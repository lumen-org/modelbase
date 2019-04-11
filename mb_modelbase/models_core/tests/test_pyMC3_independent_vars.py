import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase as mbase
import unittest



# load model
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

    def test_prediction_dependent(self):
        """
        Test if predictions of dependent variables work as intended
        """
        # I imagine that for prediction in the frontend, first all other variables than the ones in the visualization are marginalized out.
        # Then, for each interval of the variable to condition on, the target variable is conditioned on that interval and the maximum density is returned
        # Does the predict-method already do the marginalization and conditioning?? maybe this is specified by where and splitby
        # Ich nehme jetzt mal an, dass das marginalisieren automatisch erledigt wird, und marginalisiere daher nicht vorher
        self.assertTrue(len(mymod.predict(mbase.models_core.base.Density('Y'), splitby=mbase.models_core.base.Split('X1', 'equiinterval'))) > 0, 'It should be possible to predict a dependent variable conditioned on an independent one')
        self.assertTrue(len(mymod.predict(mbase.models_core.base.Density('Y'), splitby=mbase.models_core.base.Split('alpha', 'equiinterval'))) > 0, 'It should be possible to predict a dependent variable conditioned on another dependent variable')

    def test_prediction_independent(self):
        """
        Test if predictions of independent variables work as intended
        """
        self.assertTrue(len(mymod.predict(mbase.models_core.base.Density('X1'), splitby=mbase.models_core.base.Split('Y', 'equiinterval'))) == 0,
                        'It should not be possible to predict an independent variable conditioned on a dependent one')
        self.assertTrue(len(mymod.predict(mbase.models_core.base.Density('X1'), splitby=mbase.models_core.base.Split('X2', 'equiinterval'))) == 0,
                        'It should not be possible to predict an independent variable conditioned on another independent variable')

    def test_conditioning(self):
        # How is the feature implemented that sets an arbitrary interval for a variable?
        # It is not the conditioning method, I think, because that does not change the model
        pass

if __name__ == "__main__":

    unittest.main()
