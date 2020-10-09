import unittest

import numpy as np

import mb.modelbase as mbase

# Load model. The model first has to be created by create_PyMC3_testmodels.py
testcasemodel_path = 'models/'
testcasedata_path = 'data/'

mymod = mbase.Model.load(testcasemodel_path)

class Test(unittest.TestCase):

    def test_data(self):
        """
        Test if data for independent variables exists
        """
        self.assertTrue(len(mymod.data['X1']) > 0, "Data for independent variables should exist:X1")
        self.assertTrue(len(mymod.data['X2']) > 0, "Data for independent variables should exist:X2")


    def test_prediction_dependent(self):
        """
        Test if predictions of dependent variables work as intended
        """
        self.assertTrue(len(mymod.predict(mbase.Aggregation('Y'),
                                          splitby=mbase.Split('X1', 'equiinterval'))) > 0,
                        'It should be possible to predict a dependent variable conditioned on an independent one')
        self.assertTrue(len(mymod.predict(mbase.Aggregation('Y'),
                                          splitby=mbase.Split('alpha', 'equiinterval'))) > 0,
                        'It should be possible to predict a dependent variable conditioned on another dependent variable')

    def test_prediction_independent(self):
        """
        Test if predictions of independent variables work as intended
        """
        self.assertTrue(mymod.predict(mbase.Aggregation('X1'),
                                      splitby=mbase.Split('Y', 'equiinterval')).isnull().all()[0],
                        'There should be no predictions for an independent variable conditioned on a dependent one')
        self.assertTrue(mymod.predict(mbase.Aggregation('X1'),
                                      splitby=mbase.Split('X2', 'equiinterval')).isnull().all()[0],
                        'There should be no predictions for an independent variable conditioned on a dependent one')

    def test_compare_full_data_with_samples(self):
        for var in mymod.data:
            if var in mymod.model_structure.free_RVs:
                self.assertAlmostEqual(np.mean(mymod.data[var]), np.mean(mymod.samples[var]), 0,
                                       'Mean of data and posterior samples for ' + var + ' should be similar')
                self.assertAlmostEqual(np.var(mymod.data[var]), np.var(mymod.samples[var]), 0,
                                       'Variance of data and posterior samples for ' + var + ' should be similar')

    # This test fails at X1 = -3.31, even if the program works correctly, I think
    def test_compare_conditioned_data_with_samples(self):
        covariates = [field['name'] for field in mymod.fields if field['independent']]
        variates = [var for var in mymod.data if var not in covariates]
        for covariate in covariates:
            for variate in variates:
                # condition on each unique value of the covariates in samples
                for val in mymod.samples[covariate].unique():
                    conditioned_posterior_samples = mymod.samples[variate][mymod.samples[covariate] == val]
                    conditioned_data = mymod.data[variate][(mymod.data[covariate] <= val + 1) & (mymod.data[covariate] >= val - 1)]
                    self.assertAlmostEqual(np.mean(conditioned_data), np.mean(conditioned_posterior_samples), 0,
                                           'Mean of conditioned data and samples should be similar. Variate: ' +
                                           variate + ' Covariate: ' + covariate + ' covariate_value: '+ str(val))
                    self.assertAlmostEqual(np.var(conditioned_data), np.var(conditioned_posterior_samples), 0,
                                           'Variance of conditioned data and samples should be similar. Variate: ' +
                                           variate + ' Covariate: ' + covariate + ' covariate_value: '+ str(val))

if __name__ == "__main__":
    unittest.main()
