import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase as mbase
import unittest


testcasemodel_path = '/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/pymc3_testcase_model.mdl'
#testcasemodel_path = '/home/guet_jn/Desktop/mb_data/data_models/pymc3_testcase_model.mdl'

class Test_methods_on_initialized_model(unittest.TestCase):
    """
    Test the FixedProbabilisticModel methods on a model that has just been initialized
    """
    def testinit(self):
        """
        Test if newly initialized model has data, test data or samples and if mode is set to None
        """
        mymod = mbase.Model.load(testcasemodel_path)
        self.assertEqual(mymod.data.empty,1,"There should be no data")
        self.assertEqual(mymod.test_data.empty, 1, "There should be no test data")
        self.assertEqual(mymod.samples.empty, 1, "There should be no samples")
        self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None")

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        mymod = mbase.Model.load(testcasemodel_path)
        mymod_copy = mymod.copy()
        self.assertEqual(mymod.data.equals(mymod_copy.data),1,"Copied model data is different than original model data")
        self.assertEqual(mymod.test_data.equals(mymod_copy.test_data),1,
                         "Copied model test data is different than original model test data")
        self.assertEqual(mymod.samples.equals(mymod_copy.samples),1,
                         "Copied samples are different than original samples")


    def test_set_data(self):
        """
        Test if the set_data() method gives any data to the model
        and if the model data has the same columns as the input data and if mode is set to data
        """
        mymod = mbase.Model.load(testcasemodel_path)
        np.random.seed(2)
        size = 100
        mu = np.random.normal(0, 1, size=size)
        sigma = 1
        X = np.random.normal(mu, sigma, size=size)
        data = pd.DataFrame({'X': X})
        mymod.set_data(data)
        self.assertEqual(mymod.data.empty,0, "There is no data in the model")
        self.assertEqual(mymod.data.columns.equals(data.columns),1, "model data has different columns than the original data")
        self.assertEqual(mymod.mode,'data', "model mode should be set to data")

    def test_fit(self):
        """
        Test if there are samples and test data in the model and if the mode is set to model
        """
        mymod = mbase.Model.load(testcasemodel_path)
        mymod._fit()
        self.assertEqual(mymod.samples.empty, 0, "There are no samples in the model")
        self.assertEqual(mymod.test_data.empty, 0, "There is no test data in the model")
        self.assertEqual(mymod.mode, 'model', "mode should be set to model")

    def test_marginalizeout(self):
        """
        Call _marginalizeout on a model without any samples.
        An error should be thrown since the model does not yet know any variables
        """
        mymod = mbase.Model.load(testcasemodel_path)
        with self.assertRaises(ValueError):
            mymod._marginalizeout(keep='A', remove='B')
        self.assertEqual(mymod.data.empty,1,"There should be no data")
        self.assertEqual(mymod.test_data.empty, 1, "There should be no test data")
        self.assertEqual(mymod.samples.empty, 1, "There should be no samples")
        self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None")

    def test_conditionout(self):
        """
        Call _conditionout on a model without any samples.
        An error should be thrown since the model does not yet know any variables
        """
        mymod = mbase.Model.load(testcasemodel_path)
        with self.assertRaises(ValueError):
            mymod._conditionout(keep='A',remove='B')
        self.assertEqual(mymod.data.empty, 1, "There should be no data")
        self.assertEqual(mymod.test_data.empty, 1, "There should be no test data")
        self.assertEqual(mymod.samples.empty, 1, "There should be no samples")
        self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None")

    def test_density(self):
        """
        Calculate a probability density on a model. Without samples, it should be zero for every input.
        """
        mymod = mbase.Model.load(testcasemodel_path)
        self.assertEqual(mymod.density(0),0, "density without samples should be zero")
        self.assertEqual(mymod.density(100), 0, "density without samples should be zero")
        self.assertEqual(mymod.density(-100), 0, "density without samples should be zero")
        self.assertEqual(mymod.density([1,1,1]), 0, "density without samples should be zero")

    def test_maximum(self):
        """
        Calculate the maximum probability of a model without samples. It should return an empty array
        """
        mymod = mbase.Model.load(testcasemodel_path)
        self.assertTrue(len(mymod._maximum())==0,
                        "maximum density point for a model without variables should be an empty array")

class Test_methods_on_model_with_data(unittest.TestCase):
    """
    Test the FixedProbabilisticModel methods on a model that has been initialized and given data
    """
    np.random.seed(2)
    size = 100
    mu = np.random.normal(0, 1, size=size)
    sigma = 1
    X = np.random.normal(mu, sigma, size=size)
    data = pd.DataFrame({'X': X})

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        mymod = mbase.Model.load(testcasemodel_path)
        mymod.set_data(data)
        mymod_copy = mymod.copy()
        self.assertEqual(mymod.data.equals(mymod_copy.data),1,"Copied model data is different than original model data")
        self.assertEqual(mymod.test_data.equals(mymod_copy.test_data),1,
                         "Copied model test data is different than original model test data")
        self.assertEqual(mymod.samples.equals(mymod_copy.samples),1,
                         "Copied samples are different than original samples")
    # def test_fit(self):
    #     mymod = mbase.Model.load(testcasemodel_path)
    #     np.random.seed(2)
    #     size = 100
    #     mu = np.random.normal(0, 1, size=size)
    #     sigma = 1
    #     X = np.random.normal(mu, sigma, size=size)
    #     data = pd.DataFrame({'X': X})
    #     mymod.fit(data)
    #     self.assertEqual(mymod.data.empty,0, "There is no data in the model")
    #     self.assertEqual(mymod.test_data.empty, 0, "There is no test data in the model")
    #     self.assertEqual(mymod.samples.empty, 0, "There are no samples in the model")
    #     self.assertEqual(mymod.data.equals(data),1, "model data is different than the original data")

# class Test_methods_on_fitted_model(unittest.TestCase):
#     """
#     Test methods after a model has been given data and fitted
#     """

if __name__ == "__main__":
    unittest.main()









