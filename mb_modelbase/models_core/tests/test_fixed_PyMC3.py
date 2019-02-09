import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase as mbase
import unittest

# # Specify model
# basic_model = pm.Model()
# with basic_model:
#     sigma = 1
#     mu = pm.Normal('mu', mu=0, sd=sigma)
#     X = pm.Normal('X', mu=mu, sd=sigma, observed=data['X'])
#
# modelname = 'my_pymc3_model'
# mymod = mbase.FixedProbabilisticModel(modelname, basic_model)

#mymod = mbase.Model.load('/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/my_pymc3_model.mdl')
#mymod = mbase.Model.load('/home/guet_jn/Desktop/mb_data/data_models/my_pymc3_model.mdl')


class Testcopy(unittest.TestCase):
    """
    Test the FixedProbabilisticModel.copy() method
    """
    def test_copy_initialized_model(self):
        """
        Test the method on a model that was just initialized, and not changed in any other way yet
        """
        mymod = mbase.Model.load(
            '/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/pymc3_testcase_model.mdl')
        mymod_copy = mymod.copy()
        self.assertEqual(mymod.data.equals(mymod_copy.data),1,"Copied model data is different than original model data")
        self.assertEqual(mymod.test_data.equals(mymod_copy.test_data),1, "Copied model test data is different than original model test data")
        self.assertEqual(mymod.samples.equals(mymod_copy.samples),1, "Copied samples are different than original samples")

class Test_set_data(unittest.TestCase):
    """
    Test the FixedProbabilisticModel._set_data() method
    """
    def test_set_data(self):
        """
        Test if the method gives any data to the model and if the model data is the same as the input data
        """
        mymod = mbase.Model.load(
            '/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/pymc3_testcase_model.mdl')
        np.random.seed(2)
        size = 100
        mu = np.random.normal(0, 1, size=size)
        sigma = 1
        X = np.random.normal(mu, sigma, size=size)
        data = pd.DataFrame({'X': X})
        mymod.set_data(data)
        self.assertEqual(mymod.data.empty,0, "There is no data in the model")
        self.assertEqual(mymod.data.equals(data),1, "model data is different than the original data")



if __name__ == "__main__":
    unittest.main()




